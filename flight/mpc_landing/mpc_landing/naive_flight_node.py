"""naive_flight_node — the plainest possible takeoff-and-land over MAVROS.

This is the deliberately dumb sibling of `mpc_landing_node`: no marker, no
vision, no MPC, no search.  It exists to prove the boring half of the stack on a
real airframe FIRST — that MAVROS is wired, that PX4 accepts OFFBOARD, that the
vehicle climbs to a height, holds, and comes back down under the autopilot's own
LAND — before any of the perception is trusted with the descent.

    PRECHECK ──approve──► ARM ──► TAKEOFF ──► HOVER ──► LAND ──► DONE

Only the ARM step waits for a human; everything after it runs on its own.  Once
the operator has authorised a naive takeoff there is nothing more to decide —
the profile is fixed (climb to `takeoff_alt_m`, hold `hover_s`, then hand to the
autopilot's LAND), so pausing for a second approval at altitude would only leave
a vehicle hovering while it waits for a keystroke.

    ros2 run mpc_landing naive_flight_node          # ENTER approves at the gate

Under `ros2 launch` stdin is not a terminal, so approve over the service instead:

    ros2 run mpc_landing approve naive_flight_node
    ros2 run mpc_landing abort   naive_flight_node  # land now, from any phase

The MAVROS discipline here is lifted verbatim from `mpc_landing_node` because it
is the part that is easy to get subtly wrong on PX4 and was already paid for
there: BEST_EFFORT sensor QoS (a RELIABLE subscriber gets nothing from MAVROS),
the stream→mode→arm ORDER (PX4 refuses OFFBOARD until setpoints are already
flowing), keeping the setpoint stream alive through the gate so the vehicle does
not fall out of offboard while armed, and confirming every state change from
telemetry rather than believing the service reply.

Interfaces
----------
Subscribes
    /mavros/state                          mavros_msgs/State
    /mavros/local_position/pose            geometry_msgs/PoseStamped
    /mavros/extended_state                 mavros_msgs/ExtendedState
    /mavros/battery                        sensor_msgs/BatteryState
Publishes
    /mavros/setpoint_raw/local             mavros_msgs/PositionTarget
    ~/state                                std_msgs/String
Services (offered)
    ~/approve, ~/abort                     std_srvs/Trigger
Services (called)
    /mavros/set_mode, /mavros/cmd/arming, /mavros/cmd/land

ALL PARAMETERS ARE DECLARED HERE, IN `_declare`, WITH THEIR VALUES — the same
rule as the rest of flight/.  Override one for a one-off:

    ros2 run mpc_landing naive_flight_node --ros-args -p takeoff_alt_m:=3.0

When a preflight check is in the way and the operator has eyes on the airframe,
`skip_preflight` waives all of them except local position (see `_preflight_ok`
for why that one cannot be waived):

    ros2 run mpc_landing naive_flight_node --ros-args -p skip_preflight:=true
"""

from __future__ import annotations

import sys
import threading
from enum import Enum

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from sensor_msgs.msg import BatteryState
from std_msgs.msg import String
from std_srvs.srv import Trigger

from mavros_msgs.msg import ExtendedState, PositionTarget, State
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode


class Phase(str, Enum):
    PRECHECK = 'PRECHECK'          # running preflight checks
    READY_TO_ARM = 'READY_TO_ARM'  # checks passed, waiting for the one approval
    ARMING = 'ARMING'              # stream -> OFFBOARD -> arm
    TAKEOFF = 'TAKEOFF'            # climbing to takeoff_alt
    HOVER = 'HOVER'                # holding at altitude for hover_s
    LAND = 'LAND'                  # handed to the autopilot's LAND, disarming
    DONE = 'DONE'


def _sensor_qos() -> QoSProfile:
    """MAVROS publishes telemetry BEST_EFFORT; a RELIABLE subscriber gets nothing."""
    return QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                      durability=DurabilityPolicy.VOLATILE,
                      history=HistoryPolicy.KEEP_LAST, depth=5)


class NaiveFlightNode(Node):
    def __init__(self):
        super().__init__('naive_flight_node')
        self._declare()
        self._read_params()

        self.phase = Phase.PRECHECK
        self.state: State | None = None
        self.pose: PoseStamped | None = None
        self.ext: ExtendedState | None = None
        self.batt: BatteryState | None = None
        self._t_phase = self._now()
        self._t_prestream: float | None = None
        self._t_hover: float | None = None
        self._t_calls: dict[str, float] = {}
        self._announced = ''
        self._prompted = ''
        self._checks_logged = False
        self._waived: set[str] = set()

        self.create_subscription(State, '/mavros/state', self._on_state,
                                 _sensor_qos())
        self.create_subscription(PoseStamped, '/mavros/local_position/pose',
                                 self._on_pose, _sensor_qos())
        self.create_subscription(ExtendedState, '/mavros/extended_state',
                                 self._on_ext, _sensor_qos())
        self.create_subscription(BatteryState, '/mavros/battery',
                                 self._on_batt, _sensor_qos())

        self.sp_pub = self.create_publisher(PositionTarget,
                                            '/mavros/setpoint_raw/local', 10)
        self.state_pub = self.create_publisher(String, '~/state', 10)

        self.mode_cli = self.create_client(SetMode, '/mavros/set_mode')
        self.arm_cli = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.land_cli = self.create_client(CommandTOL, '/mavros/cmd/land')

        self.create_service(Trigger, '~/approve', self._on_approve)
        self.create_service(Trigger, '~/abort', self._on_abort)

        # A daemon thread blocks on stdin so the control loop never does — a node
        # that stopped publishing setpoints to wait for a keystroke would drop
        # out of offboard.
        self._stdin_ok = self.interactive and sys.stdin is not None \
            and sys.stdin.isatty()
        if self._stdin_ok:
            threading.Thread(target=self._stdin_loop, daemon=True).start()

        self.create_timer(1.0 / self.rate_hz, self._tick)
        self.get_logger().info(
            f'naive_flight_node: PX4 mode={self.mode_name} | '
            f'takeoff {self.takeoff_alt:.1f} m | hover {self.hover_s:.0f} s | '
            f'then LAND')
        if self.skip_preflight:
            self.get_logger().warn(
                'skip_preflight IS ON — link, battery and armed-state checks '
                'are waived; only local position still gates the ARM prompt')
        if self._stdin_ok:
            self.get_logger().info(
                'ARM will ask on this terminal — ENTER approves, n aborts')
        else:
            self.get_logger().info(
                f'stdin is not a terminal, so approve over the service: '
                f'ros2 run mpc_landing approve {self.get_name()}')

    # ------------------------------------------------------------- parameters
    def _declare(self) -> None:
        """THE one place any of these numbers may be set."""
        p = self.declare_parameter
        p('takeoff_alt_m', 5.0)             # target altitude above takeoff point
        p('alt_tolerance_m', 0.3)           # counts as "reached" within this
        p('climb_speed_m_s', 0.7)           # TAKEOFF climb cap
        p('hover_s', 5.0)                   # hold at altitude this long, then land
        # --- preflight thresholds
        p('min_battery_v', 14.0)            # 4S nominal; raise for 6S
        p('require_battery', True)          # false only for bench tests
        # Waive the preflight checks the operator is allowed to overrule (link,
        # battery, already-armed). NOT local position — see `_preflight_ok`.
        p('skip_preflight', False)
        # --- flight controller. PX4 assumed (OFFBOARD); ArduPilot works too.
        p('offboard_mode', 'OFFBOARD')
        # PX4 wants a steady setpoint stream before it will grant OFFBOARD, not a
        # single message; 1 s at rate_hz is ~20 of them.
        p('offboard_prestream_s', 1.0)
        p('rate_hz', 20.0)
        # Prompt on the terminal at the ARM gate. Falls back to the service when
        # stdin is not a terminal (e.g. under `ros2 launch`).
        p('interactive_approval', True)

    def _read_params(self) -> None:
        g = self.get_parameter
        self.takeoff_alt = float(g('takeoff_alt_m').value)
        self.alt_tol = float(g('alt_tolerance_m').value)
        self.climb_speed = float(g('climb_speed_m_s').value)
        self.hover_s = float(g('hover_s').value)
        self.min_batt = float(g('min_battery_v').value)
        self.require_batt = bool(g('require_battery').value)
        self.skip_preflight = bool(g('skip_preflight').value)
        self.mode_name = str(g('offboard_mode').value)
        self.prestream_s = float(g('offboard_prestream_s').value)
        self.rate_hz = float(g('rate_hz').value)
        self.interactive = bool(g('interactive_approval').value)

    # -------------------------------------------------------------- callbacks
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_state(self, m): self.state = m

    def _on_pose(self, m): self.pose = m

    def _on_ext(self, m): self.ext = m

    def _on_batt(self, m): self.batt = m

    # ------------------------------------------------------------ terminal UI
    def _stdin_loop(self) -> None:
        """Read the one approval from the terminal. ENTER = yes, n = abort."""
        for line in sys.stdin:
            answer = line.strip().lower()
            if answer.startswith('n'):
                self._abort('operator declined at the prompt')
                print('\n  ABORTING — landing.\n', flush=True)
                continue
            if self.phase is Phase.READY_TO_ARM:
                self._approve()
                print('\n  OK: arming and taking off.\n', flush=True)
            else:
                self.get_logger().warn(
                    f'ignored "{answer or "ENTER"}" — nothing waiting for '
                    f'approval (phase {self.phase.value})')

    def _prompt(self) -> None:
        if not self._stdin_ok or self.phase is not Phase.READY_TO_ARM:
            return
        if self._prompted == Phase.READY_TO_ARM.value:
            return
        self._prompted = Phase.READY_TO_ARM.value
        print(f'\n{"=" * 72}\n  preflight PASSED — approve to ARM and take off '
              f'to {self.takeoff_alt:.1f} m\n{"=" * 72}\n'
              f'  proceed?  [ENTER = yes / n = abort]  ', end='', flush=True)

    # ---------------------------------------------------------------- services
    def _approve(self) -> tuple[bool, str]:
        if self.phase is not Phase.READY_TO_ARM:
            return False, (f'nothing to approve — phase is {self.phase.value}, '
                           f'which is not a gate')
        self._to(Phase.ARMING)
        return True, 'approved: arming and taking off'

    def _abort(self, reason: str) -> None:
        """Land now, from any phase. The naive stack's only emergency action."""
        if self.phase in (Phase.DONE, Phase.LAND):
            return
        self.get_logger().warn(f'abort: {reason} — landing')
        self._to(Phase.LAND)

    def _on_approve(self, _req, res):
        ok, msg = self._approve()
        res.success, res.message = ok, msg
        (self.get_logger().info if ok else self.get_logger().warn)(msg)
        return res

    def _on_abort(self, _req, res):
        self._abort('operator aborted')
        res.success, res.message = True, 'aborting: landing'
        return res

    # ---------------------------------------------------------------- helpers
    def _to(self, phase: Phase) -> None:
        if phase is self.phase:
            return
        self.get_logger().info(f'{self.phase.value} -> {phase.value}')
        self.phase = phase
        self._t_phase = self._now()

    def _call(self, client, request, name) -> bool:
        """Fire-and-log a MAVROS service call, async so the timer never blocks."""
        if not client.service_is_ready():
            self.get_logger().error(
                f"service '{name}' not available — is MAVROS running?",
                throttle_duration_sec=5.0)
            return False
        future = client.call_async(request)
        future.add_done_callback(
            lambda f, n=name: self.get_logger().info(f'{n} -> {f.result()}'))
        return True

    def _call_throttled(self, client, request, name, period=1.0) -> None:
        """Like `_call`, but at most once per `period` s, and only if it fired.

        The LAND handler runs every tick; without this it would re-send the land
        command at rate_hz. The throttle timestamp is recorded ONLY on a call
        that actually went out, so a land command refused because the service is
        not up yet is retried on the next tick rather than blocked for a second.
        """
        if self._now() - self._t_calls.get(name, 0.0) < period:
            return
        if self._call(client, request, name):
            self._t_calls[name] = self._now()

    def _alt(self) -> float:
        return float(self.pose.pose.position.z) if self.pose else float('nan')

    def _on_ground(self) -> bool:
        """True once the vehicle has actually settled — never a geometric guess.

        Uses extended_state (the FCU's own land detector), or an already-disarmed
        FCU, so the disarm/finish waits for real ground contact. AUTO.LAND
        normally auto-disarms on the ground; this just confirms it.
        """
        if self.state and not self.state.armed:
            return True
        return (self.ext is not None
                and self.ext.landed_state == ExtendedState.LANDED_STATE_ON_GROUND)

    def _send(self, vx: float, vy: float, vz: float) -> None:
        """Stream a velocity setpoint in the local frame.

        Velocity, not position: takeoff and hover are regulation against a height
        and a hold point, and a position setpoint would re-inject the estimator's
        drift as a command. The FORCE bit is deliberately NOT set — it would
        reinterpret the (ignored) acceleration fields as a force, which PX4 does
        not support on this path and may reject.
        """
        m = PositionTarget()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = 'map'
        m.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        m.type_mask = (PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY
                       | PositionTarget.IGNORE_PZ
                       | PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY
                       | PositionTarget.IGNORE_AFZ
                       | PositionTarget.IGNORE_YAW_RATE)
        m.velocity.x, m.velocity.y, m.velocity.z = float(vx), float(vy), float(vz)
        m.yaw = 0.0
        self.sp_pub.publish(m)

    # ------------------------------------------------------------- preflight
    def _preflight_ok(self) -> bool:
        """Minimal, naive: link up, EKF ready, disarmed, and (opt) battery.

        `skip_preflight` waives every check here EXCEPT local position, which is
        not a policy judgement but a physical prerequisite: TAKEOFF regulates on
        `pose.z`, so with no pose `_alt()` is NaN, the climb setpoint is NaN, and
        PX4 discards it — waiving that one would arm the vehicle and then sit
        there at zero velocity, which is worse than refusing. Everything else is
        a call the operator standing next to the airframe is allowed to make, so
        it is waived loudly (each reason is logged once) rather than silently.
        """
        reasons, waived = [], []
        if self.pose is None:
            reasons.append('no local position — EKF not ready')

        overridable = []
        if not (self.state and self.state.connected):
            overridable.append('no FCU link (/mavros/state)')
        if self.state is None:
            overridable.append('no /mavros/state')
        elif self.state.armed:
            overridable.append('already ARMED — refusing to take over a live vehicle')
        if self.require_batt:
            v = float(self.batt.voltage) if self.batt else 0.0
            if self.batt is None:
                overridable.append('no /mavros/battery')
            elif v < self.min_batt:
                overridable.append(f'battery {v:.1f} V < {self.min_batt:.1f} V')
        (waived if self.skip_preflight else reasons).extend(overridable)

        for r in waived:
            if r not in self._waived:
                self._waived.add(r)
                self.get_logger().warn(f'preflight WAIVED by skip_preflight: {r}')
        if reasons:
            if self._now() - self._t_phase > 5.0:
                self._t_phase = self._now()
                for r in reasons:
                    self.get_logger().warn(f'preflight blocked: {r}')
            return False
        if not self._checks_logged:
            self._checks_logged = True
            self.get_logger().info('preflight PASSED')
        return True

    # ------------------------------------------------------------------- loop
    def _tick(self) -> None:
        self._publish_state()

        # Keep the offboard stream alive, unconditionally, for every phase that
        # is armed or about to be — BEFORE the phase logic, so a phase that
        # returns early (the gate waiting on a human, ARMING waiting out the
        # pre-stream) cannot starve it. PX4 drops offboard after ~0.5 s of
        # silence. Phases that fly a real setpoint overwrite this later in the
        # same tick; publishing twice is harmless, a gap is not.
        if self.phase in (Phase.READY_TO_ARM, Phase.ARMING, Phase.TAKEOFF,
                          Phase.HOVER):
            self._send(0.0, 0.0, 0.0)

        if self.phase is Phase.PRECHECK:
            if self._preflight_ok():
                self._to(Phase.READY_TO_ARM)
            return

        if self.phase is Phase.READY_TO_ARM:
            self._announce()
            return

        if self.phase is Phase.ARMING:
            # ORDER MATTERS ON PX4: stream -> mode -> arm. The stream is kept up
            # above; lead with it, then request the mode, then arm only once the
            # FCU confirms it is IN the mode (arming first and switching after is
            # the ArduPilot habit; PX4 can arm in whatever mode it was in).
            if self._t_prestream is None:
                self._t_prestream = self._now()
                self.get_logger().info(
                    f'streaming setpoints for {self.prestream_s:.1f} s before '
                    f'requesting {self.mode_name}')
                return
            if self._now() - self._t_prestream < self.prestream_s:
                return
            if self.state and self.state.mode != self.mode_name:
                req = SetMode.Request()
                req.custom_mode = self.mode_name
                self._call_throttled(self.mode_cli, req, 'set_mode')
                return
            if self.state and self.state.armed:
                self._to(Phase.TAKEOFF)
                return
            req = CommandBool.Request()
            req.value = True
            self._call_throttled(self.arm_cli, req, 'arming')
            return

        if self.phase is Phase.TAKEOFF:
            err = self.takeoff_alt - self._alt()
            if abs(err) <= self.alt_tol:
                self._send(0.0, 0.0, 0.0)
                self._to(Phase.HOVER)
                return
            # Climb at a capped rate, easing off near the target so the vehicle
            # settles instead of overshooting.
            vz = float(np.clip(err, -self.climb_speed, self.climb_speed))
            self._send(0.0, 0.0, vz)
            return

        if self.phase is Phase.HOVER:
            self._send(0.0, 0.0, 0.0)
            if self._t_hover is None:
                self._t_hover = self._now()
            if self._now() - self._t_hover >= self.hover_s:
                self._to(Phase.LAND)
            return

        if self.phase is Phase.LAND:
            # Hand to the autopilot's own landing, then disarm ONLY once it has
            # actually settled (extended_state), so motors are never cut in the
            # air. AUTO.LAND normally auto-disarms; the explicit disarm is a
            # gated backstop. Both are throttled: this runs every tick.
            self._call_throttled(self.land_cli, CommandTOL.Request(), 'land')
            if self._on_ground():
                req = CommandBool.Request()
                req.value = False
                self._call_throttled(self.arm_cli, req, 'disarm')
            if self.state and not self.state.armed:
                self._to(Phase.DONE)
                self.get_logger().info('disarmed — naive flight complete')
            return

    # ------------------------------------------------------------------ output
    def _announce(self) -> None:
        self._prompt()
        if self._stdin_ok:
            # The prompt already asked on its own clean lines; logging the same
            # thing here would print a timestamped duplicate across the line the
            # operator is typing into.
            return
        if self._announced == Phase.READY_TO_ARM.value:
            return
        self._announced = Phase.READY_TO_ARM.value
        self.get_logger().warn(
            f'>>> WAITING FOR APPROVAL — approve to ARM and take off\n'
            f'    ros2 run mpc_landing approve {self.get_name()}')

    def _publish_state(self) -> None:
        self.state_pub.publish(String(data=self.phase.value))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = NaiveFlightNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

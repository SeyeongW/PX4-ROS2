"""gimbal_teleop — drive the gimbal with WASD and watch the angles move.

    w / s      pitch up / down
    a / d      yaw left / right
    r          recentre
    [ / ]      smaller / larger step
    q or Ctrl-C to quit

The commanded target and the gimbal's REPORTED angle are shown side by side,
with the error between them, because those are three different things and only
the reported one is real:

    target  what we asked for, clamped to the gimbal's travel
    actual  what it says it is doing (SIYI 0x0D)
    err     how far behind it is — large while slewing, ~0 once settled

WHAT FRAME THESE ANGLES ARE IN
------------------------------
Measured on this airframe with `gimbal_monitor`: the A8 mini is EARTH-referenced
in pitch and roll. Tilting the aircraft does not change the reported pitch — the
gimbal holds its angle against the world and reports that. So `SET_ANGLE`
targets are world angles too, not joint angles, and -90 means "down" no matter
what the airframe is doing.

Yaw is the open question. Pitch and roll have gravity to reference; yaw does
not, so its zero is whatever the gimbal decided at power-on. Rotate the airframe
in place while watching the yaw column to see whether it holds a world heading
or follows the body — this tool is a convenient way to do exactly that.

KEYS ARE READ RAW, so there is no Enter and no line buffering. The terminal is
restored on exit, including on Ctrl-C.
"""

from __future__ import annotations

import sys
import termios
import threading
import tty

import rclpy
from rclpy.node import Node

from . import protocol as siyi
from . import siyi_commands as cmds
from . import transport as tp

HELP = ('  w/s pitch   a/d yaw   r recentre   [ ] step   q quit')


class GimbalTeleop(Node):
    def __init__(self):
        super().__init__('gimbal_teleop')
        p = self.declare_parameter
        p('transport', 'serial')
        p('serial_device', '/dev/ttyTHS1')
        p('serial_baud', 115200)
        p('gimbal_host', '192.168.144.25')
        p('gimbal_port', 37260)
        p('step_deg', 5.0)          # per keypress
        p('poll_hz', 10.0)

        g = self.get_parameter
        try:
            self.link = tp.make(str(g('transport').value),
                                host=str(g('gimbal_host').value),
                                port=int(g('gimbal_port').value),
                                device=str(g('serial_device').value),
                                baud=int(g('serial_baud').value))
        except (RuntimeError, ValueError) as exc:
            self.get_logger().fatal(str(exc))
            raise SystemExit(1) from None

        self.step = float(g('step_deg').value)
        self.target_yaw = 0.0
        self.target_pitch = 0.0
        self.actual: tuple[float, float, float] | None = None
        self._seq = 0
        self._quit = False

        # Raw keys, restored on the way out no matter how we leave.
        self._fd = sys.stdin.fileno()
        self._saved = termios.tcgetattr(self._fd) if sys.stdin.isatty() else None
        if self._saved is not None:
            tty.setcbreak(self._fd)
        else:
            print('  stdin is not a terminal — keys will not work here. '
                  'Run this with `ros2 run`, in a real terminal.')

        threading.Thread(target=self._keys, daemon=True).start()
        hz = float(g('poll_hz').value)
        self.create_timer(1.0 / hz, self._poll)
        self.create_timer(0.05, self._drain)
        self.create_timer(0.1, self._draw)

        print(f'\n  gimbal_teleop — {self.link.description}')
        print(f'  travel: pitch {cmds.A8_PITCH_MIN_DEG:.0f}..'
              f'{cmds.A8_PITCH_MAX_DEG:.0f}, yaw {cmds.A8_YAW_MIN_DEG:.0f}..'
              f'{cmds.A8_YAW_MAX_DEG:.0f} deg')
        print(HELP + '\n')
        self._send_target()

    # ------------------------------------------------------------------ input
    def _keys(self) -> None:
        while not self._quit:
            try:
                ch = sys.stdin.read(1)
            except Exception:
                return
            if not ch:
                return
            k = ch.lower()
            if k == 'q':
                self._quit = True
                rclpy.try_shutdown()
                return
            if k == 'r':
                self.target_yaw = self.target_pitch = 0.0
                # CENTER rather than SET_ANGLE(0,0): it is the gimbal's own
                # notion of home, which is what "put it back" should mean.
                self._tx(siyi.encode(siyi.CENTER, bytes([cmds.TRIGGER]),
                                     self._next()))
                continue
            if k == '[':
                self.step = max(0.5, self.step / 2.0)
                continue
            if k == ']':
                self.step = min(45.0, self.step * 2.0)
                continue
            if k == 'w':
                self.target_pitch += self.step
            elif k == 's':
                self.target_pitch -= self.step
            elif k == 'a':
                self.target_yaw -= self.step
            elif k == 'd':
                self.target_yaw += self.step
            else:
                continue
            # Clamp HERE as well as in protocol.set_angle, so the displayed
            # target is the one that was actually sent — a target creeping
            # past the stop while the gimbal sits at the limit would read as
            # a permanent tracking error.
            self.target_pitch = max(cmds.A8_PITCH_MIN_DEG,
                                    min(cmds.A8_PITCH_MAX_DEG, self.target_pitch))
            self.target_yaw = max(cmds.A8_YAW_MIN_DEG,
                                  min(cmds.A8_YAW_MAX_DEG, self.target_yaw))
            self._send_target()

    # ------------------------------------------------------------------- link
    def _next(self) -> int:
        self._seq = (self._seq + 1) % 0xFFFF
        return self._seq

    def _tx(self, packet: bytes) -> None:
        try:
            self.link.send(packet)
        except OSError as exc:
            self.get_logger().warn(f'send failed ({exc})',
                                   throttle_duration_sec=5.0)

    def _send_target(self) -> None:
        self._tx(siyi.set_angle(self.target_yaw, self.target_pitch, self._next()))

    def _poll(self) -> None:
        self._tx(siyi.request_attitude(self._next()))

    def _drain(self) -> None:
        try:
            packets = self.link.read()
        except OSError:
            return
        for data in packets:
            parsed = siyi.decode(data)
            if parsed is None or parsed[0] != siyi.ACQUIRE_GIMBAL_ATTITUDE:
                continue
            att = siyi.parse_attitude(parsed[1])
            if att is not None:
                self.actual = att

    # ----------------------------------------------------------------- output
    def _draw(self) -> None:
        tgt = f'yaw {self.target_yaw:+6.1f}  pitch {self.target_pitch:+6.1f}'
        if self.actual is None:
            line = f'  target [{tgt}]   actual [waiting for the gimbal…]'
        else:
            ar, ap, ay = self.actual
            err_y = self.target_yaw - ay
            err_p = self.target_pitch - ap
            line = (f'  target [{tgt}]   '
                    f'actual [yaw {ay:+6.1f}  pitch {ap:+6.1f}  roll {ar:+5.1f}]'
                    f'   err [{err_y:+5.1f} {err_p:+5.1f}]   step {self.step:.1f}')
        sys.stdout.write('\r' + line + '   ')
        sys.stdout.flush()

    def restore(self) -> None:
        if self._saved is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
        print()


def main(args=None):
    rclpy.init(args=args)
    node = GimbalTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.restore()
        node.link.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Log a Gazebo/PX4 flight and render diagnostic figures on disarm.

Records the vehicle odometry (position + world velocity) while armed, plus the
A* global path, the B-spline reference and per-node compute metrics, then on
landing (disarm) saves four figures in the SAME visual language as
tools/visualize_pipeline.py:

    figures/gazebo_flight_topdown.png   buildings + A* + B-spline + flown path
                                        (annotated with the real min wall clearance)
    figures/gazebo_flight_profiles.png  speed / acceleration / altitude vs time
    figures/gazebo_flight_mpc.png       MPC tracking: reference vs flown + error
    figures/gazebo_flight_compute.png   compute load + speed: A* plan time /
                                        nodes expanded per replan, MPC solve time/tick

Subscribes: /path_plan/odometry, /path_plan/trajectory, /path_plan/global_path,
/mavros/state, /path_plan/astar_stats, /path_plan/mpc_stats.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rclpy
import yaml
from matplotlib.patches import Polygon as MplPolygon
from nav_msgs.msg import Odometry, Path
from mavros_msgs.msg import State
from pathlib import Path as FsPath
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, \
    DurabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32MultiArray

from .ros_msgs import msg_to_trajectory, path_to_positions
from .world_model import WorldModel, _find_buildings

# Palette shared with tools/visualize_pipeline.py so the figures read as one set.
C_OBST = "#9aa4ad"
C_ASTAR = "#e8590c"
C_BSPL = "#2f9e44"
C_MPC = "#ae3ec9"

_LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL,
                      history=HistoryPolicy.KEEP_LAST)


class FlightLoggerNode(Node):
    def __init__(self):
        super().__init__("flight_logger")
        self.map_yaml = self.declare_parameter("map_yaml", "").value
        # Cruise band only matters for the altitude-band shading.
        self.floor = float(self.declare_parameter("cruise_floor_m", 30.0).value)
        self.ceiling = float(self.declare_parameter("cruise_ceiling_m", 40.0).value)

        self.t, self.pos, self.vel = [], [], []      # flown time-series (armed only)
        self.ref_path = None                          # latest B-spline reference (N,3)
        self.astar_path = None                        # latest A* global path (N,3)
        # Pursuit continuously REPLANS, so keep every A*/B-spline reference (not just
        # the last) -> the figures can show all commanded paths and the tracking error
        # is measured against the union of what was actually commanded.
        self.ref_paths = []                           # all B-spline references
        self.astar_paths = []                         # all A* global paths
        self.armed = False
        self.was_armed = False
        self.reported = False

        # Compute metrics: A* per-replan [t, plan_s, expanded, n_wp]; MPC per-tick
        # [t, solve_ms].  t is wall-clock (node) seconds so we can align to flight.
        self.astar_stats = []       # rows [t, plan_s, expanded, n_wp]
        self.mpc_stats = []         # rows [t, solve_ms]

        self.create_subscription(Odometry, "/path_plan/odometry",
                                 self.odom_cb, qos_profile_sensor_data)
        self.create_subscription(Float32MultiArray, "/path_plan/trajectory",
                                 self.traj_cb, 10)
        self.create_subscription(Path, "/path_plan/global_path",
                                 self.global_path_cb, _LATCHED)
        self.create_subscription(State, "/mavros/state", self.state_cb, 10)
        self.create_subscription(Float32MultiArray, "/path_plan/astar_stats",
                                 self.astar_stats_cb, 10)
        self.create_subscription(Float32MultiArray, "/path_plan/mpc_stats",
                                 self.mpc_stats_cb, 10)
        self.get_logger().info("Flight logger ready. Recording once armed...")

    def _now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # ------------------------------------------------------------- callbacks
    def odom_cb(self, msg):
        if not self.armed:
            return
        st = msg.header.stamp
        self.t.append(st.sec + st.nanosec * 1e-9)
        p = msg.pose.pose.position
        v = msg.twist.twist.linear
        self.pos.append([p.x, p.y, p.z])
        self.vel.append([v.x, v.y, v.z])

    def traj_cb(self, msg):
        _, pos, _ = msg_to_trajectory(msg)
        self.ref_path = pos
        if pos is not None and len(pos) > 1 and (
                not self.ref_paths or not np.array_equal(pos, self.ref_paths[-1])):
            self.ref_paths.append(pos)

    def global_path_cb(self, msg):
        pos = path_to_positions(msg)
        self.astar_path = pos
        if pos is not None and len(pos) > 1 and (
                not self.astar_paths or not np.array_equal(pos, self.astar_paths[-1])):
            self.astar_paths.append(pos)

    def astar_stats_cb(self, msg):
        d = list(msg.data)
        if len(d) >= 3:
            self.astar_stats.append([self._now_s(), d[0], d[1], d[2]])

    def mpc_stats_cb(self, msg):
        if not self.armed:      # only the real-time control loop while flying
            return
        if msg.data:
            self.mpc_stats.append([self._now_s(), float(msg.data[0])])

    def state_cb(self, msg):
        self.armed = msg.armed
        if self.armed:
            self.was_armed = True
        elif self.was_armed and not self.reported:
            self.get_logger().info("Disarmed! Generating flight report...")
            self.generate_report()

    # --------------------------------------------------------------- report
    def generate_report(self):
        if self.reported:
            return
        self.reported = True
        if len(self.pos) < 3:
            self.get_logger().warn("Too few samples logged; no report.")
            return
        out = FsPath(__file__).resolve().parents[2] / "figures"
        out.mkdir(exist_ok=True)

        t = np.asarray(self.t) - self.t[0]
        pos = np.asarray(self.pos)
        vel = np.asarray(self.vel)
        foots = self._footprints()

        self._fig_topdown(out / "gazebo_flight_topdown.png", pos, foots)
        self._fig_profiles(out / "gazebo_flight_profiles.png", t, pos, vel)
        self._fig_mpc(out / "gazebo_flight_mpc.png", pos)
        n_compute = self._fig_compute(out / "gazebo_flight_compute.png")
        self.get_logger().info(
            f"Saved {3 + n_compute} flight figures to {out}")

    # ------------------------------------------------------------ figure 1
    def _draw_buildings(self, ax, foots):
        for poly in foots:
            ax.add_patch(MplPolygon(poly, closed=True, facecolor=C_OBST,
                                    edgecolor="#5c636a", lw=0.4, alpha=0.55, zorder=1))

    def _fig_topdown(self, path, pos, foots):
        """Three top-down panels in one file: the A* global paths, the B-spline
        references, and the two combined — each over the flown path. Pursuit replans
        continuously, so every replanned path is overlaid (not just the last)."""
        fig, axes = plt.subplots(1, 3, figsize=(19, 7))
        clr = self._min_clearance(pos)
        note = f"min wall clearance (flown): {clr:.1f} m" if clr is not None else ""

        def draw_flown(ax):
            ax.plot(pos[:, 0], pos[:, 1], "--", color=C_MPC, lw=2, zorder=5,
                    label="flown (MPC)")
            ax.scatter(pos[0, 0], pos[0, 1], c="#1c7ed6", s=70, zorder=6, label="start")
            ax.scatter(pos[-1, 0], pos[-1, 1], c="#c92a2a", s=70, zorder=6, label="end")

        def draw_astar(ax):
            for i, ap in enumerate(self.astar_paths):
                ax.plot(ap[:, 0], ap[:, 1], "-", color=C_ASTAR, lw=1.0, alpha=0.35,
                        zorder=3, label="A* global path" if i == 0 else None)

        def draw_bspline(ax):
            for i, rp in enumerate(self.ref_paths):
                ax.plot(rp[:, 0], rp[:, 1], "-", color=C_BSPL, lw=1.3, alpha=0.4,
                        zorder=4, label="B-spline reference" if i == 0 else None)

        # Panel A: A* only; B: B-spline only; C: combined.
        self._draw_buildings(axes[0], foots); draw_astar(axes[0]); draw_flown(axes[0])
        axes[0].set_title(f"A* global path ({len(self.astar_paths)} replans)")
        self._draw_buildings(axes[1], foots); draw_bspline(axes[1]); draw_flown(axes[1])
        axes[1].set_title(f"B-spline reference ({len(self.ref_paths)} replans)")
        self._draw_buildings(axes[2], foots)
        draw_astar(axes[2]); draw_bspline(axes[2]); draw_flown(axes[2])
        axes[2].set_title("combined")

        for ax in axes:
            ax.set_aspect("equal"); ax.grid(alpha=0.3)
            ax.set_xlabel("E  x [m]"); ax.set_ylabel("N  y [m]")
            ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        fig.suptitle(f"Gazebo Flight (Top-Down): A* -> B-spline -> MPC    |    {note}",
                     fontsize=13)
        fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)

    # ------------------------------------------------------------ figure 2
    def _fig_profiles(self, path, t, pos, vel):
        speed = np.hypot(vel[:, 0], vel[:, 1])
        # acceleration from the world velocity (finite difference on real stamps)
        dt = np.gradient(t)
        dt[dt <= 1e-6] = 1e-6
        accel = np.linalg.norm(np.gradient(vel, axis=0) / dt[:, None], axis=1)
        fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
        a1.plot(t, speed, color=C_BSPL, lw=1.8)
        a1.set_ylabel("speed [m/s]"); a1.grid(alpha=0.3)
        a1.set_title("Gazebo flight — speed / acceleration / altitude")
        a2.plot(t, accel, color=C_ASTAR, lw=1.5)
        a2.set_ylabel("|accel| [m/s²]"); a2.grid(alpha=0.3)
        a3.axhspan(self.floor, self.ceiling, color="#1c7ed6", alpha=0.12,
                   label=f"cruise band {self.floor:.0f}-{self.ceiling:.0f} m")
        a3.plot(t, pos[:, 2], color="#1c7ed6", lw=1.8)
        a3.set_ylabel("altitude z [m]"); a3.set_xlabel("time [s]")
        a3.legend(loc="lower right"); a3.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)

    # ------------------------------------------------------------ figure 3
    def _fig_mpc(self, path, pos):
        """Closed-loop tracking. Pursuit replans continuously, so the error is the
        distance from each flown point to the NEAREST point of ANY reference that was
        actually commanded (the union of all B-spline replans) -- not the last one
        (comparing the whole flight to only the final descent spline gave a bogus
        hundreds-of-metres 'error')."""
        refs = self.ref_paths or ([self.ref_path] if self.ref_path is not None else [])
        refs = [r for r in refs if r is not None and len(r) > 1]
        if not refs:
            return
        from scipy.spatial import cKDTree
        allref = np.vstack([r[:, :2] for r in refs])
        err, _ = cKDTree(allref).query(pos[:, :2])
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.5))
        for i, r in enumerate(refs):
            a1.plot(r[:, 0], r[:, 1], "-", color=C_BSPL, lw=1.2, alpha=0.4,
                    label="reference (B-spline, all replans)" if i == 0 else None)
        a1.plot(pos[:, 0], pos[:, 1], "--", color=C_MPC, lw=2, label="flown (MPC)")
        a1.set_aspect("equal"); a1.set_xlabel("x [m]"); a1.set_ylabel("y [m]")
        a1.set_title("Closed-loop tracking (top-down)")
        a1.legend(fontsize=8); a1.grid(alpha=0.3)
        a2.plot(err, color=C_MPC, lw=1.2)
        a2.set_xlabel("flown sample"); a2.set_ylabel("dist to nearest commanded ref [m]")
        a2.set_title(f"Tracking error vs all references — mean {err.mean():.2f} m, "
                     f"median {np.median(err):.2f} m, max {err.max():.2f} m")
        a2.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)

    # ------------------------------------------------------------ figure 4
    def _fig_compute(self, path):
        """Computation load + speed: A* planning cost per replan and the MPC
        controller solve time per tick. Returns 1 if saved, 0 if no data."""
        astar = np.asarray(self.astar_stats, float) if self.astar_stats else None
        mpc = np.asarray(self.mpc_stats, float) if self.mpc_stats else None
        if astar is None and mpc is None:
            self.get_logger().warn("No compute stats captured; skipping compute figure.")
            return 0

        fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 9))

        # --- A* planning: time (ms) per replan + nodes expanded (twin axis) ---
        if astar is not None and len(astar):
            idx = np.arange(1, len(astar) + 1)
            plan_ms = astar[:, 1] * 1e3
            expanded = astar[:, 2]
            a1.bar(idx, plan_ms, color=C_ASTAR, alpha=0.75, label="plan time")
            a1.set_xlabel("A* replan #")
            a1.set_ylabel("plan time [ms]", color=C_ASTAR)
            a1.tick_params(axis="y", labelcolor=C_ASTAR)
            a1.set_xticks(idx)
            at = a1.twinx()
            at.plot(idx, expanded, "-o", color="#1c7ed6", lw=1.5, ms=4,
                    label="nodes expanded")
            at.set_ylabel("nodes expanded", color="#1c7ed6")
            at.tick_params(axis="y", labelcolor="#1c7ed6")
            a1.set_title(
                f"A* planning cost — {len(astar)} replan(s), "
                f"mean {plan_ms.mean():.0f} ms / {expanded.mean():.0f} nodes, "
                f"max {plan_ms.max():.0f} ms")
            a1.grid(alpha=0.3)
        else:
            a1.set_title("A* planning cost — no data")

        # --- MPC controller: solve time (ms) per tick vs flight time ----------
        if mpc is not None and len(mpc) > 1:
            tm = mpc[:, 0] - mpc[0, 0]
            solve_ms = mpc[:, 1]
            a2.plot(tm, solve_ms, color=C_MPC, lw=1.0, alpha=0.9, label="solve time")
            # real-time budget = median control period between ticks
            dt = np.diff(mpc[:, 0])
            dt = dt[dt > 1e-4]
            if len(dt):
                budget_ms = float(np.median(dt)) * 1e3
                rate = 1e3 / budget_ms
                a2.axhline(budget_ms, color="#c92a2a", ls="--", lw=1.2,
                           label=f"control period {budget_ms:.0f} ms ({rate:.0f} Hz)")
            a2.set_xlabel("flight time [s]")
            a2.set_ylabel("MPC solve [ms]")
            a2.set_title(
                f"MPC controller solve time — {len(mpc)} ticks, "
                f"mean {solve_ms.mean():.2f} ms, max {solve_ms.max():.2f} ms")
            a2.legend(loc="upper right", framealpha=0.9)
            a2.grid(alpha=0.3)
        else:
            a2.set_title("MPC controller solve time — no data")

        fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
        return 1

    # --------------------------------------------------------------- helpers
    def _footprints(self):
        if not self.map_yaml or not FsPath(self.map_yaml).exists():
            return []
        doc = yaml.safe_load(FsPath(self.map_yaml).read_text())
        return [np.asarray(b["footprint"]["outer"], float)
                for b in _find_buildings(doc) or []]

    def _min_clearance(self, pos):
        """Euclidean distance of the closest flown point to any building (raw)."""
        if not self.map_yaml or not FsPath(self.map_yaml).exists():
            return None
        world = WorldModel.from_city_yaml(self.map_yaml, inflation_xy_m=0.0,
                                          ground_clearance_m=-1e4, ceiling_m=1e4,
                                          overfly_allowed=False)
        return min(world.clearance(p) for p in pos)


def main(args=None):
    rclpy.init(args=args)
    node = FlightLoggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.generate_report()          # also emit on Ctrl+C if we have data
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

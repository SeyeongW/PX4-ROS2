#!/usr/bin/env python3
"""Log a Gazebo/PX4 flight and render diagnostic figures on disarm.

Records the vehicle odometry (position + world velocity) while armed, plus the
A* global path and the B-spline reference, then on landing (disarm) saves three
figures in the SAME visual language as tools/visualize_pipeline.py:

    figures/gazebo_flight_topdown.png   buildings + A* + B-spline + flown path
                                        (annotated with the real min wall clearance)
    figures/gazebo_flight_profiles.png  speed / acceleration / altitude vs time
    figures/gazebo_flight_mpc.png       MPC tracking: reference vs flown + error

Subscribes: /path_plan/odometry, /path_plan/trajectory, /path_plan/global_path,
/mavros/state.
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
        self.ref_path = None                          # B-spline reference (N,3)
        self.astar_path = None                        # A* global path (N,3)
        self.armed = False
        self.was_armed = False
        self.reported = False

        self.create_subscription(Odometry, "/path_plan/odometry",
                                 self.odom_cb, qos_profile_sensor_data)
        self.create_subscription(Float32MultiArray, "/path_plan/trajectory",
                                 self.traj_cb, 10)
        self.create_subscription(Path, "/path_plan/global_path",
                                 self.global_path_cb, _LATCHED)
        self.create_subscription(State, "/mavros/state", self.state_cb, 10)
        self.get_logger().info("Flight logger ready. Recording once armed...")

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

    def global_path_cb(self, msg):
        self.astar_path = path_to_positions(msg)

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
        self.get_logger().info(f"Saved 3 flight figures to {out}")

    # ------------------------------------------------------------ figure 1
    def _fig_topdown(self, path, pos, foots):
        fig, ax = plt.subplots(figsize=(11, 11))
        for poly in foots:
            ax.add_patch(MplPolygon(poly, closed=True, facecolor=C_OBST,
                                    edgecolor="#5c636a", lw=0.4, alpha=0.55, zorder=1))
        if self.astar_path is not None and len(self.astar_path) > 1:
            ax.plot(self.astar_path[:, 0], self.astar_path[:, 1], "-o", color=C_ASTAR,
                    lw=1.5, ms=3, label="A* global path", zorder=3)
        if self.ref_path is not None and len(self.ref_path) > 1:
            ax.plot(self.ref_path[:, 0], self.ref_path[:, 1], "-", color=C_BSPL,
                    lw=3, alpha=0.7, label="B-spline reference", zorder=4)
        ax.plot(pos[:, 0], pos[:, 1], "--", color=C_MPC, lw=2,
                label="flown path (MPC)", zorder=5)
        ax.scatter(pos[0, 0], pos[0, 1], c="#1c7ed6", s=90, zorder=6, label="start")
        ax.scatter(pos[-1, 0], pos[-1, 1], c="#c92a2a", s=90, zorder=6, label="end")

        clr = self._min_clearance(pos)
        note = f"min wall clearance (flown): {clr:.1f} m" if clr is not None else ""
        ax.set_aspect("equal")
        ax.set_xlabel("E  x [m]"); ax.set_ylabel("N  y [m]")
        ax.set_title(f"Gazebo Flight (Top-Down): A* -> B-spline -> MPC\n{note}")
        ax.legend(loc="upper right", framealpha=0.9); ax.grid(alpha=0.3)
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
        if self.ref_path is None or len(self.ref_path) < 2:
            return
        from scipy.spatial import cKDTree
        err, _ = cKDTree(self.ref_path[:, :2]).query(pos[:, :2])
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.5))
        a1.plot(self.ref_path[:, 0], self.ref_path[:, 1], "-", color=C_BSPL, lw=3,
                alpha=0.6, label="reference (B-spline)")
        a1.plot(pos[:, 0], pos[:, 1], "--", color=C_MPC, lw=2, label="flown (MPC)")
        a1.set_aspect("equal"); a1.set_xlabel("x [m]"); a1.set_ylabel("y [m]")
        a1.set_title("MPC closed-loop tracking (top-down)")
        a1.legend(); a1.grid(alpha=0.3)
        a2.plot(err, color=C_MPC, lw=1.5)
        a2.set_xlabel("sample"); a2.set_ylabel("lateral tracking error [m]")
        a2.set_title(f"Tracking error (mean {err.mean():.2f} m, max {err.max():.2f} m)")
        a2.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)

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

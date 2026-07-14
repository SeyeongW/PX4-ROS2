"""Serialisation helpers between the pure algorithm modules and ROS 2 messages.

Kept separate so the algorithm modules (astar/sfc/bspline/mpc) stay ROS-free and
unit-testable.  Corridors and trajectories use ``std_msgs/Float32MultiArray`` to
avoid a custom .msg interface package:

* corridor    -> (M, 6) rows [xmin, ymin, zmin, xmax, ymax, zmax]
* trajectory  -> (N, 7) rows [t, px, py, pz, vx, vy, vz]
"""

from __future__ import annotations

import numpy as np
from geometry_msgs.msg import Point, Pose, PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

FRAME = "map"  # ENU world frame


def _labelled(rows: int, cols: int) -> list[MultiArrayDimension]:
    return [
        MultiArrayDimension(label="rows", size=rows, stride=rows * cols),
        MultiArrayDimension(label="cols", size=cols, stride=cols),
    ]


# --------------------------------------------------------------------- Path
def positions_to_path(positions: np.ndarray, stamp) -> Path:
    path = Path()
    path.header.frame_id = FRAME
    path.header.stamp = stamp
    for p in np.asarray(positions, float):
        ps = PoseStamped()
        ps.header.frame_id = FRAME
        ps.header.stamp = stamp
        ps.pose.position = Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))
        ps.pose.orientation.w = 1.0
        path.poses.append(ps)
    return path


def path_to_positions(path: Path) -> np.ndarray:
    return np.array([[p.pose.position.x, p.pose.position.y, p.pose.position.z]
                     for p in path.poses], dtype=float).reshape(-1, 3)


# ----------------------------------------------------------------- Corridor
def corridor_to_msg(boxes_min: np.ndarray, boxes_max: np.ndarray) -> Float32MultiArray:
    data = np.hstack([np.asarray(boxes_min, float), np.asarray(boxes_max, float)])
    msg = Float32MultiArray()
    msg.layout.dim = _labelled(data.shape[0], 6)
    msg.data = data.astype(np.float32).ravel().tolist()
    return msg


def msg_to_corridor(msg: Float32MultiArray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(msg.data, float).reshape(-1, 6)
    return arr[:, :3].copy(), arr[:, 3:].copy()


# --------------------------------------------------------------- Trajectory
def trajectory_to_msg(t: np.ndarray, pos: np.ndarray, vel: np.ndarray
                      ) -> Float32MultiArray:
    data = np.column_stack([np.asarray(t, float), np.asarray(pos, float),
                            np.asarray(vel, float)])
    msg = Float32MultiArray()
    msg.layout.dim = _labelled(data.shape[0], 7)
    msg.data = data.astype(np.float32).ravel().tolist()
    return msg


def msg_to_trajectory(msg: Float32MultiArray):
    arr = np.asarray(msg.data, float).reshape(-1, 7)
    return arr[:, 0].copy(), arr[:, 1:4].copy(), arr[:, 4:7].copy()

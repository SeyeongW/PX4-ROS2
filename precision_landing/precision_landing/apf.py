"""Artificial Potential Field (APF) obstacle repulsion — dynamic path-planning
baseline. A small additive repulsive velocity layered on top of the existing
attractive velocity servo (_servo_to) in precision_landing_node (APPROACH leg)
and mission_manager_node (MISSION leg).

Obstacles come from gazebo/config/obstacle_map.yaml (Method A: known map — see
gazebo/worlds/obstacle_field.sdf, which places the same obstacles physically).
Each square footprint is conservatively approximated by its bounding circle
(radius = half the footprint diagonal) so the repulsion math stays a simple
point-obstacle gradient.
"""
import math
from dataclasses import dataclass
from typing import List, Tuple

import yaml


@dataclass
class Obstacle:
    e: float
    n: float
    radius: float   # bounding-circle radius around the (square) footprint


def load_obstacles(path: str) -> List[Obstacle]:
    with open(path) as f:
        data = yaml.safe_load(f)
    fw, fd = data['footprint']             # [east_size, north_size] metres, default
    default_radius = math.hypot(fw, fd) / 2.0
    obstacles = []
    for o in data['obstacles']:
        # Per-obstacle footprint override -- needed for maps like
        # obstacle_map_city.yaml where each building has its own real
        # footprint instead of one size shared by every obstacle.
        if 'footprint' in o:
            ow, od = o['footprint']
            radius = math.hypot(ow, od) / 2.0
        else:
            radius = default_radius
        obstacles.append(Obstacle(float(o['e']), float(o['n']), radius))
    return obstacles


def repulsive_velocity(pos_e: float, pos_n: float, obstacles: List[Obstacle],
                        influence_radius: float, gain: float,
                        vel_cap: float) -> Tuple[float, float]:
    """Sum of repulsive velocities from every obstacle within influence_radius
    of the drone's current position (surface distance, not centre distance,
    so influence_radius is measured from the obstacle's physical edge). Zero
    once outside every obstacle's influence — a drone far from all obstacles
    gets exactly the existing attractive-only behaviour.

    LINEAR ramp (0 at the influence boundary, `gain` m/s right at the
    obstacle's surface) rather than the classical Khatib 1/d^2 potential: the
    inverse-square gradient stays negligible until the drone is within a
    couple of metres of the surface, which is far too little lead distance
    for a velocity-controlled vehicle (with real inertia/acceleration limits)
    approaching at several m/s to actually deflect before impact. The linear
    ramp gives a meaningfully-sized push starting from much farther out.
    Result is capped to vel_cap so a close obstacle steers around rather than
    overpowering the attractive term into a dead stop.
    """
    vE = vN = 0.0
    for obs in obstacles:
        dx, dn = pos_e - obs.e, pos_n - obs.n
        centre_dist = math.hypot(dx, dn)
        surf_dist = max(centre_dist - obs.radius, 0.0)
        if surf_dist >= influence_radius:
            continue
        mag = gain * (influence_radius - surf_dist) / influence_radius
        if centre_dist > 1e-6:
            ux, un = dx / centre_dist, dn / centre_dist
        else:
            ux, un = 1.0, 0.0   # degenerate case: directly on the obstacle centre
        vE += mag * ux
        vN += mag * un
    speed = math.hypot(vE, vN)
    if speed > vel_cap:
        scale = vel_cap / speed
        vE *= scale
        vN *= scale
    return vE, vN

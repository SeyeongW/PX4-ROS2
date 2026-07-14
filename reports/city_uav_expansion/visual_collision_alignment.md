# UAV city visual/collision/YAML alignment

- Source of truth: `gazebo/maps/city_coordinates.yaml` (274 buildings).
- Active reduction: 205 retained / 69 removed.
- Derived coordinate geometry: `gazebo/maps/city_coordinates_uav.yaml`.
- Gazebo visual URI: `mesh/buildings_uav.dae` at scale `1 1 1`.
- Gazebo collision: 205 exact DART SDF polyline prisms.
- Mesh SHA256: `3692d96187c9b512d2a29bb4c8bb3d3e91442819c3d88672f7c755d985d0fb7e`.
- Vertex/triangle count: 21096 / 7032.
- Maximum YAML-to-visual-mesh XY boundary error: `0.0 m` (limit `0.01 m`).
- Collision source-vertex alignment error: `0.0 m` (limit `0.01 m`).
- Collision maximum undercoverage/outward error: `0.0 m`.
- Building foundations and ground datum are unchanged from source (`0.0 m` error).
- Roofs use `deterministic_active_hash_rank_20_to_50m_mean35_v1` with exact range
  `20..50 m` above ground and
  exact arithmetic mean `35 m`.
- Selected footprint scale: `2.5` — match origin/main initial-city proportions by scaling both building centroids and local XY footprints by 2.5; retain the deterministic 205-building selection and remap only roof heights to 20--50m with exact 35m mean.

The closed visual mesh and every collision prism are regenerated directly from
the same transformed outer/hole rings. No world-level building scale or pose
is used. Harmonic 8.14 DART cannot construct SDF mesh collisions, so the DAE is
visual-only and exact extruded polylines provide physical collision. The
courtyard remains open and PX4 keeps its stable DART dynamics.

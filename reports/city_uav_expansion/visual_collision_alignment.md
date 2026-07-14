# UAV city visual/collision/YAML alignment

- Source of truth: `gazebo/maps/city_coordinates.yaml` (274 buildings).
- Active reduction: 205 retained / 69 removed.
- Derived coordinate geometry: `gazebo/maps/city_coordinates_uav.yaml`.
- Gazebo visual URI: `mesh/buildings_uav.dae` at scale `1 1 1`.
- Gazebo collision: 205 exact DART SDF polyline prisms.
- Mesh SHA256: `91337e73daa84e83845a45b6a5c153972c25dd8825d59a1b340b63d0ba491ac5`.
- Vertex/triangle count: 21096 / 7032.
- Maximum YAML-to-visual-mesh XY boundary error: `0.0 m` (limit `0.01 m`).
- Collision source-vertex alignment error: `0.0 m` (limit `0.01 m`).
- Collision maximum undercoverage/outward error: `0.0 m`.
- Building foundations and ground datum are unchanged from source (`0.0 m` error).
- Roofs use `deterministic_hash_rank_30_to_70m_v1` with exact range
  `30..70 m` above ground.
- Selected footprint scale: `2.5` — match origin/main initial-city proportions by scaling both building centroids and local XY footprints by 2.5; retain the deterministic 205-building selection and remap only roof heights to 30--70m.

The closed visual mesh and every collision prism are regenerated directly from
the same transformed outer/hole rings. No world-level building scale or pose
is used. Harmonic 8.14 DART cannot construct SDF mesh collisions, so the DAE is
visual-only and exact extruded polylines provide physical collision. The
courtyard remains open and PX4 keeps its stable DART dynamics.

# UAV city visual/collision/YAML alignment

- Source of truth: `gazebo/maps/city_coordinates.yaml` (274 buildings).
- Active reduction: 205 retained / 69 removed.
- Derived coordinate geometry: `gazebo/maps/city_coordinates_uav.yaml`.
- Gazebo visual URI: `mesh/buildings_uav.dae` at scale `1 1 1`.
- Gazebo collision: one static shared `mesh/buildings_uav.dae` triangle mesh.
- Mesh SHA256: `4dcbaf4079bd89fb04df9e367e82086b6ad2b23116239f73fdd357a151bda298`.
- Vertex/triangle count: 21144 / 7048.
- Maximum YAML-to-visual-mesh XY boundary error: `0.0 m` (limit `0.01 m`).
- Collision source-vertex alignment error: `0.0 m` (limit `0.01 m`).
- Collision maximum undercoverage/outward error: `0.0 m`.
- Maximum foundation/roof Z error from source: `0.0 m` (limit `1e-9 m`).
- Selected footprint scale: `0.9` — restore origin/jo building XY: 2.5x centroids and 0.9x footprints; create passages only through deterministic building removal.

The closed mesh is regenerated directly from every transformed outer/hole
ring. No world-level building scale or pose is used. Gazebo visual and DART
collision both reference that same single file at scale `1 1 1`, so the
courtyard remains open and no per-building collision entities are needed.

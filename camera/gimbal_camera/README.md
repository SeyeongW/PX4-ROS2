# gimbal_camera

Standalone 3-axis gimbal camera based on the PX4 CGO3 gimbal, in **two
simulators from one folder**: Gazebo Sim (Harmonic / gz-sim8) and Gazebo
Classic 11. Same geometry, same PID gains, same ROS topics — pick a launch
file. In both cases the joints are driven by a PID that runs *inside the
simulator* at the physics rate, so you command **target angles** and the camera
holds them rock-steady — no tremor.

## Layout
```
gimbal_camera/
├── models/
│   ├── gimbal/          # gz-sim model (model.sdf + meshes) — meshes shared
│   └── gimbal_classic/  # Gazebo Classic model (reuses ../gimbal/meshes)
├── worlds/
│   ├── gimbal.sdf            # gz world (+ gz-sim systems)
│   └── gimbal_classic.world  # Classic world (+ gazebo_ros_init)
├── plugins/             # Classic only: PID joint controller, see below
├── launch/
│   ├── gimbal_camera.launch.py          # gz  (+ 2 bridge nodes)
│   └── gimbal_camera_classic.launch.py  # Classic (no bridges needed)
└── scripts/gimbal_keyboard_control.py   # drives BOTH
```

## Requirements

**gz-sim (Harmonic)** — ROS 2 Humble + `ros_gz_sim`, `ros_gz_bridge`,
`ros_gz_image`. No build needed; the launch file resolves its own paths.

**Gazebo Classic 11** — additionally:
```bash
sudo apt install gazebo11 ros-humble-gazebo-ros-pkgs
cmake -S ~/gimbal_camera/plugins -B ~/gimbal_camera/plugins/build
cmake --build ~/gimbal_camera/plugins/build
```
The one build step is unavoidable: `gazebo_ros_pkgs` ships **no joint position
controller**. Its `joint_pose_trajectory` plugin sets angles kinematically with
no loop, which throws away exactly the steadiness this gimbal is for, so
`plugins/` reimplements gz's `JointPositionController` control law
(`force = -(p·e + i·∫e + d·ė)`, clamped) — which is what lets the gains carry
over from `models/gimbal/model.sdf` unchanged. Sensor plugins (camera, IMU,
joint states) do come from `gazebo_ros_pkgs`.

```bash
source /opt/ros/humble/setup.bash
```

## Run
```bash
# Terminal 1 — simulator (gz: + bridges)
ros2 launch ~/gimbal_camera/launch/gimbal_camera.launch.py
#   ...or Gazebo Classic (no bridges — the plugins speak ROS directly):
ros2 launch ~/gimbal_camera/launch/gimbal_camera_classic.launch.py
#   headless: append gui:=false

# Terminal 2 — keyboard control (must be a real terminal)
~/gimbal_camera/scripts/gimbal_keyboard_control.py
```

Keys: `a`/`d` yaw · `w`/`s` pitch · `z`/`c` roll · `space` reset · `q` quit
(0.05 rad per keypress).

## Topics
| topic | type | dir |
|---|---|---|
| `/gimbal/yaw_cmd` `/gimbal/pitch_cmd` `/gimbal/roll_cmd` | `std_msgs/Float64` | target angle [rad] |
| `/gimbal/joint_states` | `sensor_msgs/JointState` | feedback |
| `/gimbal/camera` | `sensor_msgs/Image` | camera |
| `/gimbal/camera_info` | `sensor_msgs/CameraInfo` | camera |

Command an angle without the keyboard:
```bash
ros2 topic pub -1 /gimbal/pitch_cmd std_msgs/msg/Float64 "{data: -0.8}"
```
View the camera:
```bash
ros2 run rqt_image_view rqt_image_view /gimbal/camera
```

## Joint ranges
- yaw (`cgo3_vertical_arm_joint`): continuous
- roll (`cgo3_horizontal_arm_joint`): ±45°
- pitch (`cgo3_camera_joint`): −135° … +45°

## Notes
- The gz launch forces `gz_version:=8` (Harmonic). Without it, ros_gz defaults
  to Fortress (`ign gazebo`) which can't read this world.
- Gimbal PID gains live in the model SDF (the three joint-controller plugins),
  not in the Python node — `models/gimbal/model.sdf` for gz,
  `models/gimbal_classic/model.sdf` for Classic. They are identical.
- Classic needs no `ros_gz_bridge`: `libgazebo_ros_camera`,
  `libgazebo_ros_imu_sensor` and `libgazebo_ros_joint_state_publisher` publish
  ROS messages directly, and the joint controller subscribes to ROS directly.
  That is 3 fewer processes than the gz path.
- The Classic model shares `models/gimbal/meshes/` rather than copying 2.7 MB
  of STL, so both models must stay under `models/`.
- Classic's SDF 1.7 has no `<dynamic_bias_stddev>`; the IMU's dynamic-bias
  sigma is carried over as the static `<bias_stddev>` and the 1000 s
  correlation time dropped (over any realistic session it is near-constant).

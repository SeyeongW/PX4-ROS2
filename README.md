# ArduPilot ROS 2 Offboard Control

This repository provides a ROS 2 implementation for offboard control of ArduPilot-based vehicles (Copter/Rover/Plane) using **MAVROS**.

## Prerequisites

Before using this package, you must have ROS 2 (Humble recommended) installed on your system.

### 1. Install MAVROS and Dependencies
First, install the MAVROS package and its message definitions:
```bash
sudo apt-get update
sudo apt-get install ros-humble-mavros ros-humble-mavros-msgs
```

### 2. Install GeographicLib Datasets
MAVROS requires GeographicLib datasets for coordinate transformations (e.g., GPS to local frames). Run the following script to install them:
```bash
sudo /opt/ros/humble/lib/mavros/install_geographiclib_datasets.sh
```

## Setup and Installation

Follow these steps to clone and build the repository in your ROS 2 workspace.

### 1. Create a Workspace (if not already exists)
```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
```

### 2. Clone the Repository
Clone this repository into your workspace:
```bash
git clone https://github.com/SeyeongW/PX4-ROS2.git
```
*(Note: Replace the URL if the repository location is different)*

### 3. Install Package Dependencies
Use `rosdep` to install any remaining dependencies automatically:
```bash
cd ~/ros2_ws
rosdep update
rosdep install --from-paths src --ignore-src -r -y
```

### 4. Build the Workspace
```bash
colcon build --packages-select offboard
source install/setup.bash
```

## Running the Offboard Control Node

Ensure your flight controller (SITL or real hardware) is connected via MAVROS.

### 1. Launch MAVROS
Start the MAVROS node (example for SITL):
```bash
ros2 launch mavros apm_sitl.launch
```

### 2. Run the Control Node
In a new terminal:
```bash
ros2 run offboard offboard_control
```

## Node Logic Description

The `offboard_control` node implements a state machine to handle the following flight phases:
- **Warmup**: Sets the mode to `GUIDED` and arms the vehicle.
- **Takeoff**: Increases altitude to the target height (default: 50m).
- **Hold**: Waits for a short period before starting the mission.
- **Move**: Executes a lawnmower path based on the starting position and heading.
- **Landing (RTL)**: Switches to `RTL` (Return to Launch) mode once the mission is complete.

All coordinates follow the **ENU** (East-North-Up) standard used in ROS.
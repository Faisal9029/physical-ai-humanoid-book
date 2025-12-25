---
sidebar_position: 5
---

# Gazebo Setup and Configuration

This guide will walk you through the installation and configuration of Gazebo, a lightweight and versatile simulation environment for robotics development. While Isaac Sim is our primary simulation platform, Gazebo provides an excellent alternative with ROS 2 integration.

## Overview

Gazebo is a 3D simulation environment that provides:

- Physics simulation with ODE, Bullet, and Simbody engines
- 3D rendering capabilities
- Sensor simulation
- Plugin architecture for customization
- Native ROS 2 integration
- Cross-platform compatibility

## Prerequisites

Before installing Gazebo, ensure you have:

- Ubuntu 22.04 LTS installed
- ROS 2 Humble Hawksbill installed
- Basic development tools (gcc, cmake, etc.)
- At least 5GB free disk space

## Install Gazebo Garden

### Add Gazebo Repository

```bash
# Install prerequisites
sudo apt install software-properties-common
sudo add-apt-repository ppa:ignitionrobotics/garden
sudo apt-get update
```

### Install Gazebo

```bash
sudo apt-get install gz-garden
```

### Verify Installation

```bash
gz --version
```

You should see output similar to: `Gazebo Garden [version]`

## Install Gazebo ROS 2 Bridge

### Install Gazebo ROS 2 Packages

```bash
sudo apt install -y ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros2-control
sudo apt install -y ros-humble-gazebo-msgs
```

### Install Additional ROS 2 Controllers

```bash
sudo apt install -y ros-humble-ros2-control ros-humble-ros2-controllers
sudo apt install -y ros-humble-joint-state-broadcaster ros-humble-joint-trajectory-controller
```

## Gazebo Configuration

### Environment Variables

Add Gazebo settings to your `~/.bashrc`:

```bash
# Gazebo Configuration
export GZ_SIM_SYSTEM_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/gazebo-11/plugins
export GZ_SIM_RESOURCE_PATH=/usr/share/gazebo-11/models
export GAZEBO_MODEL_PATH=/usr/share/gazebo-11/models
export GAZEBO_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/gazebo-11/plugins
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/gazebo-11/plugins:$LD_LIBRARY_PATH
```

Source the environment:

```bash
source ~/.bashrc
```

## Basic Gazebo Usage

### Launch Gazebo

```bash
gz sim
```

This will open the Gazebo GUI with a default empty world.

### Launch Gazebo with a Specific World

```bash
# Launch with an empty world
gz sim -r -v 4 empty.sdf

# Launch with a simple world
gz sim -r -v 4 shapes.sdf
```

### Command Line Options

```bash
# Verbose output
gz sim -v 4

# Run without GUI
gz sim -s

# Load a specific world file
gz sim -r my_world.sdf
```

## Gazebo World Creation

### Create a Simple World File

Create `~/gazebo_ws/worlds/simple_room.sdf`:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="simple_room">
    <!-- Include a default ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Include a default light source -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Add a simple box -->
    <model name="box">
      <pose>0 0 0.5 0 0 0</pose>
      <link name="link">
        <visual name="visual">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.2 0.2 1</ambient>
            <diffuse>0.8 0.2 0.2 1</diffuse>
          </material>
        </visual>
        <collision name="collision">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </collision>
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.166667</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.166667</iyy>
            <iyz>0</iyz>
            <izz>0.166667</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <!-- Add a simple sphere -->
    <model name="sphere">
      <pose>2 0 1 0 0 0</pose>
      <link name="link">
        <visual name="visual">
          <geometry>
            <sphere>
              <radius>0.5</radius>
            </sphere>
          </geometry>
          <material>
            <ambient>0.2 0.8 0.2 1</ambient>
            <diffuse>0.2 0.8 0.2 1</diffuse>
          </material>
        </visual>
        <collision name="collision">
          <geometry>
            <sphere>
              <radius>0.5</radius>
            </sphere>
          </geometry>
        </collision>
        <inertial>
          <mass>0.5</mass>
          <inertia>
            <ixx>0.05</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.05</iyy>
            <iyz>0</iyz>
            <izz>0.05</izz>
          </inertia>
        </inertial>
      </link>
    </model>
  </world>
</sdf>
```

### Launch Your Custom World

```bash
gz sim -r ~/gazebo_ws/worlds/simple_room.sdf
```

## Robot Model Integration

### Create a Simple Robot Model

Create directory structure:

```bash
mkdir -p ~/gazebo_ws/models/my_robot/meshes
mkdir -p ~/gazebo_ws/models/my_robot/materials/textures
mkdir -p ~/gazebo_ws/models/my_robot/materials/scripts
```

Create `~/gazebo_ws/models/my_robot/model.sdf`:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <model name="my_robot">
    <link name="chassis">
      <pose>0 0 0.1 0 0 0</pose>
      <collision name="collision">
        <geometry>
          <box>
            <size>0.5 0.3 0.2</size>
          </box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box>
            <size>0.5 0.3 0.2</size>
          </box>
        </geometry>
      </visual>
      <inertial>
        <mass>1.0</mass>
        <inertia>
          <ixx>0.01</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.02</iyy>
          <iyz>0</iyz>
          <izz>0.03</izz>
        </inertia>
      </inertial>
    </link>

    <!-- Simple caster wheel -->
    <link name="caster_wheel">
      <pose>0.15 0 -0.1 0 0 0</pose>
      <collision name="collision">
        <geometry>
          <sphere>
            <radius>0.05</radius>
          </sphere>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <sphere>
            <radius>0.05</radius>
          </sphere>
        </geometry>
      </visual>
      <inertial>
        <mass>0.1</mass>
        <inertia>
          <ixx>0.0001</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.0001</iyy>
          <iyz>0</iyz>
          <izz>0.0001</izz>
        </inertia>
      </inertial>
    </link>

    <!-- Joint connecting caster to chassis -->
    <joint name="caster_joint" type="fixed">
      <parent>chassis</parent>
      <child>caster_wheel</child>
    </joint>
  </model>
</sdf>
```

Create `~/gazebo_ws/models/my_robot/model.config`:

```xml
<?xml version="1.0"?>
<model>
  <name>My Robot</name>
  <version>1.0</version>
  <sdf version="1.7">model.sdf</sdf>
  <description>A simple robot model for Gazebo.</description>
</model>
```

### Set Model Path

Add to your `~/.bashrc`:

```bash
export GAZEBO_MODEL_PATH=$HOME/gazebo_ws/models:$GAZEBO_MODEL_PATH
```

## ROS 2 Integration

### Create a ROS 2 Package for Gazebo Integration

```bash
# Create a new ROS 2 package
cd ~/physical_ai_ws/src
ros2 pkg create --build-type ament_python gazebo_integration --dependencies rclpy std_msgs geometry_msgs sensor_msgs gazebo_msgs
```

### Install Gazebo ROS 2 Control

```bash
sudo apt install -y ros-humble-gazebo-ros2-control ros-humble-gazebo-ros2-control-demos
sudo apt install -y ros-humble-ros2-control-test-assets
```

### Example Launch File

Create `~/physical_ai_ws/src/gazebo_integration/launch/gazebo.launch.py`:

```python
import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Get the package share directory
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')

    # Launch Gazebo with empty world
    gzserver_cmd = ExecuteProcess(
        cmd=['gzserver', '--verbose', '-s', 'libgazebo_ros_init.so', '-s', 'libgazebo_ros_factory.so'],
        output='screen'
    )

    gzclient_cmd = ExecuteProcess(
        cmd=['gzclient', '--verbose'],
        output='screen',
        condition=launch.conditions.IfCondition(LaunchConfiguration('gui'))
    )

    # Create the launch description and populate
    ld = LaunchDescription()

    # Add commands to the launch description
    ld.add_action(gzserver_cmd)
    ld.add_action(gzclient_cmd)

    return ld
```

## Gazebo Performance Optimization

### Graphics Settings

For better performance, especially on cloud systems:

```bash
# For cloud instances, try software rendering
export MESA_GL_VERSION_OVERRIDE=3.3
export LIBGL_ALWAYS_SOFTWARE=1

# Reduce rendering quality for better performance
export OGRE_RESOURCEMANAGER_STRICT=0
```

### Physics Settings

Optimize physics simulation:

```bash
# In Gazebo, adjust these parameters:
# Real Time Update Rate: 1000 (higher for more accuracy)
# Max Step Size: 0.001 (smaller for more accuracy)
# Number of Physics Threads: 4 (match your CPU cores)
```

## Troubleshooting

### Common Issues and Solutions

#### Issue: "gz: command not found"

**Solution**:
```bash
# Check if Gazebo is installed
dpkg -l | grep gz

# Add to PATH if needed
export PATH=/usr/bin:$PATH
source ~/.bashrc
```

#### Issue: "Gazebo fails to start with OpenGL errors"

**Solution**:
```bash
# Check OpenGL support
glxinfo | grep "OpenGL version"

# Try software rendering
export LIBGL_ALWAYS_SOFTWARE=1
gz sim
```

#### Issue: "Gazebo ROS 2 bridge not working"

**Solution**:
```bash
# Verify ROS 2 environment
source /opt/ros/humble/setup.bash

# Check if packages are installed
dpkg -l | grep gazebo-ros

# Verify environment variables
echo $GAZEBO_MODEL_PATH
echo $GAZEBO_PLUGIN_PATH
```

### Verification Commands

```bash
# Check Gazebo installation
gz --version

# Check Gazebo ROS 2 packages
source /opt/ros/humble/setup.bash
ros2 pkg list | grep gazebo

# Test basic Gazebo launch
gz sim --version
```

## Comparison: Gazebo vs Isaac Sim

| Feature | Gazebo | Isaac Sim |
|---------|--------|-----------|
| Rendering | Good | Excellent (photorealistic) |
| Physics | ODE, Bullet, Simbody | PhysX (advanced) |
| GPU Acceleration | Limited | Full RTX support |
| ROS 2 Integration | Native | Excellent |
| Learning Curve | Moderate | Steeper |
| Resource Usage | Lower | Higher |
| AI Perception Tools | Basic | Advanced |

## Validation Checklist

- [ ] Gazebo launches without errors
- [ ] Basic world loads successfully
- [ ] Custom world file works
- [ ] Robot model displays correctly
- [ ] ROS 2 integration packages installed
- [ ] Environment variables properly set
- [ ] Launch files work correctly

## Next Steps

Once Gazebo is successfully installed and configured:

1. Continue to [Environment Validation](./environment-validation.md) to test your complete setup
2. Compare Gazebo and Isaac Sim performance for your use case
3. Learn more advanced Gazebo features in Week 4

Your Gazebo environment is now ready as an alternative simulation platform for the Physical AI and Humanoid Robotics course!
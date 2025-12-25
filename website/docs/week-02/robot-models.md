---
sidebar_position: 2
---

# Robot Models and URDF

This guide covers the fundamentals of robot modeling using URDF (Unified Robot Description Format). URDF is the standard format for representing robot models in ROS, defining the physical and visual properties of robots.

## Overview

URDF (Unified Robot Description Format) is an XML format that describes robot models. It defines the physical structure of a robot, including:

- Links (rigid bodies)
- Joints (connections between links)
- Visual and collision properties
- Inertial properties
- Materials and colors

## URDF Structure

### Basic URDF Components

A URDF robot model consists of:

1. **Links**: Rigid parts of the robot
2. **Joints**: Connections between links
3. **Materials**: Visual appearance
4. **Gazebo plugins**: Simulation-specific properties

### Simple URDF Example

Let's start with a simple 2-link robot arm:

```xml
<?xml version="1.0"?>
<robot name="simple_arm" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.1" radius="0.1"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.1" radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <!-- First link -->
  <link name="link1">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.5"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.05 0.05 0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Joint connecting base to link1 -->
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>
</robot>
```

## Creating Your First Robot Model

### Setting Up the Project Structure

First, let's create a ROS 2 package for our robot models:

```bash
# Navigate to your workspace
cd ~/physical_ai_ws/src

# Create a robot description package
ros2 pkg create --build-type ament_cmake robot_description --dependencies xacro

# Create directory structure
mkdir -p robot_description/urdf
mkdir -p robot_description/meshes
mkdir -p robot_description/materials/textures
```

### Basic URDF Robot

Create `~/physical_ai_ws/src/robot_description/urdf/simple_robot.urdf`:

```xml
<?xml version="1.0"?>
<robot name="simple_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder radius="0.2" length="0.1"/>
      </geometry>
      <material name="green">
        <color rgba="0 1 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.2" length="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10"/>
      <inertia ixx="0.42" ixy="0" ixz="0" iyy="0.42" iyz="0" izz="0.8"/>
    </inertial>
  </link>

  <!-- Left wheel -->
  <link name="left_wheel">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1"/>
      <inertia ixx="0.0025" ixy="0" ixz="0" iyy="0.0025" iyz="0" izz="0.005"/>
    </inertial>
  </link>

  <!-- Right wheel -->
  <link name="right_wheel">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1"/>
      <inertia ixx="0.0025" ixy="0" ixz="0" iyy="0.0025" iyz="0" izz="0.005"/>
    </inertial>
  </link>

  <!-- Joints -->
  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel"/>
    <origin xyz="0 0.15 -0.05" rpy="1.57 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <joint name="right_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="right_wheel"/>
    <origin xyz="0 -0.15 -0.05" rpy="1.57 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>
</robot>
```

## URDF Components Explained

### Links

Links represent rigid bodies in the robot. Each link contains:

- **Visual**: How the link looks in visualization
- **Collision**: How the link interacts in physics simulation
- **Inertial**: Physical properties for dynamics

```xml
<link name="example_link">
  <!-- Visual properties -->
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <box size="1 1 1"/>
    </geometry>
    <material name="red">
      <color rgba="1 0 0 1"/>
    </material>
  </visual>

  <!-- Collision properties -->
  <collision>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <box size="1 1 1"/>
    </geometry>
  </collision>

  <!-- Inertial properties -->
  <inertial>
    <mass value="1.0"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
  </inertial>
</link>
```

### Joints

Joints connect links and define how they can move relative to each other:

```xml
<joint name="example_joint" type="revolute">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <origin xyz="0 0 0.1" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
</joint>
```

## Joint Types

### 1. Fixed Joint

A fixed joint creates a rigid connection between two links:

```xml
<joint name="fixed_joint" type="fixed">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <origin xyz="0.1 0 0" rpy="0 0 0"/>
</joint>
```

### 2. Revolute Joint

A revolute joint allows rotation around a single axis with limited range:

```xml
<joint name="revolute_joint" type="revolute">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <origin xyz="0 0 0.1" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
</joint>
```

### 3. Continuous Joint

A continuous joint allows unlimited rotation around a single axis:

```xml
<joint name="continuous_joint" type="continuous">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <origin xyz="0 0 0.1" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
</joint>
```

### 4. Prismatic Joint

A prismatic joint allows linear motion along a single axis:

```xml
<joint name="prismatic_joint" type="prismatic">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <origin xyz="0 0 0.1" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="0" upper="0.5" effort="100" velocity="1"/>
</joint>
```

### 5. Floating Joint

A floating joint allows motion in all 6 degrees of freedom:

```xml
<joint name="floating_joint" type="floating">
  <parent link="parent_link"/>
  <child link="child_link"/>
</joint>
```

## Advanced URDF Features

### Using Xacro for Macros

Xacro (XML Macros) allows you to create reusable URDF components:

Create `~/physical_ai_ws/src/robot_description/urdf/simple_robot.xacro`:

```xml
<?xml version="1.0"?>
<robot name="simple_robot_xacro" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Properties -->
  <xacro:property name="base_width" value="0.2"/>
  <xacro:property name="base_length" value="0.3"/>
  <xacro:property name="base_height" value="0.1"/>
  <xacro:property name="wheel_radius" value="0.1"/>
  <xacro:property name="wheel_width" value="0.05"/>
  <xacro:property name="wheel_offset" value="0.15"/>

  <!-- Materials -->
  <material name="blue">
    <color rgba="0 0 1 1"/>
  </material>
  <material name="black">
    <color rgba="0 0 0 1"/>
  </material>

  <!-- Wheel macro -->
  <xacro:macro name="wheel" params="prefix parent x y z">
    <link name="${prefix}_wheel">
      <visual>
        <origin xyz="0 0 0" rpy="1.57 0 0"/>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
        <material name="black"/>
      </visual>
      <collision>
        <origin xyz="0 0 0" rpy="1.57 0 0"/>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="0.5"/>
        <inertia ixx="0.00125" ixy="0" ixz="0" iyy="0.00125" iyz="0" izz="0.0025"/>
      </inertial>
    </link>

    <joint name="${prefix}_wheel_joint" type="continuous">
      <parent link="${parent}"/>
      <child link="${prefix}_wheel"/>
      <origin xyz="${x} ${y} ${z}" rpy="0 0 0"/>
      <axis xyz="0 0 1"/>
    </joint>
  </xacro:macro>

  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="${base_width} ${base_length} ${base_height}"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <geometry>
        <box size="${base_width} ${base_length} ${base_height}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.2"/>
    </inertial>
  </link>

  <!-- Use the wheel macro -->
  <xacro:wheel prefix="left" parent="base_link" x="0" y="${wheel_offset}" z="-0.05"/>
  <xacro:wheel prefix="right" parent="base_link" x="0" y="-${wheel_offset}" z="-0.05"/>

</robot>
```

## URDF Validation

### Check URDF Syntax

```bash
# Install urdfdom tools if not already installed
sudo apt install ros-humble-urdfdom-tools

# Validate URDF syntax
check_urdf ~/physical_ai_ws/src/robot_description/urdf/simple_robot.urdf
```

### Visualize URDF in RViz2

```bash
# Launch robot state publisher
ros2 run robot_state_publisher robot_state_publisher --ros-args -p robot_description:=$(cat ~/physical_ai_ws/src/robot_description/urdf/simple_robot.urdf)

# In another terminal, launch RViz2
rviz2
```

In RViz2, add a RobotModel display and set the Robot Description to "robot_description".

## Loading URDF in Simulation

### Launch in Gazebo

Create `~/physical_ai_ws/src/robot_description/launch/robot_spawn.launch.py`:

```python
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare launch arguments
    urdf_model = DeclareLaunchArgument(
        'urdf_model',
        default_value='simple_robot.urdf',
        description='URDF file name'
    )

    # Get URDF path
    robot_description_path = PathJoinSubstitution([
        FindPackageShare('robot_description'),
        'urdf',
        LaunchConfiguration('urdf_model')
    ])

    # Robot State Publisher node
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': robot_description_path
        }]
    )

    # Joint State Publisher GUI (for testing joints)
    joint_state_publisher_gui = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui'
    )

    # RViz2 node
    rviz2 = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', PathJoinSubstitution([
            FindPackageShare('robot_description'),
            'rviz',
            'robot.rviz'
        ])]
    )

    return LaunchDescription([
        urdf_model,
        robot_state_publisher,
        joint_state_publisher_gui,
        rviz2
    ])
```

### Test URDF in Simulation

```bash
# Build the workspace
cd ~/physical_ai_ws
colcon build --packages-select robot_description

# Source the workspace
source install/setup.bash

# Launch the robot visualization
ros2 launch robot_description robot_spawn.launch.py
```

## Best Practices

### 1. Naming Conventions

- Use consistent naming: `link_name`, `joint_name`, `material_name`
- Use descriptive names that indicate function
- Follow ROS naming conventions (lowercase, underscores)

### 2. Origin and Coordinate Systems

- Use consistent coordinate frames (typically X forward, Y left, Z up)
- Define origins clearly relative to parent links
- Use RPY (Roll-Pitch-Yaw) for rotations

### 3. Inertial Properties

- Calculate inertial properties accurately for realistic simulation
- Use CAD software to calculate inertial properties if possible
- For simple shapes, use standard formulas

### 4. Mesh Usage

For complex geometries, use mesh files:

```xml
<visual>
  <geometry>
    <mesh filename="package://robot_description/meshes/complex_part.stl"/>
  </geometry>
</visual>
```

## Common Issues and Troubleshooting

### URDF Parsing Errors

- Check XML syntax (proper closing tags, quotes)
- Ensure all links are connected by joints
- Verify joint parent/child relationships

### Inertial Issues

- Ensure all links have inertial properties
- Use realistic mass values
- Calculate inertia tensors correctly

### Joint Issues

- Verify joint limits are appropriate
- Check joint types match intended motion
- Ensure joint axes are correctly oriented

## Next Steps

After mastering basic URDF modeling:

1. Practice creating more complex robot models
2. Learn about sensor integration in URDF
3. Explore advanced kinematic chains
4. Continue to [Joint Control Concepts](./joint-control.md) to understand how to control these models

Your understanding of URDF is now foundational for creating and simulating robots in the Physical AI and Humanoid Robotics course!
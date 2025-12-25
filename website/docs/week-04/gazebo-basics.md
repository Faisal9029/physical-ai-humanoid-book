---
sidebar_position: 2
---

# Gazebo Simulation Basics

This guide covers the fundamentals of Gazebo simulation, including installation, configuration, and basic usage. Gazebo is a powerful open-source simulation environment that provides realistic physics simulation and rendering for robotics applications.

## Overview

Gazebo is a 3D simulation environment that provides:
- Realistic physics simulation using ODE, Bullet, or Simbody
- High-quality rendering capabilities
- Sensor simulation for cameras, LiDAR, IMU, and other sensors
- Plugin architecture for custom functionality
- ROS integration through gazebo_ros_pkgs

### Key Components

Gazebo consists of several key components:
1. **gz-sim**: The core simulation engine
2. **gz-gui**: The graphical user interface
3. **gz-physics**: Physics engine abstraction
4. **gz-sensors**: Sensor simulation framework

## Installation and Setup

### Prerequisites

Before installing Gazebo Garden, ensure you have:
- Ubuntu 22.04 LTS
- ROS 2 Humble Hawksbill
- Compatible graphics drivers (for GUI)
- At least 4GB RAM and 2GB free disk space

### Installing Gazebo Garden

```bash
# Add Gazebo repository
sudo apt install software-properties-common
sudo add-apt-repository ppa:ignitionrobotics/garden
sudo apt-get update

# Install Gazebo Garden
sudo apt-get install gz-garden

# Verify installation
gz --version
```

### Installing Gazebo ROS 2 Integration

```bash
# Install Gazebo ROS 2 packages
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros2-control

# Install additional dependencies
sudo apt install ros-humble-gazebo-msgs ros-humble-gazebo-plugins
```

## Basic Gazebo Usage

### Launching Gazebo

```bash
# Launch Gazebo with GUI
gz sim

# Launch Gazebo with a specific world
gz sim -r -v 4 shapes.sdf

# Launch Gazebo without GUI (headless)
gz sim -s

# Launch with verbose output
gz sim -v 4
```

### Command Line Options

Common command line options:
- `-r`: Run simulation immediately
- `-v N`: Set verbosity level (0-4)
- `-s`: Run in headless mode (no GUI)
- `-g`: Run GUI in separate process
- `--iterations N`: Run for N simulation iterations

## World Files and SDF Format

### SDF (Simulation Description Format)

SDF is the XML-based format used to describe simulation worlds in Gazebo:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="basic_world">
    <!-- Include default ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Include default light source -->
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
  </world>
</sdf>
```

### Creating a Custom World

Create `~/gazebo_ws/worlds/basic_room.sdf`:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="basic_room">
    <!-- Physics properties -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <!-- Include default models -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Room walls -->
    <model name="wall_north">
      <pose>0 5 1 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>10 0.2 2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>10 0.2 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>1</mass>
          <inertia>
            <ixx>1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>1</iyy>
            <iyz>0</iyz>
            <izz>1</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <model name="wall_south">
      <pose>0 -5 1 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>10 0.2 2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>10 0.2 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>1</mass>
          <inertia>
            <ixx>1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>1</iyy>
            <iyz>0</iyz>
            <izz>1</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <model name="wall_east">
      <pose>5 0 1 0 0 1.57</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>10 0.2 2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>10 0.2 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>1</mass>
          <inertia>
            <ixx>1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>1</iyy>
            <iyz>0</iyz>
            <izz>1</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <model name="wall_west">
      <pose>-5 0 1 0 0 1.57</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>10 0.2 2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>10 0.2 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>1</mass>
          <inertia>
            <ixx>1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>1</iyy>
            <iyz>0</iyz>
            <izz>1</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <!-- Add a table in the center -->
    <model name="table">
      <pose>0 0 0.5 0 0 0</pose>
      <link name="table_top">
        <collision name="collision">
          <geometry>
            <box>
              <size>2 1 0.1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>2 1 0.1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.6 0.2 1</ambient>
            <diffuse>0.9 0.7 0.3 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>10</mass>
          <inertia>
            <ixx>1.833</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>4.167</iyy>
            <iyz>0</iyz>
            <izz>5.667</izz>
          </inertia>
        </inertial>
      </link>
      <link name="leg1">
        <pose>-0.8 -0.4 0.25 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box>
              <size>0.1 0.1 0.5</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.1 0.1 0.5</size>
            </box>
          </geometry>
          <material>
            <ambient>0.2 0.2 0.2 1</ambient>
            <diffuse>0.3 0.3 0.3 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>1</mass>
          <inertia>
            <ixx>0.0208</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.0208</iyy>
            <iyz>0</iyz>
            <izz>0.005</izz>
          </inertia>
        </inertial>
      </link>
      <link name="leg2">
        <pose>0.8 -0.4 0.25 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box>
              <size>0.1 0.1 0.5</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.1 0.1 0.5</size>
            </box>
          </geometry>
          <material>
            <ambient>0.2 0.2 0.2 1</ambient>
            <diffuse>0.3 0.3 0.3 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>1</mass>
          <inertia>
            <ixx>0.0208</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.0208</iyy>
            <iyz>0</iyz>
            <izz>0.005</izz>
          </inertia>
        </inertial>
      </link>
      <link name="leg3">
        <pose>-0.8 0.4 0.25 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box>
              <size>0.1 0.1 0.5</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.1 0.1 0.5</size>
            </box>
          </geometry>
          <material>
            <ambient>0.2 0.2 0.2 1</ambient>
            <diffuse>0.3 0.3 0.3 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>1</mass>
          <inertia>
            <ixx>0.0208</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.0208</iyy>
            <iyz>0</iyz>
            <izz>0.005</izz>
          </inertia>
        </inertial>
      </link>
      <link name="leg4">
        <pose>0.8 0.4 0.25 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box>
              <size>0.1 0.1 0.5</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.1 0.1 0.5</size>
            </box>
          </geometry>
          <material>
            <ambient>0.2 0.2 0.2 1</ambient>
            <diffuse>0.3 0.3 0.3 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>1</mass>
          <inertia>
            <ixx>0.0208</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.0208</iyy>
            <iyz>0</iyz>
            <izz>0.005</izz>
          </inertia>
        </inertial>
      </link>
      <joint name="top_to_leg1" type="fixed">
        <parent>table_top</parent>
        <child>leg1</child>
      </joint>
      <joint name="top_to_leg2" type="fixed">
        <parent>table_top</parent>
        <child>leg2</child>
      </joint>
      <joint name="top_to_leg3" type="fixed">
        <parent>table_top</parent>
        <child>leg3</child>
      </joint>
      <joint name="top_to_leg4" type="fixed">
        <parent>table_top</parent>
        <child>leg4</child>
      </joint>
    </model>
  </world>
</sdf>
```

### Launching Custom World

```bash
# Launch with your custom world
gz sim -r -v 4 ~/gazebo_ws/worlds/basic_room.sdf
```

## Robot Integration with Gazebo

### Adding Gazebo Plugins to URDF

To integrate your robot with Gazebo, add the following plugins to your URDF:

```xml
<!-- In your robot URDF file -->
<robot name="my_robot">
  <!-- Your robot links and joints -->

  <!-- Gazebo ROS Control Plugin -->
  <gazebo>
    <plugin filename="libgazebo_ros2_control.so" name="gazebo_ros2_control">
      <parameters>$(find robot_description)/config/controllers.yaml</parameters>
    </plugin>
  </gazebo>

  <!-- Gazebo Materials -->
  <gazebo reference="base_link">
    <material>Gazebo/Blue</material>
  </gazebo>

  <gazebo reference="wheel_link">
    <material>Gazebo/Black</material>
  </gazebo>
</robot>
```

### Creating a Gazebo Launch File

Create `~/physical_ai_ws/src/robot_description/launch/gazebo.launch.py`:

```python
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )

    world = DeclareLaunchArgument(
        'world',
        default_value='empty.sdf',
        description='Choose one of the world files from `/usr/share/gazebo-11/worlds` or specify full path'
    )

    # Get URDF via xacro
    robot_description_content = Command([
        'xacro ',
        PathJoinSubstitution([
            FindPackageShare('robot_description'),
            'urdf',
            'simple_robot.urdf'
        ])
    ])

    robot_description = {'robot_description': robot_description_content}

    # Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='both',
        parameters=[robot_description, {'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )

    # Gazebo Server
    gzserver = ExecuteProcess(
        cmd=['gzserver',
             '-s', 'libgazebo_ros_init.so',
             '-s', 'libgazebo_ros_factory.so',
             LaunchConfiguration('world')],
        output='screen'
    )

    # Gazebo Client
    gzclient = ExecuteProcess(
        cmd=['gzclient'],
        output='screen',
        condition=IfCondition('true')  # Change to LaunchConfiguration('gui') if you want to control GUI
    )

    # Spawn Robot
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description',
                   '-entity', 'my_robot',
                   '-x', '0.0',
                   '-y', '0.0',
                   '-z', '0.5'],
        output='screen'
    )

    return LaunchDescription([
        use_sim_time,
        world,
        robot_state_publisher,
        gzserver,
        gzclient,
        spawn_robot
    ])
```

## Physics Configuration

### Physics Engine Selection

Gazebo supports multiple physics engines. Configure in your world file:

```xml
<world name="physics_example">
  <!-- ODE Physics Engine -->
  <physics type="ode">
    <max_step_size>0.001</max_step_size>
    <real_time_factor>1</real_time_factor>
    <real_time_update_rate>1000</real_time_update_rate>
    <ode>
      <solver>
        <type>quick</type>
        <iters>10</iters>
        <sor>1.3</sor>
      </solver>
      <constraints>
        <cfm>0</cfm>
        <erp>0.2</erp>
        <contact_max_correcting_vel>100</contact_max_correcting_vel>
        <contact_surface_layer>0.001</contact_surface_layer>
      </constraints>
    </ode>
  </physics>

  <!-- Bullet Physics Engine -->
  <physics type="bullet">
    <max_step_size>0.001</max_step_size>
    <real_time_factor>1</real_time_factor>
    <real_time_update_rate>1000</real_time_update_rate>
    <bullet>
      <solver>
        <type>sequential_impulse</type>
        <iterations>50</iterations>
      </solver>
      <constraints>
        <cfm>0</cfm>
        <erp>0.2</erp>
      </constraints>
    </bullet>
  </physics>
</world>
```

### Collision Properties

Configure collision properties in your models:

```xml
<model name="collision_example">
  <link name="link">
    <collision name="collision">
      <geometry>
        <box>
          <size>1 1 1</size>
        </box>
      </geometry>
      <!-- Surface properties -->
      <surface>
        <friction>
          <ode>
            <mu>1.0</mu>
            <mu2>1.0</mu2>
            <fdir1>0 0 0</fdir1>
            <slip1>0</slip1>
            <slip2>0</slip2>
          </ode>
          <torsional>
            <coefficient>1.0</coefficient>
            <use_patch_radius>1</use_patch_radius>
          </torsional>
        </friction>
        <bounce>
          <restitution_coefficient>0.1</restitution_coefficient>
          <threshold>100000</threshold>
        </bounce>
        <contact>
          <ode>
            <min_depth>0.001</min_depth>
            <max_vel>100</max_vel>
          </ode>
        </contact>
      </surface>
    </collision>
  </link>
</model>
```

## Sensor Integration

### Camera Sensor

Add a camera sensor to your robot:

```xml
<model name="robot_with_camera">
  <link name="camera_link">
    <visual name="visual">
      <geometry>
        <box>
          <size>0.05 0.05 0.05</size>
        </box>
      </geometry>
    </visual>
    <collision name="collision">
      <geometry>
        <box>
          <size>0.05 0.05 0.05</size>
        </box>
      </geometry>
    </collision>
    <sensor name="camera" type="camera">
      <camera>
        <horizontal_fov>1.047</horizontal_fov> <!-- 60 degrees -->
        <image>
          <width>640</width>
          <height>480</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.1</near>
          <far>10</far>
        </clip>
      </camera>
      <always_on>1</always_on>
      <update_rate>30</update_rate>
      <visualize>true</visualize>
    </sensor>
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
  <joint name="camera_joint" type="fixed">
    <parent>base_link</parent>
    <child>camera_link</child>
    <origin xyz="0.1 0 0.1" rpy="0 0 0"/>
  </joint>
</model>
```

### LiDAR Sensor

Add a LiDAR sensor to your robot:

```xml
<sensor name="lidar" type="ray">
  <ray>
    <scan>
      <horizontal>
        <samples>360</samples>
        <resolution>1</resolution>
        <min_angle>-3.14159</min_angle>
        <max_angle>3.14159</max_angle>
      </horizontal>
    </scan>
    <range>
      <min>0.1</min>
      <max>30</max>
      <resolution>0.01</resolution>
    </range>
  </ray>
  <always_on>1</always_on>
  <update_rate>10</update_rate>
  <visualize>true</visualize>
</sensor>
```

### IMU Sensor

Add an IMU sensor to your robot:

```xml
<sensor name="imu" type="imu">
  <always_on>1</always_on>
  <update_rate>100</update_rate>
  <imu>
    <angular_velocity>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.01</stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.01</stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.01</stddev>
        </noise>
      </z>
    </angular_velocity>
    <linear_acceleration>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
        </noise>
      </z>
    </linear_acceleration>
  </imu>
</sensor>
```

## ROS 2 Integration

### Gazebo ROS 2 Control

To integrate ROS 2 controllers with Gazebo:

```xml
<gazebo>
  <plugin name="gazebo_ros2_control" filename="libgazebo_ros2_control.so">
    <parameters>$(find robot_description)/config/controllers.yaml</parameters>
  </plugin>
</gazebo>
```

### Sensor Data Publishing

Sensors automatically publish data to ROS 2 topics. For example:
- Camera: `/camera/image_raw`, `/camera/camera_info`
- LiDAR: `/scan` or `/laser_scan`
- IMU: `/imu/data`

## Simulation Control

### Using gz Commands

Gazebo provides command-line tools for simulation control:

```bash
# List all topics
gz topic -l

# Echo a topic
gz topic -e /world/empty/model/ground_plane/link/ground_plane/pose/info

# Publish to a topic
gz topic -t /world/empty/create -m gz.msgs.EntityFactory -p 'name: "my_model", sdf_filename: "model.sdf"'

# List all services
gz service -l

# Call a service
gz service -s /world/empty/control -r gz.msgs.WorldControl -p 'pause: true'
```

### Simulation Services

Common Gazebo services:
- `/world/[world_name]/control` - Control simulation (pause/unpause/reset)
- `/world/[world_name]/set_physics` - Configure physics properties
- `/world/[world_name]/create` - Spawn entities
- `/world/[world_name]/delete` - Delete entities

## Performance Optimization

### Simulation Parameters

Optimize simulation performance by adjusting these parameters:

```xml
<physics type="ode">
  <!-- Smaller step size = more accurate but slower -->
  <max_step_size>0.001</max_step_size>

  <!-- Real-time factor = 1 for real-time simulation -->
  <real_time_factor>1</real_time_factor>

  <!-- Update rate = 1000 Hz is common -->
  <real_time_update_rate>1000</real_time_update_rate>

  <!-- Adjust solver parameters for performance -->
  <ode>
    <solver>
      <iters>20</iters>  <!-- Fewer iterations = faster but less accurate -->
      <sor>1.3</sor>     <!-- Successive over-relaxation parameter -->
    </solver>
  </ode>
</physics>
```

### Visualization Optimization

For headless operation or performance:

```bash
# Run without GUI
gz sim -s

# Reduce visual quality for better performance
export OGRE_RESOURCEMANAGER_STRICT=0
```

## Troubleshooting Common Issues

### Gazebo Won't Start

Common solutions:
1. Check if NVIDIA drivers are properly installed
2. Verify X11 forwarding if running remotely
3. Try running with software rendering: `export LIBGL_ALWAYS_SOFTWARE=1`

### Robot Falls Through Ground

This usually indicates:
1. Missing collision properties in URDF
2. Incorrect inertial properties
3. Physics engine issues

### Sensor Data Not Publishing

Check:
1. Gazebo ROS 2 bridge is running
2. Sensor topics are being published: `gz topic -l | grep sensor`
3. ROS 2 bridge configuration is correct

### Controller Not Working

Verify:
1. Controller configuration files are correct
2. Joint names match between URDF and controller config
3. ros2_control plugin is properly configured in URDF

## Advanced Features

### Plugins

Gazebo supports various plugins for extended functionality:

```xml
<!-- Custom plugin example -->
<model name="custom_model">
  <plugin filename="libMyCustomPlugin.so" name="my_custom_plugin">
    <param1>value1</param1>
    <param2>value2</param2>
  </plugin>
</model>
```

### Scripts

Gazebo supports simulation scripts for complex behaviors:

```cpp
// Example C++ plugin
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>

class MyModelPlugin : public gazebo::ModelPlugin
{
public:
  void Load(gazebo::physics::ModelPtr _model, sdf::ElementPtr _sdf)
  {
    this->model = _model;
    this->updateConnection = gazebo::event::Events::ConnectWorldUpdateBegin(
        std::bind(&MyModelPlugin::OnUpdate, this));
  }

  void OnUpdate()
  {
    // Custom behavior here
  }

private:
  gazebo::physics::ModelPtr model;
  gazebo::event::ConnectionPtr updateConnection;
};

GZ_REGISTER_MODEL_PLUGIN(MyModelPlugin)
```

## Best Practices

### 1. World Design

- Keep worlds simple for better performance
- Use appropriate physics properties
- Include visual references for orientation

### 2. Robot Modeling

- Ensure proper inertial properties
- Include collision geometry
- Test with simple shapes first

### 3. Sensor Configuration

- Add realistic noise models
- Configure appropriate update rates
- Test sensor data validity

### 4. Performance

- Use appropriate physics parameters
- Consider running headless for automated testing
- Monitor simulation real-time factor

## Next Steps

After mastering Gazebo simulation basics:

1. Continue to [Isaac Sim Fundamentals](./isaac-sim-basics.md) to learn about advanced simulation features
2. Practice creating custom simulation environments
3. Integrate your robot with Gazebo simulation
4. Test control systems in simulated environments

Your understanding of Gazebo simulation is now foundational for robot testing and development in the Physical AI and Humanoid Robotics course!
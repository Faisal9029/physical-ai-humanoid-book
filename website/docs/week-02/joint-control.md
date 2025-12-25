---
sidebar_position: 3
---

# Joint Control Concepts

This guide covers the fundamentals of robot joint control using ROS 2. Understanding joint control is essential for making your robot models move and interact with the environment.

## Overview

Joint control is the process of commanding robot joints to move to specific positions, velocities, or apply specific torques. In ROS 2, joint control is implemented using the ros2_control framework, which provides:

- Hardware abstraction
- Controller management
- Real-time safety features
- Simulation integration

## Joint Control Architecture

### ros2_control Components

The ros2_control framework consists of several key components:

1. **Hardware Interface**: Communicates with physical or simulated hardware
2. **Controller Manager**: Manages available controllers
3. **Controllers**: Implement specific control algorithms
4. **Joint State Publisher**: Publishes current joint states

### Control Hierarchy

```
Hardware (Real Robot or Simulation)
    ↓
Hardware Interface
    ↓
Controller Manager
    ↓
Controllers (JointTrajectoryController, JointStateController, etc.)
    ↓
ROS 2 Topics/Services (command and feedback)
```

## Joint State Message

The `sensor_msgs/JointState` message is fundamental to joint control:

```python
# JointState message definition
std_msgs/Header header
string[] name          # Joint names
float64[] position    # Joint positions (radians or meters)
float64[] velocity    # Joint velocities (rad/s or m/s)
float64[] effort      # Joint efforts (torque or force)
```

### Publishing Joint States

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

class JointStatePublisher(Node):
    def __init__(self):
        super().__init__('joint_state_publisher')

        # Create publisher for joint states
        self.joint_state_publisher = self.create_publisher(
            JointState,
            'joint_states',
            10
        )

        # Timer for publishing joint states
        self.timer = self.create_timer(0.01, self.publish_joint_states)

        # Initialize joint names
        self.joint_names = ['joint1', 'joint2', 'joint3']
        self.joint_positions = [0.0, 0.0, 0.0]
        self.joint_velocities = [0.0, 0.0, 0.0]
        self.joint_efforts = [0.0, 0.0, 0.0]

    def publish_joint_states(self):
        # Create joint state message
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        msg.name = self.joint_names
        msg.position = self.joint_positions
        msg.velocity = self.joint_velocities
        msg.effort = self.joint_efforts

        # Publish the message
        self.joint_state_publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = JointStatePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Joint Control Types

### 1. Position Control

Position control commands joints to move to specific positions:

```python
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

class PositionController(Node):
    def __init__(self):
        super().__init__('position_controller')

        # Create publisher for trajectory commands
        self.trajectory_publisher = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )

        # Timer to send position commands
        self.timer = self.create_timer(2.0, self.send_position_command)

    def send_position_command(self):
        # Create trajectory message
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = ['joint1', 'joint2', 'joint3']

        # Create trajectory point
        point = JointTrajectoryPoint()
        point.positions = [1.57, 0.0, -1.57]  # Target positions
        point.velocities = [0.0, 0.0, 0.0]    # Desired velocities
        point.accelerations = [0.0, 0.0, 0.0] # Desired accelerations
        point.time_from_start = Duration(sec=1, nanosec=0)  # Move duration

        trajectory_msg.points = [point]
        trajectory_msg.header.stamp = self.get_clock().now().to_msg()

        # Publish trajectory
        self.trajectory_publisher.publish(trajectory_msg)
        self.get_logger().info('Published position command')

def main(args=None):
    rclpy.init(args=args)
    node = PositionController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 2. Velocity Control

Velocity control commands joints to move at specific velocities:

```python
class VelocityController(Node):
    def __init__(self):
        super().__init__('velocity_controller')

        self.velocity_publisher = self.create_publisher(
            JointTrajectory,
            '/velocity_controller/joint_trajectory',
            10
        )

        self.timer = self.create_timer(0.1, self.send_velocity_command)

    def send_velocity_command(self):
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = ['joint1', 'joint2']

        point = JointTrajectoryPoint()
        point.positions = [0.0, 0.0]           # Current positions (for interpolation)
        point.velocities = [0.5, -0.3]         # Target velocities
        point.time_from_start = Duration(sec=0, nanosec=100000000)  # 0.1 seconds

        trajectory_msg.points = [point]
        trajectory_msg.header.stamp = self.get_clock().now().to_msg()

        self.velocity_publisher.publish(trajectory_msg)
```

### 3. Effort Control

Effort control applies specific torques or forces to joints:

```python
class EffortController(Node):
    def __init__(self):
        super().__init__('effort_controller')

        self.effort_publisher = self.create_publisher(
            JointTrajectory,
            '/effort_controller/joint_trajectory',
            10
        )

        self.timer = self.create_timer(0.01, self.send_effort_command)

    def send_effort_command(self):
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = ['joint1']

        point = JointTrajectoryPoint()
        point.efforts = [5.0]  # Apply 5.0 Nm of torque
        point.time_from_start = Duration(sec=0, nanosec=10000000)  # 0.01 seconds

        trajectory_msg.points = [point]
        trajectory_msg.header.stamp = self.get_clock().now().to_msg()

        self.effort_publisher.publish(trajectory_msg)
```

## ros2_control Configuration

### Controller Configuration File

Create `~/physical_ai_ws/src/robot_description/config/controllers.yaml`:

```yaml
controller_manager:
  ros__parameters:
    update_rate: 100  # Hz

    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster

    joint_trajectory_controller:
      type: joint_trajectory_controller/JointTrajectoryController

    velocity_controller:
      type: velocity_controllers/JointGroupVelocityController

    effort_controller:
      type: effort_controllers/JointGroupEffortController

joint_trajectory_controller:
  ros__parameters:
    joints:
      - joint1
      - joint2
      - joint3

    interface_name: position

    command_interfaces:
      - position

    state_interfaces:
      - position
      - velocity

velocity_controller:
  ros__parameters:
    joints:
      - joint1
      - joint2

effort_controller:
  ros__parameters:
    joints:
      - joint1
```

### Robot Hardware Configuration

Create `~/physical_ai_ws/src/robot_description/config/robot_hardware.yaml`:

```yaml
simple_robot:
  ros__parameters:
    # Use the gazebo simulation plugin
    gazebo:
      ros__parameters:
        # The name of the robot in Gazebo
        robot_name: simple_robot
        # The prefix to be used for the links, joints and transmissions of the robot
        prefix: ""
        # The update rate for the robot
        update_rate: 100
        # The position of the robot in the Gazebo world
        initial_position:
          joint1: 0.0
          joint2: 0.0
          joint3: 0.0

    # Define the robot's transmissions
    transmissions:
      simple_transmission1:
        type: transmission_interface/SimpleTransmission
        joint: joint1
        actuator: joint1_motor
        mechanical_reduction: 1
      simple_transmission2:
        type: transmission_interface/SimpleTransmission
        joint: joint2
        actuator: joint2_motor
        mechanical_reduction: 1
```

## Joint Trajectory Controller

The JointTrajectoryController is the most commonly used controller for precise motion control:

### Configuration

```yaml
joint_trajectory_controller:
  ros__parameters:
    joints:
      - joint1
      - joint2
      - joint3

    command_interfaces:
      - position

    state_interfaces:
      - position
      - velocity

    # Allow commands to be preempted by new commands
    allow_partial_joints_goal: false

    # Goal time tolerance
    goal_time_tolerance: 0.5

    # Constrained joints (for closed chains)
    constraints:
      stopped_velocity_tolerance: 0.01
      joint1:
        trajectory: 0.05
        goal: 0.01
      joint2:
        trajectory: 0.05
        goal: 0.01
      joint3:
        trajectory: 0.05
        goal: 0.01
```

### Commanding Joint Trajectories

```python
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import math

class TrajectoryCommander(Node):
    def __init__(self):
        super().__init__('trajectory_commander')

        self.trajectory_publisher = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )

        # Send trajectory after a delay
        self.timer = self.create_timer(1.0, self.send_trajectory)

    def send_trajectory(self):
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = ['joint1', 'joint2', 'joint3']

        # Create multiple trajectory points for smooth motion
        points = []

        # Point 1: Start position
        point1 = JointTrajectoryPoint()
        point1.positions = [0.0, 0.0, 0.0]
        point1.velocities = [0.0, 0.0, 0.0]
        point1.time_from_start = Duration(sec=0, nanosec=0)
        points.append(point1)

        # Point 2: Midpoint
        point2 = JointTrajectoryPoint()
        point2.positions = [0.5, 0.3, -0.2]
        point2.velocities = [0.0, 0.0, 0.0]
        point2.time_from_start = Duration(sec=2, nanosec=0)
        points.append(point2)

        # Point 3: Final position
        point3 = JointTrajectoryPoint()
        point3.positions = [1.0, 0.6, -0.4]
        point3.velocities = [0.0, 0.0, 0.0]
        point3.time_from_start = Duration(sec=4, nanosec=0)
        points.append(point3)

        trajectory_msg.points = points
        trajectory_msg.header.stamp = self.get_clock().now().to_msg()

        self.trajectory_publisher.publish(trajectory_msg)
        self.get_logger().info('Published trajectory command')

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryCommander()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Joint Limits and Safety

### Joint Limit Enforcement

Joint limits are critical for robot safety:

```yaml
# In the URDF file or separate limits file
joint_limits:
  joint1:
    has_position_limits: true
    min_position: -1.57
    max_position: 1.57
    has_velocity_limits: true
    max_velocity: 1.0
    has_acceleration_limits: true
    max_acceleration: 2.0
    has_effort_limits: true
    max_effort: 100.0
```

### Safety Controllers

```python
class JointSafetyController(Node):
    def __init__(self):
        super().__init__('joint_safety_controller')

        # Subscribe to joint states
        self.joint_state_subscriber = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10
        )

        # Publisher for emergency stops
        self.emergency_stop_publisher = self.create_publisher(
            Bool,
            'emergency_stop',
            10
        )

        self.joint_limits = {
            'joint1': {'min': -1.57, 'max': 1.57},
            'joint2': {'min': -1.0, 'max': 1.0},
            'joint3': {'min': -2.0, 'max': 0.5}
        }

    def joint_state_callback(self, msg):
        for i, joint_name in enumerate(msg.name):
            if joint_name in self.joint_limits:
                position = msg.position[i]
                limits = self.joint_limits[joint_name]

                # Check if joint is within limits
                if position < limits['min'] or position > limits['max']:
                    self.get_logger().error(f'Joint {joint_name} limit exceeded: {position}')
                    # Publish emergency stop
                    stop_msg = Bool()
                    stop_msg.data = True
                    self.emergency_stop_publisher.publish(stop_msg)
```

## Simulation Integration

### Gazebo ROS 2 Control

To integrate with Gazebo simulation, add the following plugin to your URDF:

```xml
<gazebo>
  <plugin filename="libgazebo_ros2_control.so" name="gazebo_ros2_control">
    <parameters>$(find robot_description)/config/controllers.yaml</parameters>
  </plugin>
</gazebo>
```

### Isaac Sim Integration

For Isaac Sim, use the appropriate ROS bridge components:

```python
# Isaac Sim ROS 2 bridge configuration
from omni.isaac.ros2_bridge import get_ros2_context

# Configure joint control topics
# joint_states, /joint_trajectory_controller/joint_trajectory, etc.
```

## Practical Example: Controlling a Robot Arm

### Complete Control Node

Create `~/physical_ai_ws/src/robot_description/scripts/arm_controller.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration
import math

class RobotArmController(Node):
    def __init__(self):
        super().__init__('robot_arm_controller')

        # Publisher for joint trajectory commands
        self.trajectory_publisher = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )

        # Subscriber for joint states
        self.joint_state_subscriber = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10
        )

        # Timer for periodic control updates
        self.control_timer = self.create_timer(0.1, self.control_loop)

        # Initialize joint state
        self.current_positions = {}
        self.joint_names = ['joint1', 'joint2', 'joint3']

        # Trajectory parameters
        self.trajectory_time = 0.0
        self.trajectory_period = 10.0  # seconds for full trajectory

        self.get_logger().info('Robot Arm Controller initialized')

    def joint_state_callback(self, msg):
        """Update current joint positions"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_positions[name] = msg.position[i]

    def control_loop(self):
        """Main control loop"""
        # Update trajectory time
        self.trajectory_time += 0.1

        # Calculate trajectory based on time
        if self.trajectory_time > self.trajectory_period:
            self.trajectory_time = 0.0

        # Generate a sinusoidal trajectory for demonstration
        t = self.trajectory_time / self.trajectory_period * 2 * math.pi

        target_positions = [
            math.sin(t) * 1.0,      # Joint 1: sin wave
            math.sin(t * 0.5) * 0.5, # Joint 2: slower sin wave
            math.sin(t * 0.3) * 0.3  # Joint 3: even slower
        ]

        # Send trajectory command
        self.send_trajectory_command(target_positions)

    def send_trajectory_command(self, positions):
        """Send a single point trajectory command"""
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = positions
        point.velocities = [0.0, 0.0, 0.0]  # Zero velocity at goal
        point.time_from_start = Duration(sec=0, nanosec=100000000)  # 0.1 seconds

        trajectory_msg.points = [point]
        trajectory_msg.header.stamp = self.get_clock().now().to_msg()

        self.trajectory_publisher.publish(trajectory_msg)

def main(args=None):
    rclpy.init(args=args)
    controller = RobotArmController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down Robot Arm Controller')
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Launch Files for Joint Control

### Control Launch File

Create `~/physical_ai_ws/src/robot_description/launch/control.launch.py`:

```python
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Declare launch arguments
    urdf_model_path = DeclareLaunchArgument(
        'urdf_model',
        default_value='simple_robot.urdf',
        description='URDF file name'
    )

    # Get URDF path
    robot_description_config = Command([
        'xacro ',
        os.path.join(
            get_package_share_directory('robot_description'),
            'urdf',
            LaunchConfiguration('urdf_model')
        )
    ])

    # Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': robot_description_config
        }]
    )

    # Controller Manager
    controller_manager = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[{
            'robot_description': robot_description_config,
            os.path.join(
                get_package_share_directory('robot_description'),
                'config',
                'controllers.yaml'
            )
        }],
        output='both'
    )

    # Joint State Broadcaster Spawner
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
        output='screen'
    )

    # Joint Trajectory Controller Spawner
    joint_trajectory_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_trajectory_controller'],
        output='screen'
    )

    # Robot Arm Controller
    arm_controller = Node(
        package='robot_description',
        executable='arm_controller.py',
        output='screen'
    )

    # Delay spawning of joint_state_broadcaster until after controller_manager starts
    delay_joint_state_broadcaster_spawner_after_controller_manager = RegisterEventHandler(
        event_handler=OnProcessStart(
            target_action=controller_manager,
            on_start=[
                joint_state_broadcaster_spawner,
            ],
        )
    )

    # Delay spawning of joint_trajectory_controller until after joint_state_broadcaster_spawner starts
    delay_joint_trajectory_controller_spawner_after_joint_state_broadcaster_spawner = RegisterEventHandler(
        event_handler=OnProcessStart(
            target_action=joint_state_broadcaster_spawner,
            on_start=[
                joint_trajectory_controller_spawner,
            ],
        )
    )

    # Delay spawning of arm_controller until after joint_trajectory_controller_spawner starts
    delay_arm_controller_after_joint_trajectory_controller_spawner = RegisterEventHandler(
        event_handler=OnProcessStart(
            target_action=joint_trajectory_controller_spawner,
            on_start=[
                arm_controller,
            ],
        )
    )

    return LaunchDescription([
        urdf_model_path,
        robot_state_publisher,
        controller_manager,
        delay_joint_state_broadcaster_spawner_after_controller_manager,
        delay_joint_trajectory_controller_spawner_after_joint_state_broadcaster_spawner,
        delay_arm_controller_after_joint_trajectory_controller_spawner
    ])
```

## Testing Joint Control

### Command Line Testing

Use ros2 control command-line tools to test controllers:

```bash
# List available controllers
ros2 control list_controllers

# Check controller state
ros2 control list_controller_types

# Switch controllers
ros2 control switch_controllers --start joint_trajectory_controller

# Send command to joint trajectory controller
ros2 action send_goal /joint_trajectory_controller/follow_joint_trajectory \
  control_msgs/action/FollowJointTrajectory \
  '{trajectory: {joint_names: [joint1, joint2, joint3], points: [{positions: [1.0, 0.5, -0.5], time_from_start: {sec: 2, nanosec: 0}}]}}'
```

## Troubleshooting Joint Control Issues

### Common Problems

1. **Controller Not Loading**
   - Check configuration file syntax
   - Verify joint names match URDF
   - Ensure proper startup order

2. **Joint Limits Exceeded**
   - Check URDF joint limits
   - Verify trajectory constraints
   - Implement proper safety limits

3. **Communication Issues**
   - Check topic names and namespaces
   - Verify ROS 2 domain IDs match
   - Confirm network connectivity

### Debugging Commands

```bash
# Check available topics
ros2 topic list | grep joint

# Monitor joint states
ros2 topic echo /joint_states

# Check controller status
ros2 service call /controller_manager/list_controllers \
  controller_manager_msgs/srv/ListControllers

# Monitor control commands
ros2 topic echo /joint_trajectory_controller/joint_trajectory
```

## Next Steps

After mastering joint control concepts:

1. Continue to [Forward and Inverse Kinematics](./kinematics.md) to understand robot motion mathematics
2. Practice creating more complex control patterns
3. Explore advanced control techniques
4. Learn about sensor integration in Week 5

Your understanding of joint control is now fundamental for making your robots move and interact with the environment in the Physical AI and Humanoid Robotics course!
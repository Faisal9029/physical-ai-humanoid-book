---
sidebar_position: 2
---

# ROS 2 Controllers

This guide covers the fundamentals of ROS 2 controllers and how to configure and use them for robot control. Controllers are essential components that command robot joints and mechanisms to achieve desired behaviors.

## Overview

ROS 2 controllers are software components that implement specific control algorithms to command robot hardware. The ros2_control framework provides a standardized way to create, configure, and manage these controllers. This framework enables the same controllers to work with both simulated and real robots.

### Key Components

The ros2_control framework consists of several key components:

1. **Hardware Interface**: Abstraction layer for communicating with physical or simulated hardware
2. **Controller Manager**: Runtime manager for loading, unloading, and switching controllers
3. **Controllers**: Software components that implement control algorithms
4. **Resource Manager**: Tracks available joints and interfaces

### Control Architecture

```
Hardware (Real Robot or Simulation)
    ↓
Hardware Interface (ros2_control)
    ↓
Controller Manager
    ↓
Controllers (JointTrajectoryController, etc.)
    ↓
ROS 2 Topics/Services (command and feedback)
```

## Controller Types

### Joint Trajectory Controller

The JointTrajectoryController is the most commonly used controller for precise motion control. It follows smooth trajectories with position, velocity, and acceleration profiles.

#### Configuration

Create `~/physical_ai_ws/src/robot_description/config/joint_trajectory_controller.yaml`:

```yaml
joint_trajectory_controller:
  ros__parameters:
    # List of joints to control
    joints:
      - joint1
      - joint2
      - joint3

    # Command interfaces to use
    command_interfaces:
      - position

    # State interfaces to use
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

#### Usage Example

```python
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import math

class TrajectoryControllerNode(Node):
    def __init__(self):
        super().__init__('trajectory_controller_node')

        # Publisher for joint trajectory commands
        self.trajectory_publisher = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )

        # Timer to send trajectory commands
        self.timer = self.create_timer(5.0, self.send_trajectory)

    def send_trajectory(self):
        """Send a trajectory command to the controller"""
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = ['joint1', 'joint2', 'joint3']

        # Create trajectory points
        points = []

        # Start position
        start_point = JointTrajectoryPoint()
        start_point.positions = [0.0, 0.0, 0.0]
        start_point.velocities = [0.0, 0.0, 0.0]
        start_point.time_from_start = Duration(sec=0, nanosec=0)
        points.append(start_point)

        # Midpoint
        mid_point = JointTrajectoryPoint()
        mid_point.positions = [0.5, 0.3, -0.2]
        mid_point.velocities = [0.0, 0.0, 0.0]
        mid_point.time_from_start = Duration(sec=2, nanosec=0)
        points.append(mid_point)

        # Final position
        final_point = JointTrajectoryPoint()
        final_point.positions = [1.0, 0.6, -0.4]
        final_point.velocities = [0.0, 0.0, 0.0]
        final_point.time_from_start = Duration(sec=4, nanosec=0)
        points.append(final_point)

        trajectory_msg.points = points
        trajectory_msg.header.stamp = self.get_clock().now().to_msg()

        self.trajectory_publisher.publish(trajectory_msg)
        self.get_logger().info('Published trajectory command')

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Joint Group Position Controller

The Joint Group Position Controller allows direct position control of multiple joints simultaneously.

#### Configuration

Create `~/physical_ai_ws/src/robot_description/config/joint_group_position_controller.yaml`:

```yaml
joint_group_position_controller:
  ros__parameters:
    joints:
      - joint1
      - joint2
      - joint3

    interface_name: position
```

### Joint Group Velocity Controller

The Joint Group Velocity Controller allows direct velocity control of multiple joints.

#### Configuration

```yaml
velocity_controller:
  ros__parameters:
    joints:
      - joint1
      - joint2
```

### Joint Group Effort Controller

The Joint Group Effort Controller allows direct torque/force control of multiple joints.

#### Configuration

```yaml
effort_controller:
  ros__parameters:
    joints:
      - joint1
```

## Controller Manager Configuration

### Main Controller Configuration

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

# Include individual controller configurations
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

## Creating Custom Controllers

### Basic Controller Structure

To create a custom controller, you need to inherit from the appropriate base class:

```python
from controller_interface import ControllerInterface
from hardware_interface import HardwareInterface
import rclpy
from rclpy.lifecycle import LifecycleState
from controller_interface import return_type

class CustomController(ControllerInterface):
    def __init__(self):
        super().__init__()

    def init(self, controller_name):
        """Initialize the controller"""
        self.controller_name = controller_name
        return return_type.OK

    def update(self, time, period):
        """Main control loop"""
        # Implement your control algorithm here
        # Read from hardware interface
        # Apply control commands
        return return_type.OK

    def on_configure(self, state: LifecycleState) -> return_type:
        """Configure the controller"""
        return return_type.OK

    def on_activate(self, state: LifecycleState) -> return_type:
        """Activate the controller"""
        return return_type.OK

    def on_deactivate(self, state: LifecycleState) -> return_type:
        """Deactivate the controller"""
        return return_type.OK
```

### PID Controller Example

Here's a simple PID controller implementation:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
import numpy as np

class PIDControllerNode(Node):
    def __init__(self):
        super().__init__('pid_controller_node')

        # Controller parameters
        self.kp = [10.0, 10.0, 10.0]  # Proportional gains
        self.ki = [0.1, 0.1, 0.1]     # Integral gains
        self.kd = [1.0, 1.0, 1.0]     # Derivative gains

        # Controller state
        self.error_integral = np.zeros(3)
        self.previous_error = np.zeros(3)
        self.command = np.zeros(3)

        # Publishers and subscribers
        self.command_publisher = self.create_publisher(
            Float64MultiArray, '/position_commands', 10
        )

        self.joint_state_subscriber = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )

        # Timer for control loop
        self.control_timer = self.create_timer(0.01, self.control_loop)

        # Desired positions
        self.desired_positions = np.array([1.0, 0.5, -0.5])

    def joint_state_callback(self, msg):
        """Update current joint positions"""
        if len(msg.position) >= 3:
            self.current_positions = np.array(msg.position[:3])

    def control_loop(self):
        """PID control loop"""
        if hasattr(self, 'current_positions'):
            # Calculate error
            error = self.desired_positions - self.current_positions

            # Update integral term
            self.error_integral += error * 0.01  # dt = 0.01

            # Calculate derivative term
            derivative = (error - self.previous_error) / 0.01

            # Calculate PID output
            self.command = (self.kp * error +
                           self.ki * self.error_integral +
                           self.kd * derivative)

            # Publish command
            cmd_msg = Float64MultiArray()
            cmd_msg.data = self.command.tolist()
            self.command_publisher.publish(cmd_msg)

            # Store current error for next derivative calculation
            self.previous_error = error

def main(args=None):
    rclpy.init(args=args)
    node = PIDControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Controller Spawning and Management

### Spawning Controllers

Controllers are typically spawned using the controller_manager spawner:

```bash
# Spawn a single controller
ros2 run controller_manager spawner joint_trajectory_controller

# Spawn multiple controllers
ros2 run controller_manager spawner joint_trajectory_controller joint_state_broadcaster

# Spawn with namespace
ros2 run controller_manager spawner joint_trajectory_controller --ros-args -r __ns:=/my_robot
```

### Controller Management Commands

```bash
# List available controllers
ros2 control list_controllers

# List controller types
ros2 control list_controller_types

# Switch controllers
ros2 control switch_controllers --start joint_trajectory_controller --stop velocity_controller

# Load controller without starting
ros2 control load_controller joint_trajectory_controller

# Unload controller
ros2 control unload_controller joint_trajectory_controller
```

## Controller Configuration in Launch Files

### Complete Launch File

Create `~/physical_ai_ws/src/robot_description/launch/controllers.launch.py`:

```python
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Declare launch arguments
    urdf_model = DeclareLaunchArgument(
        'urdf_model',
        default_value='simple_robot.urdf',
        description='URDF file name'
    )

    # Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': os.path.join(
                get_package_share_directory('robot_description'),
                'urdf',
                LaunchConfiguration('urdf_model')
            )
        }]
    )

    # Controller Manager
    controller_manager = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[
            os.path.join(
                get_package_share_directory('robot_description'),
                'config',
                'controllers.yaml'
            )
        ],
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

    # Velocity Controller Spawner
    velocity_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['velocity_controller'],
        output='screen'
    )

    # Delay spawning of controllers to ensure proper startup order
    delay_joint_state_broadcaster_spawner_after_controller_manager = RegisterEventHandler(
        event_handler=OnProcessStart(
            target_action=controller_manager,
            on_start=[
                joint_state_broadcaster_spawner,
            ],
        )
    )

    delay_joint_trajectory_controller_spawner_after_joint_state_broadcaster_spawner = RegisterEventHandler(
        event_handler=OnProcessStart(
            target_action=joint_state_broadcaster_spawner,
            on_start=[
                joint_trajectory_controller_spawner,
            ],
        )
    )

    delay_velocity_controller_spawner_after_joint_trajectory_controller_spawner = RegisterEventHandler(
        event_handler=OnProcessStart(
            target_action=joint_trajectory_controller_spawner,
            on_start=[
                velocity_controller_spawner,
            ],
        )
    )

    return LaunchDescription([
        urdf_model,
        robot_state_publisher,
        controller_manager,
        delay_joint_state_broadcaster_spawner_after_controller_manager,
        delay_joint_trajectory_controller_spawner_after_joint_state_broadcaster_spawner,
        delay_velocity_controller_spawner_after_joint_trajectory_controller_spawner
    ])
```

## Advanced Controller Concepts

### Hardware Interface Configuration

For real robots, you need to configure the hardware interface:

```yaml
# In robot hardware configuration
robot_hardware:
  ros__parameters:
    # Hardware interface type
    interface: 'velocity'

    # Joint mapping
    joints:
      - name: 'joint1'
        id: 1
        type: 'revolute'
        limits:
          min_position: -3.14
          max_position: 3.14
          max_velocity: 1.0
          max_effort: 100.0
```

### Real-time Performance

For real-time applications, consider:

- Set appropriate update rates (typically 100-1000 Hz)
- Use real-time kernel if needed
- Optimize controller algorithms
- Monitor control loop timing

```python
def update(self, time, period):
    """Time-critical update function"""
    start_time = self.get_clock().now()

    # Your control algorithm here
    result = self.control_algorithm()

    end_time = self.get_clock().now()
    execution_time = (end_time - start_time).nanoseconds / 1e9

    if execution_time > period:
        self.get_logger().warn(f'Control loop exceeded period: {execution_time:.4f}s > {period:.4f}s')

    return result
```

## Simulation Integration

### Gazebo Integration

For Gazebo simulation, add the ros2_control plugin to your URDF:

```xml
<!-- In your robot URDF -->
<gazebo>
  <plugin filename="libgazebo_ros2_control.so" name="gazebo_ros2_control">
    <parameters>$(find robot_description)/config/controllers.yaml</parameters>
  </plugin>
</gazebo>
```

### Isaac Sim Integration

For Isaac Sim, use the appropriate ROS bridge components:

```python
# Isaac Sim controller configuration would go here
# This typically involves setting up the appropriate ROS 2 bridge components
```

## Controller Performance Tuning

### PID Tuning

For PID controllers, common tuning methods include:

```python
def tune_pid_controller(self):
    """Example of PID tuning approach"""
    # Ziegler-Nichols method
    # 1. Set Ki and Kd to 0
    # 2. Increase Kp until oscillation occurs
    # 3. Record critical gain (Kc) and oscillation period (Pc)
    # 4. Use Ziegler-Nichols formulas:
    #    Kp = 0.6 * Kc
    #    Ki = 2 * Kp / Pc
    #    Kd = Kp * Pc / 8

    # Or use more advanced methods like:
    # - Model-based tuning
    # - Auto-tuning algorithms
    # - Optimization-based tuning
    pass
```

### Performance Metrics

Monitor controller performance with metrics:

```python
def calculate_performance_metrics(self, desired, actual):
    """Calculate control performance metrics"""
    error = desired - actual

    # Root Mean Square Error
    rmse = np.sqrt(np.mean(error**2))

    # Mean Absolute Error
    mae = np.mean(np.abs(error))

    # Maximum error
    max_error = np.max(np.abs(error))

    return {
        'rmse': rmse,
        'mae': mae,
        'max_error': max_error
    }
```

## Troubleshooting Common Issues

### Controller Not Loading

Common causes and solutions:

1. **Missing configuration file**: Verify controller config exists and is properly formatted
2. **Incorrect joint names**: Check that joint names in config match URDF
3. **Wrong interface types**: Ensure command/state interfaces match hardware
4. **Dependencies not installed**: Install required controller packages

### Poor Performance

1. **Wrong update rate**: Adjust controller update rate in controller_manager config
2. **Poor PID parameters**: Tune PID gains appropriately
3. **Hardware limitations**: Check if hardware can follow commands
4. **Communication delays**: Reduce network/IPC delays

### Command Interface Issues

```bash
# Check available topics
ros2 topic list | grep controller

# Check topic types
ros2 topic info /joint_trajectory_controller/joint_trajectory

# Monitor command execution
ros2 topic echo /joint_trajectory_controller/joint_trajectory
```

## Best Practices

### Configuration Management

1. **Use separate config files**: Keep controller configs separate for modularity
2. **Parameter validation**: Validate parameters at startup
3. **Clear naming conventions**: Use consistent naming for controllers
4. **Documentation**: Document all parameters and their effects

### Safety Considerations

```python
def validate_command(self, command):
    """Validate controller commands for safety"""
    # Check joint limits
    for i, (cmd, limit) in enumerate(zip(command, self.joint_limits)):
        if cmd < limit['min'] or cmd > limit['max']:
            self.get_logger().error(f'Command {cmd} exceeds limit for joint {i}')
            return False

    # Check velocity limits
    if self.current_velocity is not None:
        velocity = (command - self.previous_command) / self.control_period
        if any(abs(v) > max_vel for v, max_vel in zip(velocity, self.max_velocities)):
            self.get_logger().warn('Velocity limit exceeded')
            return False

    return True
```

### Testing and Validation

1. **Unit tests**: Test controller logic independently
2. **Integration tests**: Test controller with simulated hardware
3. **Performance tests**: Verify real-time constraints
4. **Safety tests**: Validate emergency stop functionality

## Advanced Controller Types

### Cartesian Controllers

For end-effector control in Cartesian space:

```python
class CartesianController:
    def __init__(self):
        # Uses inverse kinematics to convert Cartesian commands to joint commands
        pass

    def cartesian_to_joint(self, cartesian_pose):
        """Convert Cartesian pose to joint positions using IK"""
        # Implementation would use kinematics library
        pass
```

### Adaptive Controllers

Controllers that adjust parameters based on system behavior:

```python
class AdaptiveController:
    def __init__(self):
        self.model_parameters = np.array([1.0, 1.0])  # Initial estimates
        self.learning_rate = 0.01

    def update_model(self, error, input_signal):
        """Update internal model based on observed behavior"""
        # Adaptive algorithm updates model parameters
        self.model_parameters += self.learning_rate * error * input_signal
```

## Next Steps

After mastering ROS 2 controllers:

1. Continue to [Joint State Publisher](./joint-state-publisher.md) to understand how to publish joint state information
2. Practice creating custom controllers for specific applications
3. Learn about controller switching and management
4. Explore advanced control techniques and algorithms

Your understanding of ROS 2 controllers is now fundamental for implementing precise robot motion control in the Physical AI and Humanoid Robotics course!
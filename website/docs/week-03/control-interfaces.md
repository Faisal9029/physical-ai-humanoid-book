---
sidebar_position: 5
---

# Control Interfaces

This guide covers the various control interfaces available in ROS 2 for commanding robot motion. Control interfaces define how commands are sent to robot hardware and how feedback is received. Understanding these interfaces is crucial for implementing effective robot control systems.

## Overview

Control interfaces in ROS 2 are standardized ways to command robot joints and mechanisms. The most common interfaces are:

1. **Position Interface**: Commands joint positions
2. **Velocity Interface**: Commands joint velocities
3. **Effort Interface**: Commands joint torques/forces
4. **Mixed Interfaces**: Combinations of the above

### Command Types

Different control interfaces serve different purposes:

- **Position Control**: For precise positioning tasks
- **Velocity Control**: For speed-controlled movements
- **Effort Control**: For force-controlled interactions
- **Trajectory Control**: For smooth, coordinated motion

## Position Control Interface

Position control is the most commonly used interface for precise positioning tasks.

### Configuration

Create `~/physical_ai_ws/src/robot_description/config/position_controller.yaml`:

```yaml
position_controller:
  ros__parameters:
    # List of joints to control
    joints:
      - joint1
      - joint2
      - joint3

    # Command interface type
    command_interfaces:
      - position

    # State interface type
    state_interfaces:
      - position

    # PID controller parameters
    gains:
      joint1:
        p: 100.0
        i: 0.01
        d: 10.0
      joint2:
        p: 100.0
        i: 0.01
        d: 10.0
      joint3:
        p: 100.0
        i: 0.01
        d: 10.0

    # Position tolerance
    position_tolerance: 0.01

    # Maximum velocity for smooth motion
    max_velocity: 1.0
```

### Implementation Example

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
import numpy as np

class PositionController(Node):
    def __init__(self):
        super().__init__('position_controller')

        # Publisher for position commands
        self.position_command_publisher = self.create_publisher(
            Float64MultiArray, '/position_commands', 10
        )

        # Subscriber for joint states
        self.joint_state_subscriber = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )

        # Timer for control loop
        self.control_timer = self.create_timer(0.01, self.control_loop)

        # Controller parameters
        self.kp = np.array([100.0, 100.0, 100.0])  # Proportional gains
        self.ki = np.array([0.1, 0.1, 0.1])        # Integral gains
        self.kd = np.array([10.0, 10.0, 10.0])     # Derivative gains

        # Controller state
        self.error_integral = np.zeros(3)
        self.previous_error = np.zeros(3)
        self.current_positions = np.zeros(3)
        self.desired_positions = np.array([0.5, 0.3, -0.2])

        self.get_logger().info('Position Controller initialized')

    def joint_state_callback(self, msg):
        """Update current joint positions"""
        if len(msg.position) >= 3:
            self.current_positions = np.array(msg.position[:3])

    def control_loop(self):
        """Position control loop"""
        # Calculate position error
        error = self.desired_positions - self.current_positions

        # Update integral term
        self.error_integral += error * 0.01  # dt = 0.01

        # Calculate derivative term
        derivative = (error - self.previous_error) / 0.01

        # Calculate PID output
        command = (self.kp * error +
                   self.ki * self.error_integral +
                   self.kd * derivative)

        # Publish position command
        cmd_msg = Float64MultiArray()
        cmd_msg.data = (self.current_positions + command * 0.01).tolist()  # Apply incremental change
        self.position_command_publisher.publish(cmd_msg)

        # Store current error for next derivative calculation
        self.previous_error = error

        self.get_logger().info(f'Position command: {cmd_msg.data}')

def main(args=None):
    rclpy.init(args=args)
    node = PositionController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Velocity Control Interface

Velocity control is used for speed-controlled movements and when precise position control is not required.

### Configuration

Create `~/physical_ai_ws/src/robot_description/config/velocity_controller.yaml`:

```yaml
velocity_controller:
  ros__parameters:
    # List of joints to control
    joints:
      - joint1
      - joint2

    # Command interface type
    command_interfaces:
      - velocity

    # State interface type
    state_interfaces:
      - position
      - velocity

    # Velocity limits
    velocity_limits:
      joint1: 2.0  # rad/s
      joint2: 2.0  # rad/s

    # Acceleration limits
    acceleration_limits:
      joint1: 5.0  # rad/s²
      joint2: 5.0  # rad/s²

    # PID controller parameters
    gains:
      joint1:
        p: 10.0
        i: 0.1
        d: 1.0
      joint2:
        p: 10.0
        i: 0.1
        d: 1.0
```

### Implementation Example

```python
class VelocityController(Node):
    def __init__(self):
        super().__init__('velocity_controller')

        # Publisher for velocity commands
        self.velocity_command_publisher = self.create_publisher(
            Float64MultiArray, '/velocity_commands', 10
        )

        # Subscriber for joint states
        self.joint_state_subscriber = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )

        # Timer for control loop
        self.control_timer = self.create_timer(0.01, self.control_loop)

        # Controller parameters
        self.kp = np.array([10.0, 10.0])  # Proportional gains
        self.ki = np.array([0.1, 0.1])    # Integral gains
        self.kd = np.array([1.0, 1.0])    # Derivative gains

        # Controller state
        self.error_integral = np.zeros(2)
        self.previous_error = np.zeros(2)
        self.current_velocities = np.zeros(2)
        self.desired_velocities = np.array([0.5, -0.3])  # rad/s

    def joint_state_callback(self, msg):
        """Update current joint velocities"""
        if len(msg.velocity) >= 2:
            self.current_velocities = np.array(msg.velocity[:2])

    def control_loop(self):
        """Velocity control loop"""
        # Calculate velocity error
        error = self.desired_velocities - self.current_velocities

        # Update integral term
        self.error_integral += error * 0.01

        # Calculate derivative term
        derivative = (error - self.previous_error) / 0.01

        # Calculate PID output
        command = (self.kp * error +
                   self.ki * self.error_integral +
                   self.kd * derivative)

        # Publish velocity command
        cmd_msg = Float64MultiArray()
        cmd_msg.data = (self.current_velocities + command).tolist()
        self.velocity_command_publisher.publish(cmd_msg)

        # Store current error
        self.previous_error = error

        self.get_logger().info(f'Velocity command: {cmd_msg.data}')
```

## Effort Control Interface

Effort control is used for force-controlled interactions, such as when the robot needs to apply specific forces or torques.

### Configuration

Create `~/physical_ai_ws/src/robot_description/config/effort_controller.yaml`:

```yaml
effort_controller:
  ros__parameters:
    # List of joints to control
    joints:
      - joint1

    # Command interface type
    command_interfaces:
      - effort

    # State interface type
    state_interfaces:
      - position
      - velocity
      - effort

    # Effort limits
    effort_limits:
      joint1: 100.0  # Nm

    # PID controller parameters
    gains:
      joint1:
        p: 1000.0
        i: 10.0
        d: 100.0

    # Feedforward terms
    feedforward:
      joint1:
        acceleration: 10.0
```

### Implementation Example

```python
class EffortController(Node):
    def __init__(self):
        super().__init__('effort_controller')

        # Publisher for effort commands
        self.effort_command_publisher = self.create_publisher(
            Float64MultiArray, '/effort_commands', 10
        )

        # Subscriber for joint states
        self.joint_state_subscriber = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )

        # Timer for control loop
        self.control_timer = self.create_timer(0.001, self.control_loop)  # Higher frequency for effort control

        # Controller parameters
        self.kp = np.array([1000.0])  # Proportional gains
        self.ki = np.array([10.0])    # Integral gains
        self.kd = np.array([100.0])   # Derivative gains

        # Controller state
        self.error_integral = np.zeros(1)
        self.previous_error = np.zeros(1)
        self.current_positions = np.zeros(1)
        self.current_velocities = np.zeros(1)
        self.current_efforts = np.zeros(1)
        self.desired_positions = np.array([0.5])  # Target position for position-based effort control

    def joint_state_callback(self, msg):
        """Update current joint states"""
        if len(msg.position) >= 1:
            self.current_positions = np.array(msg.position[:1])
        if len(msg.velocity) >= 1:
            self.current_velocities = np.array(msg.velocity[:1])
        if len(msg.effort) >= 1:
            self.current_efforts = np.array(msg.effort[:1])

    def control_loop(self):
        """Effort control loop"""
        # Calculate position error for position-based control
        error = self.desired_positions - self.current_positions

        # Update integral term
        self.error_integral += error * 0.001  # dt = 0.001

        # Calculate derivative term (velocity error)
        velocity_error = np.array([0.0]) - self.current_velocities  # Desired velocity = 0

        # Calculate PID output
        position_term = self.kp * error
        integral_term = self.ki * self.error_integral
        derivative_term = self.kd * velocity_error

        command = position_term + integral_term + derivative_term

        # Apply effort limits
        command = np.clip(command, -100.0, 100.0)  # Limit to ±100 Nm

        # Publish effort command
        cmd_msg = Float64MultiArray()
        cmd_msg.data = command.tolist()
        self.effort_command_publisher.publish(cmd_msg)

        # Store current error
        self.previous_error = error

        self.get_logger().info(f'Effort command: {cmd_msg.data}, Current effort: {self.current_efforts[0]:.3f}')
```

## Trajectory Control Interface

Trajectory control provides smooth, coordinated motion following position, velocity, and acceleration profiles.

### Configuration

Create `~/physical_ai_ws/src/robot_description/config/joint_trajectory_controller.yaml`:

```yaml
joint_trajectory_controller:
  ros__parameters:
    # List of joints to control
    joints:
      - joint1
      - joint2
      - joint3

    # Command interfaces
    command_interfaces:
      - position

    # State interfaces
    state_interfaces:
      - position
      - velocity

    # Allow partial joint goals
    allow_partial_joints_goal: false

    # Goal time tolerance
    goal_time_tolerance: 0.5

    # Constraints
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

    # Open loop control (for position-only interfaces)
    open_loop_control: false
```

### Implementation Example

```python
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

class TrajectoryController(Node):
    def __init__(self):
        super().__init__('trajectory_controller')

        # Publisher for trajectory commands
        self.trajectory_publisher = self.create_publisher(
            JointTrajectory, '/joint_trajectory', 10
        )

        # Timer for sending trajectory commands
        self.timer = self.create_timer(5.0, self.send_trajectory)

    def send_trajectory(self):
        """Send a trajectory command"""
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = ['joint1', 'joint2', 'joint3']

        # Create trajectory points
        points = []

        # Start point
        start_point = JointTrajectoryPoint()
        start_point.positions = [0.0, 0.0, 0.0]
        start_point.velocities = [0.0, 0.0, 0.0]
        start_point.time_from_start = Duration(sec=0, nanosec=0)
        points.append(start_point)

        # Intermediate point
        mid_point = JointTrajectoryPoint()
        mid_point.positions = [0.5, 0.3, -0.2]
        mid_point.velocities = [0.0, 0.0, 0.0]
        mid_point.time_from_start = Duration(sec=2, nanosec=0)
        points.append(mid_point)

        # Final point
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
    node = TrajectoryController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Mixed Control Interfaces

Some advanced applications require mixed control interfaces where different joints use different control types.

### Configuration Example

```yaml
mixed_controller:
  ros__parameters:
    joints:
      - joint1  # Position controlled
      - joint2  # Velocity controlled
      - joint3  # Effort controlled

    # Mixed command interfaces
    command_interfaces:
      - position   # For joint1
      - velocity   # For joint2
      - effort     # For joint3

    # State interfaces
    state_interfaces:
      - position
      - velocity
      - effort
```

### Implementation Example

```python
class MixedController(Node):
    def __init__(self):
        super().__init__('mixed_controller')

        # Publishers for different command types
        self.position_publisher = self.create_publisher(Float64MultiArray, '/position_commands', 10)
        self.velocity_publisher = self.create_publisher(Float64MultiArray, '/velocity_commands', 10)
        self.effort_publisher = self.create_publisher(Float64MultiArray, '/effort_commands', 10)

        # Subscriber for joint states
        self.joint_state_subscriber = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )

        # Timer for control loop
        self.control_timer = self.create_timer(0.01, self.control_loop)

        # Controller states
        self.current_positions = np.zeros(3)
        self.current_velocities = np.zeros(3)
        self.current_efforts = np.zeros(3)

        # Desired values for different joints
        self.desired_positions = np.array([0.5, 0.0, 0.0])  # Only joint1
        self.desired_velocities = np.array([0.0, 0.5, 0.0])  # Only joint2
        self.desired_efforts = np.array([0.0, 0.0, 10.0])    # Only joint3

    def joint_state_callback(self, msg):
        """Update current joint states"""
        if len(msg.position) >= 3:
            self.current_positions = np.array(msg.position[:3])
        if len(msg.velocity) >= 3:
            self.current_velocities = np.array(msg.velocity[:3])
        if len(msg.effort) >= 3:
            self.current_efforts = np.array(msg.effort[:3])

    def control_loop(self):
        """Mixed control loop"""
        # Calculate commands for each joint based on its control type
        position_cmd = Float64MultiArray()
        position_cmd.data = self.desired_positions.tolist()
        self.position_publisher.publish(position_cmd)

        velocity_cmd = Float64MultiArray()
        velocity_cmd.data = self.desired_velocities.tolist()
        self.velocity_publisher.publish(velocity_cmd)

        effort_cmd = Float64MultiArray()
        effort_cmd.data = self.desired_efforts.tolist()
        self.effort_publisher.publish(effort_cmd)

        self.get_logger().info(
            f'Mixed commands - Position: {position_cmd.data}, '
            f'Velocity: {velocity_cmd.data}, Effort: {effort_cmd.data}'
        )
```

## Hardware Interface Implementation

### Position Hardware Interface

For real hardware, you need to implement the hardware interface:

```python
from ros2_control_python import HardwareInterface, ReturnCode
import numpy as np

class PositionHardwareInterface(HardwareInterface):
    def __init__(self, hardware_info):
        super().__init__()
        self.joint_names = []
        self.position_commands = []
        self.position_states = []
        self.velocity_states = []
        self.effort_states = []

        # Parse hardware info
        for joint in hardware_info.joints:
            self.joint_names.append(joint.name)
            self.position_commands.append(0.0)
            self.position_states.append(0.0)
            self.velocity_states.append(0.0)
            self.effort_states.append(0.0)

    def configure(self, interfaces):
        """Configure the hardware interface"""
        # Check that we have the right interfaces
        for interface in interfaces:
            if interface.name == 'position' and interface.direction == 'output':
                continue
            elif interface.name in ['position', 'velocity', 'effort'] and interface.direction == 'input':
                continue
            else:
                return ReturnCode.ERROR

        return ReturnCode.OK

    def start(self):
        """Start the hardware interface"""
        # Initialize hardware communication
        return ReturnCode.OK

    def stop(self):
        """Stop the hardware interface"""
        # Clean up hardware communication
        return ReturnCode.OK

    def read(self, time, period):
        """Read current states from hardware"""
        try:
            # Read positions, velocities, and efforts from hardware
            for i, joint_name in enumerate(self.joint_names):
                # This would interface with your specific hardware
                # Example: self.position_states[i] = hardware.read_position(joint_name)
                pass

            return ReturnCode.OK
        except Exception as e:
            self.get_logger().error(f'Error reading hardware: {str(e)}')
            return ReturnCode.ERROR

    def write(self, time, period):
        """Write commands to hardware"""
        try:
            # Send position commands to hardware
            for i, (joint_name, command) in enumerate(zip(self.joint_names, self.position_commands)):
                # This would send commands to your specific hardware
                # Example: hardware.send_position_command(joint_name, command)
                pass

            return ReturnCode.OK
        except Exception as e:
            self.get_logger().error(f'Error writing to hardware: {str(e)}')
            return ReturnCode.ERROR
```

## Controller Selection Guidelines

### When to Use Each Interface

#### Position Control
- **Use when**: Precise positioning is required
- **Examples**: Pick-and-place operations, precise positioning tasks
- **Advantages**: High accuracy, good for static poses
- **Disadvantages**: May cause high forces if constrained

#### Velocity Control
- **Use when**: Speed control is more important than position
- **Examples**: Mobile robot navigation, conveyor belt tracking
- **Advantages**: Smooth motion, good for dynamic tasks
- **Disadvantages**: Less precise positioning

#### Effort Control
- **Use when**: Force control is required
- **Examples**: Assembly tasks, compliant motion, contact tasks
- **Advantages**: Force limiting, compliant behavior
- **Disadvantages**: Requires accurate force sensing, more complex

#### Trajectory Control
- **Use when**: Smooth, coordinated motion is needed
- **Examples**: Complex manipulation tasks, coordinated multi-joint motion
- **Advantages**: Smooth motion profiles, coordinated control
- **Disadvantages**: More complex setup, requires planning

## Safety Considerations

### Position Limits

Always implement position limits:

```python
def apply_position_limits(self, positions, joint_limits):
    """Apply position limits to commands"""
    limited_positions = []
    for pos, (min_limit, max_limit) in zip(positions, joint_limits):
        limited_pos = max(min_limit, min(max_limit, pos))
        limited_positions.append(limited_pos)
    return limited_positions
```

### Velocity and Acceleration Limits

```python
def apply_motion_limits(self, current_state, desired_state, max_velocity, max_acceleration, dt):
    """Apply velocity and acceleration limits"""
    # Calculate desired velocity
    desired_velocity = (desired_state - current_state) / dt

    # Limit velocity
    limited_velocity = np.clip(desired_velocity, -max_velocity, max_velocity)

    # Calculate acceleration
    acceleration = (limited_velocity - self.previous_velocity) / dt

    # Limit acceleration
    limited_acceleration = np.clip(acceleration, -max_acceleration, max_acceleration)

    # Recalculate velocity from limited acceleration
    new_velocity = self.previous_velocity + limited_acceleration * dt
    new_velocity = np.clip(new_velocity, -max_velocity, max_velocity)

    # Calculate new position
    new_position = current_state + new_velocity * dt

    self.previous_velocity = new_velocity

    return new_position
```

### Emergency Stop

```python
def emergency_stop(self):
    """Emergency stop function"""
    # Set all commands to zero or safe positions
    zero_cmd = Float64MultiArray()
    zero_cmd.data = [0.0] * len(self.joint_names)

    self.position_publisher.publish(zero_cmd)
    self.velocity_publisher.publish(zero_cmd)
    self.effort_publisher.publish(zero_cmd)

    self.get_logger().warn('Emergency stop activated!')
```

## Performance Considerations

### Control Loop Timing

Different control interfaces require different loop frequencies:

```python
# Position control: 50-200 Hz typically sufficient
position_loop_freq = 100  # Hz

# Velocity control: 100-500 Hz for smooth motion
velocity_loop_freq = 200  # Hz

# Effort control: 1000+ Hz for stable force control
effort_loop_freq = 1000   # Hz
```

### Real-time Considerations

For real-time applications:

```python
import threading
import time

class RealTimeController:
    def __init__(self, control_frequency):
        self.control_period = 1.0 / control_frequency
        self.control_thread = threading.Thread(target=self.real_time_control_loop)
        self.running = True

    def real_time_control_loop(self):
        """Real-time control loop with precise timing"""
        next_loop_time = time.time()

        while self.running:
            # Do control computation
            self.control_computation()

            # Sleep until next loop time
            next_loop_time += self.control_period
            sleep_time = next_loop_time - time.time()

            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # Missed deadline
                print("Control loop deadline missed!")
```

## Integration with ros2_control

### Controller Manager Configuration

```yaml
controller_manager:
  ros__parameters:
    update_rate: 100  # Hz

    # Available controllers
    position_controller:
      type: position_controllers/JointGroupPositionController

    velocity_controller:
      type: velocity_controllers/JointGroupVelocityController

    effort_controller:
      type: effort_controllers/JointGroupEffortController

    joint_trajectory_controller:
      type: joint_trajectory_controller/JointTrajectoryController
```

### Controller Spawning

```bash
# Spawn position controller
ros2 run controller_manager spawner position_controller

# Spawn velocity controller
ros2 run controller_manager spawner velocity_controller

# Switch between controllers
ros2 control switch_controllers --start position_controller --stop velocity_controller
```

## Troubleshooting Common Issues

### Interface Mismatch

Ensure command and state interfaces match:

```python
# Check available interfaces
ros2 control list_hardware_interfaces

# Verify controller configuration matches hardware
ros2 param get /controller_manager position_controller
```

### Command Saturation

Monitor for saturated commands:

```python
def monitor_command_saturation(self, commands, limits):
    """Monitor if commands are hitting limits"""
    saturation_count = 0
    for cmd, limit in zip(commands, limits):
        if abs(cmd) >= limit * 0.95:  # 95% of limit
            saturation_count += 1

    if saturation_count > 0:
        self.get_logger().warn(f'{saturation_count} commands near saturation')
```

### Feedback Issues

Ensure proper feedback loop:

```python
def validate_feedback(self, command, feedback, tolerance=0.1):
    """Validate that feedback is properly following command"""
    error = abs(command - feedback)
    if error > tolerance:
        self.get_logger().warn(f'Large error between command ({command}) and feedback ({feedback}): {error}')
        return False
    return True
```

## Advanced Control Interfaces

### Impedance Control

For compliant manipulation:

```python
class ImpedanceController:
    def __init__(self):
        # Stiffness and damping parameters
        self.stiffness = np.array([1000.0, 1000.0, 1000.0])  # N/m
        self.damping = np.array([200.0, 200.0, 200.0])       # Ns/m

    def compute_impedance_force(self, position_error, velocity_error):
        """Compute impedance control force"""
        return self.stiffness * position_error + self.damping * velocity_error
```

### Admittance Control

For force-guided motion:

```python
class AdmittanceController:
    def __init__(self):
        # Admittance parameters (inverse of impedance)
        self.admittance = np.array([0.001, 0.001, 0.001])  # m/N

    def compute_motion_from_force(self, applied_force):
        """Compute motion command from applied force"""
        return self.admittance * applied_force
```

## Best Practices

### 1. Interface Selection

Choose the right interface for your application:
- Use position control for precise positioning
- Use velocity control for smooth motion
- Use effort control for force-sensitive tasks
- Use trajectory control for coordinated motion

### 2. Safety First

Always implement safety features:
- Position, velocity, and effort limits
- Emergency stop functionality
- Collision detection and avoidance
- Graceful degradation

### 3. Proper Tuning

Tune controller parameters appropriately:
- Start with conservative gains
- Test incrementally
- Monitor for oscillations or instability
- Consider the specific dynamics of your robot

### 4. Monitoring and Diagnostics

Implement comprehensive monitoring:

```python
def publish_diagnostics(self):
    """Publish controller diagnostics"""
    # Check tracking performance
    # Monitor command saturation
    # Validate feedback quality
    # Report controller status
    pass
```

## Next Steps

After mastering control interfaces:

1. Continue to Week 4: [Simulation Fundamentals](../week-04/index.md) to learn about robot simulation
2. Practice implementing different control strategies for your specific robot
3. Learn about advanced control techniques like model predictive control
4. Explore sensor integration for feedback control

Your understanding of control interfaces is now fundamental for implementing effective robot control systems in the Physical AI and Humanoid Robotics course!
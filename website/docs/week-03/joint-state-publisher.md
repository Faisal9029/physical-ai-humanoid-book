---
sidebar_position: 3
---

# Joint State Publisher

This guide covers the Joint State Publisher, a critical ROS 2 component that publishes joint state information for your robot. The Joint State Publisher is essential for robot visualization, simulation, and control systems.

## Overview

The Joint State Publisher is a ROS 2 node that publishes joint state information in the form of `sensor_msgs/JointState` messages. This information is crucial for:

- **Robot Visualization**: RViz2 uses joint states to display robot models in the correct configuration
- **Simulation**: Simulation environments use joint states to update robot models
- **Control Systems**: Controllers often need current joint positions for feedback
- **TF Transforms**: Robot State Publisher uses joint states to compute and broadcast transforms

### JointState Message Structure

The `sensor_msgs/JointState` message contains:

```python
std_msgs/Header header
string[] name          # Joint names
float64[] position    # Joint positions (radians or meters)
float64[] velocity    # Joint velocities (rad/s or m/s)
float64[] effort      # Joint efforts (torque or force)
```

## Joint State Publisher Types

### Joint State Publisher (GUI)

The `joint_state_publisher` node can operate in two modes:

1. **GUI Mode**: Provides a graphical interface to manually set joint positions
2. **Non-GUI Mode**: Publishes default joint positions or reads from parameters

#### Launch with GUI

```bash
ros2 run joint_state_publisher joint_state_publisher --ros-args -p use_gui:=true
```

#### Launch without GUI

```bash
ros2 run joint_state_publisher joint_state_publisher --ros-args -p use_gui:=false
```

### Joint State Publisher GUI

The GUI version provides sliders for each joint, allowing manual control of joint positions:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import tkinter as tk
from tkinter import Scale, HORIZONTAL

class JointStatePublisherGUI(Node):
    def __init__(self):
        super().__init__('joint_state_publisher_gui')

        # Publisher for joint states
        self.joint_state_publisher = self.create_publisher(JointState, 'joint_states', 10)

        # Define joint names and initial positions
        self.joint_names = ['joint1', 'joint2', 'joint3']
        self.joint_positions = [0.0, 0.0, 0.0]
        self.joint_velocities = [0.0, 0.0, 0.0]
        self.joint_efforts = [0.0, 0.0, 0.0]

        # Create GUI
        self.create_gui()

        # Timer to publish joint states
        self.timer = self.create_timer(0.05, self.publish_joint_states)  # 20 Hz

    def create_gui(self):
        """Create the GUI with sliders for each joint"""
        self.root = tk.Tk()
        self.root.title('Joint State Publisher')

        # Create sliders for each joint
        self.sliders = []
        for i, joint_name in enumerate(self.joint_names):
            slider = Scale(
                self.root,
                from_=-3.14,  # -π
                to=3.14,      # π
                resolution=0.01,
                orient=HORIZONTAL,
                label=joint_name,
                command=lambda value, idx=i: self.update_joint_position(value, idx)
            )
            slider.set(0.0)  # Initial position
            slider.pack(fill='x', padx=10, pady=5)
            self.sliders.append(slider)

    def update_joint_position(self, value, joint_index):
        """Update joint position when slider changes"""
        self.joint_positions[joint_index] = float(value)

    def publish_joint_states(self):
        """Publish joint state message"""
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        msg.name = self.joint_names
        msg.position = self.joint_positions.copy()
        msg.velocity = self.joint_velocities
        msg.effort = self.joint_efforts

        self.joint_state_publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)

    # Create and start GUI in a separate thread
    import threading
    node = JointStatePublisherGUI()

    # Start GUI in separate thread
    gui_thread = threading.Thread(target=node.root.mainloop)
    gui_thread.daemon = True
    gui_thread.start()

    # Spin the ROS node
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.shutdown()
        node.root.quit()

if __name__ == '__main__':
    main()
```

## Configuration Parameters

### Common Parameters

The Joint State Publisher accepts several parameters:

```yaml
joint_state_publisher:
  ros__parameters:
    # Whether to use the GUI
    use_gui: false

    # Source of joint states (if not using GUI)
    source_list: []  # List of topics to subscribe to for joint states

    # Initial joint positions
    initial_positions:
      joint1: 0.0
      joint2: 0.0
      joint3: 0.0

    # Rate at which to publish joint states
    rate: 50  # Hz

    # Joint limits for safety
    ignore_timestamp: false
```

### Launch File Configuration

Create `~/physical_ai_ws/src/robot_description/launch/joint_state_publisher.launch.py`:

```python
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Declare launch arguments
    use_gui = DeclareLaunchArgument(
        'use_gui',
        default_value='false',
        description='Whether to use joint_state_publisher_gui'
    )

    # Robot description parameter
    robot_description_path = os.path.join(
        get_package_share_directory('robot_description'),
        'urdf',
        'simple_robot.urdf'
    )

    # Joint State Publisher
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[
            {
                'use_gui': LaunchConfiguration('use_gui'),
                'rate': 50,
                'source_list': []  # No additional sources by default
            }
        ],
        remappings=[
            # Remap to avoid conflicts with other joint state publishers
            ('/joint_states', '/robot/joint_states')
        ]
    )

    # Joint State Publisher GUI (only if use_gui is true)
    joint_state_publisher_gui = Node(
        condition=launch.conditions.IfCondition(LaunchConfiguration('use_gui')),
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        parameters=[
            {
                'rate': 20,
            }
        ]
    )

    return LaunchDescription([
        use_gui,
        joint_state_publisher,
        joint_state_publisher_gui
    ])
```

## Robot State Publisher Integration

### Relationship Between Publishers

The Joint State Publisher and Robot State Publisher work together:

1. **Joint State Publisher**: Publishes joint positions, velocities, and efforts
2. **Robot State Publisher**: Uses joint states to compute and broadcast TF transforms

```python
# Both nodes are typically needed together
joint_state_publisher = Node(
    package='joint_state_publisher',
    executable='joint_state_publisher',
    parameters=[{'use_gui': False}]
)

robot_state_publisher = Node(
    package='robot_state_publisher',
    executable='robot_state_publisher',
    parameters=[{'robot_description': robot_description_param}]
)
```

## Custom Joint State Publisher

### Advanced Joint State Publisher

For more sophisticated applications, you might need a custom joint state publisher:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import numpy as np
import math

class AdvancedJointStatePublisher(Node):
    def __init__(self):
        super().__init__('advanced_joint_state_publisher')

        # Publisher for joint states
        self.joint_state_publisher = self.create_publisher(JointState, 'joint_states', 10)

        # Subscriber for trajectory commands (for simulation)
        self.trajectory_subscriber = self.create_subscription(
            JointTrajectory,
            'joint_trajectory_commands',
            self.trajectory_callback,
            10
        )

        # Timer for publishing joint states
        self.timer = self.create_timer(0.01, self.publish_joint_states)  # 100 Hz

        # Joint state variables
        self.joint_names = ['joint1', 'joint2', 'joint3']
        self.current_positions = [0.0, 0.0, 0.0]
        self.current_velocities = [0.0, 0.0, 0.0]
        self.current_efforts = [0.0, 0.0, 0.0]

        # Trajectory execution variables
        self.trajectory_points = []
        self.current_trajectory_idx = 0
        self.trajectory_active = False

        # Simulation parameters
        self.simulation_time = 0.0
        self.time_step = 0.01

        self.get_logger().info('Advanced Joint State Publisher initialized')

    def trajectory_callback(self, msg):
        """Handle incoming trajectory commands"""
        if msg.joint_names != self.joint_names:
            self.get_logger().warn('Joint names do not match')
            return

        # Store trajectory points
        self.trajectory_points = msg.points
        self.current_trajectory_idx = 0
        self.trajectory_active = True

        self.get_logger().info(f'Received trajectory with {len(msg.points)} points')

    def interpolate_trajectory(self):
        """Interpolate between trajectory points"""
        if not self.trajectory_active or not self.trajectory_points:
            return False

        if self.current_trajectory_idx >= len(self.trajectory_points):
            self.trajectory_active = False
            return False

        current_point = self.trajectory_points[self.current_trajectory_idx]

        # Check if we should move to the next point
        if self.simulation_time >= current_point.time_from_start.sec + current_point.time_from_start.nanosec * 1e-9:
            if self.current_trajectory_idx < len(self.trajectory_points) - 1:
                self.current_trajectory_idx += 1
                current_point = self.trajectory_points[self.current_trajectory_idx]
            else:
                self.trajectory_active = False
                return False

        # For simplicity, just use the current point
        # In a real implementation, you'd interpolate between points
        if len(current_point.positions) == len(self.current_positions):
            self.current_positions = list(current_point.positions)

        if len(current_point.velocities) == len(self.current_velocities):
            self.current_velocities = list(current_point.velocities)

        return True

    def simulate_joint_dynamics(self):
        """Simulate simple joint dynamics"""
        # Add some simple dynamics simulation
        for i in range(len(self.current_positions)):
            # Apply damping
            self.current_velocities[i] *= 0.99  # Simple damping

            # Update position based on velocity
            self.current_positions[i] += self.current_velocities[i] * self.time_step

    def publish_joint_states(self):
        """Publish joint state message"""
        # Update joint states based on current mode
        if self.trajectory_active:
            self.interpolate_trajectory()

        # Simulate dynamics if not following a trajectory
        if not self.trajectory_active:
            self.simulate_joint_dynamics()

        # Create and publish joint state message
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        msg.name = self.joint_names
        msg.position = self.current_positions.copy()
        msg.velocity = self.current_velocities.copy()
        msg.effort = self.current_efforts.copy()

        self.joint_state_publisher.publish(msg)

        self.simulation_time += self.time_step

def main(args=None):
    rclpy.init(args=args)
    node = AdvancedJointStatePublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Advanced Joint State Publisher')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Integration with Hardware

### Real Robot Hardware Interface

For real robots, the Joint State Publisher is often replaced by hardware interfaces:

```python
class HardwareJointStatePublisher(Node):
    def __init__(self):
        super().__init__('hardware_joint_state_publisher')

        # Publisher for joint states
        self.joint_state_publisher = self.create_publisher(JointState, 'joint_states', 10)

        # Interface to hardware (this would be specific to your hardware)
        self.hardware_interface = self.initialize_hardware()

        # Timer for reading from hardware
        self.timer = self.create_timer(0.001, self.read_hardware_states)  # 1 kHz

    def initialize_hardware(self):
        """Initialize connection to robot hardware"""
        # This would connect to your specific hardware
        # Could be serial, Ethernet, CAN, etc.
        pass

    def read_hardware_states(self):
        """Read joint states from hardware"""
        # Read positions, velocities, efforts from hardware
        positions = self.hardware_interface.read_positions()
        velocities = self.hardware_interface.read_velocities()
        efforts = self.hardware_interface.read_efforts()

        # Publish joint state message
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        msg.name = self.joint_names
        msg.position = positions
        msg.velocity = velocities
        msg.effort = efforts

        self.joint_state_publisher.publish(msg)
```

## Simulation Integration

### Gazebo Integration

In Gazebo simulation, joint states are typically published by the Gazebo ROS 2 control plugin:

```xml
<!-- In your robot URDF -->
<gazebo>
  <plugin filename="libgazebo_ros2_control.so" name="gazebo_ros2_control">
    <parameters>$(find robot_description)/config/controllers.yaml</parameters>
  </plugin>
</gazebo>
```

### Isaac Sim Integration

For Isaac Sim, joint states are published through the Isaac ROS bridge:

```python
# Isaac Sim typically handles joint state publishing automatically
# through the appropriate ROS bridge components
```

## Joint Limits and Safety

### Joint State Validation

For safety, validate joint states before publishing:

```python
class SafeJointStatePublisher(Node):
    def __init__(self):
        super().__init__('safe_joint_state_publisher')

        self.joint_state_publisher = self.create_publisher(JointState, 'joint_states', 10)

        # Define joint limits
        self.joint_limits = {
            'joint1': {'min': -3.14, 'max': 3.14},
            'joint2': {'min': -2.0, 'max': 2.0},
            'joint3': {'min': -1.57, 'max': 1.57}
        }

        self.timer = self.create_timer(0.01, self.publish_joint_states)

        self.joint_names = ['joint1', 'joint2', 'joint3']
        self.current_positions = [0.0, 0.0, 0.0]

    def validate_joint_positions(self, positions):
        """Validate joint positions against limits"""
        valid_positions = []
        for i, (pos, joint_name) in enumerate(zip(positions, self.joint_names)):
            limits = self.joint_limits.get(joint_name, {'min': -float('inf'), 'max': float('inf')})

            if pos < limits['min']:
                self.get_logger().warn(f'Joint {joint_name} position {pos} below limit {limits["min"]}')
                pos = limits['min']
            elif pos > limits['max']:
                self.get_logger().warn(f'Joint {joint_name} position {pos} above limit {limits["max"]}')
                pos = limits['max']

            valid_positions.append(pos)

        return valid_positions

    def publish_joint_states(self):
        """Publish validated joint states"""
        # Validate positions
        validated_positions = self.validate_joint_positions(self.current_positions)

        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        msg.name = self.joint_names
        msg.position = validated_positions
        msg.velocity = [0.0, 0.0, 0.0]  # Example velocities
        msg.effort = [0.0, 0.0, 0.0]    # Example efforts

        self.joint_state_publisher.publish(msg)
```

## Performance Considerations

### Publishing Rate Optimization

The publishing rate affects both performance and smoothness:

```python
# For visualization (lower rate is usually sufficient)
visualization_rate = 30  # Hz

# For control (higher rate for better performance)
control_rate = 100  # Hz

# For high-performance control
high_performance_rate = 1000  # Hz
```

### Memory and CPU Usage

Optimize for resource usage:

```python
class OptimizedJointStatePublisher(Node):
    def __init__(self):
        super().__init__('optimized_joint_state_publisher')

        self.joint_state_publisher = self.create_publisher(JointState, 'joint_states', 1)

        # Pre-allocate message objects to reduce memory allocation
        self.joint_state_msg = JointState()
        self.header = Header()

        # Use numpy arrays for efficient calculations
        self.positions = np.array([0.0, 0.0, 0.0])
        self.velocities = np.array([0.0, 0.0, 0.0])
        self.efforts = np.array([0.0, 0.0, 0.0])

        self.timer = self.create_timer(0.01, self.publish_joint_states)

    def publish_joint_states(self):
        """Optimized joint state publishing"""
        # Update header efficiently
        self.header.stamp = self.get_clock().now().to_msg()
        self.joint_state_msg.header = self.header

        # Set data efficiently
        self.joint_state_msg.position = self.positions.tolist()
        self.joint_state_msg.velocity = self.velocities.tolist()
        self.joint_state_msg.effort = self.efforts.tolist()

        # Publish without copying
        self.joint_state_publisher.publish(self.joint_state_msg)
```

## Troubleshooting Common Issues

### Missing Joint States

If joint states are missing:

1. **Check node status**:
   ```bash
   ros2 node list | grep joint
   ros2 node info /joint_state_publisher
   ```

2. **Check topic publication**:
   ```bash
   ros2 topic list | grep joint
   ros2 topic echo /joint_states
   ```

3. **Verify robot description**:
   ```bash
   ros2 param get /robot_state_publisher robot_description
   ```

### Joint Name Mismatches

Ensure joint names match between:
- URDF file
- Controller configurations
- Joint state publisher parameters
- Any other nodes that reference joints

### Performance Issues

If experiencing performance issues:

1. **Check publishing rate**: Ensure it's appropriate for your application
2. **Monitor CPU usage**: Use `htop` or similar tools
3. **Reduce message frequency**: If visualization only, lower the rate
4. **Optimize hardware interface**: For real robots, optimize the hardware communication

## Launch Integration

### Complete Launch Example

Create `~/physical_ai_ws/src/robot_description/launch/robot_state.launch.py`:

```python
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true'
    )

    use_gui = DeclareLaunchArgument(
        'use_gui',
        default_value='false',
        description='Use joint_state_publisher_gui instead of joint_state_publisher if true'
    )

    robot_description_path = os.path.join(
        get_package_share_directory('robot_description'),
        'urdf',
        'simple_robot.urdf'
    )

    # Read robot description from file
    with open(robot_description_path, 'r') as infp:
        robot_description = infp.read()

    # Joint State Publisher
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[
            {
                'use_gui': LaunchConfiguration('use_gui'),
                'rate': 50
            }
        ],
        condition=IfCondition(PythonExpression(['not ', LaunchConfiguration('use_gui')]))
    )

    # Joint State Publisher GUI
    joint_state_publisher_gui = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        condition=IfCondition(LaunchConfiguration('use_gui'))
    )

    # Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[
            {
                'robot_description': robot_description,
                'publish_frequency': 50.0,
                'use_sim_time': LaunchConfiguration('use_sim_time')
            }
        ]
    )

    # Group all nodes to run in the same namespace
    joint_state_publisher_group = GroupAction(
        actions=[
            PushRosNamespace('robot'),
            joint_state_publisher,
            joint_state_publisher_gui,
            robot_state_publisher
        ]
    )

    return LaunchDescription([
        use_sim_time,
        use_gui,
        joint_state_publisher_group
    ])
```

## Best Practices

### 1. Parameter Management

Use launch files and parameter files for configuration:

```yaml
# config/joint_state_publisher_params.yaml
joint_state_publisher:
  ros__parameters:
    use_gui: false
    rate: 50
    source_list: []
```

### 2. Namespace Management

Use namespaces to avoid topic conflicts:

```python
# In your launch file
from launch_ros.actions import PushRosNamespace

# Group nodes under a namespace
namespace_group = GroupAction(
    actions=[
        PushRosNamespace('my_robot'),
        joint_state_publisher_node,
        robot_state_publisher_node
    ]
)
```

### 3. Error Handling

Implement proper error handling:

```python
def publish_joint_states(self):
    try:
        # Validate joint states
        if not self.validate_states():
            self.get_logger().error('Invalid joint states, not publishing')
            return

        # Publish message
        self.joint_state_publisher.publish(self.joint_state_msg)

    except Exception as e:
        self.get_logger().error(f'Error publishing joint states: {str(e)}')
```

### 4. Monitoring and Diagnostics

Add diagnostic capabilities:

```python
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus

class MonitoredJointStatePublisher(Node):
    def __init__(self):
        super().__init__('monitored_joint_state_publisher')

        # Diagnostic publisher
        self.diag_publisher = self.create_publisher(DiagnosticArray, '/diagnostics', 10)

        # Monitor publishing rate
        self.publish_count = 0
        self.last_publish_time = self.get_clock().now()

        self.diag_timer = self.create_timer(1.0, self.publish_diagnostics)

    def publish_diagnostics(self):
        """Publish diagnostic information"""
        diag_array = DiagnosticArray()
        diag_array.header.stamp = self.get_clock().now().to_msg()

        status = DiagnosticStatus()
        status.name = 'joint_state_publisher'
        status.level = DiagnosticStatus.OK
        status.message = 'OK'

        # Add performance metrics
        current_time = self.get_clock().now()
        dt = (current_time - self.last_publish_time).nanoseconds / 1e9
        publish_rate = self.publish_count / dt if dt > 0 else 0

        status.values.append(KeyValue(key='publish_rate', value=str(publish_rate)))
        status.values.append(KeyValue(key='expected_rate', value='50.0'))

        diag_array.status.append(status)
        self.diag_publisher.publish(diag_array)

        # Reset for next cycle
        self.publish_count = 0
        self.last_publish_time = current_time
```

## Next Steps

After mastering Joint State Publisher:

1. Continue to [Robot State Publisher](./robot-state-publisher.md) to understand how to broadcast TF transforms
2. Practice integrating joint state publishing with robot simulation
3. Learn about controller feedback and how it uses joint states
4. Explore advanced topics like sensor fusion for state estimation

Your understanding of Joint State Publisher is now fundamental for robot state management in the Physical AI and Humanoid Robotics course!
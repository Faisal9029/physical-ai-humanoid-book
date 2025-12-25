---
sidebar_position: 4
---

# Forward and Inverse Kinematics

This guide covers the fundamental concepts of robot kinematics, specifically forward and inverse kinematics, which are essential for controlling robot motion and positioning. Kinematics is the study of motion without considering the forces that cause it.

## Overview

Kinematics is a branch of mechanics that describes the motion of points, bodies (objects), and systems of bodies (groups of objects) without considering the forces that cause them to move. In robotics, kinematics deals with the relationship between joint angles and the position and orientation of the robot's end-effector.

### Key Concepts

- **Forward Kinematics (FK)**: Calculate end-effector pose from joint angles
- **Inverse Kinematics (IK)**: Calculate joint angles from desired end-effector pose
- **Degrees of Freedom (DOF)**: Number of independent movements a robot can make
- **Configuration Space**: All possible joint configurations
- **Workspace**: All possible end-effector positions

## Forward Kinematics

### Definition

Forward kinematics is the process of calculating the position and orientation of the robot's end-effector given the joint angles. It's called "forward" because we're moving forward from joint space to Cartesian space.

### Mathematical Foundation

Forward kinematics uses transformation matrices to represent the position and orientation of each link relative to the previous one.

#### Denavit-Hartenberg (DH) Parameters

The DH convention is a systematic method for defining coordinate frames on a robotic manipulator:

- **a (link length)**: Distance along x-axis from z_i to z_{i+1}
- **α (link twist)**: Angle about x-axis from z_i to z_{i+1}
- **d (link offset)**: Distance along z_i from x_i to x_{i+1}
- **θ (joint angle)**: Angle about z_i from x_i to x_{i+1}

### Transformation Matrix

The transformation matrix from frame i to frame i+1 is:

```
T_i^(i+1) = [ cos(θ)   -sin(θ)cos(α)   sin(θ)sin(α)   a*cos(θ) ]
            [ sin(θ)    cos(θ)cos(α)  -cos(θ)sin(α)   a*sin(θ) ]
            [ 0         sin(α)        cos(α)          d        ]
            [ 0         0             0               1        ]
```

### 2-Link Planar Robot Example

Let's implement forward kinematics for a simple 2-link planar robot:

```python
import numpy as np
import math

def forward_kinematics_2link(theta1, theta2, l1=1.0, l2=1.0):
    """
    Calculate forward kinematics for a 2-link planar robot.

    Args:
        theta1: Joint angle 1 (radians)
        theta2: Joint angle 2 (radians)
        l1: Length of link 1
        l2: Length of link 2

    Returns:
        (x, y): End-effector position
    """
    # Calculate end-effector position
    x = l1 * math.cos(theta1) + l2 * math.cos(theta1 + theta2)
    y = l1 * math.sin(theta1) + l2 * math.sin(theta1 + theta2)

    return x, y

def calculate_jacobian_2link(theta1, theta2, l1=1.0, l2=1.0):
    """
    Calculate the Jacobian matrix for a 2-link planar robot.

    Args:
        theta1: Joint angle 1 (radians)
        theta2: Joint angle 2 (radians)
        l1: Length of link 1
        l2: Length of link 2

    Returns:
        Jacobian matrix (2x2)
    """
    # Partial derivatives
    dx_dtheta1 = -l1 * math.sin(theta1) - l2 * math.sin(theta1 + theta2)
    dx_dtheta2 = -l2 * math.sin(theta1 + theta2)
    dy_dtheta1 = l1 * math.cos(theta1) + l2 * math.cos(theta1 + theta2)
    dy_dtheta2 = l2 * math.cos(theta1 + theta2)

    # Jacobian matrix
    J = np.array([
        [dx_dtheta1, dx_dtheta2],
        [dy_dtheta1, dy_dtheta2]
    ])

    return J

# Example usage
theta1 = math.pi / 4  # 45 degrees
theta2 = math.pi / 6  # 30 degrees
l1 = 1.0
l2 = 0.8

x, y = forward_kinematics_2link(theta1, theta2, l1, l2)
print(f"End-effector position: ({x:.3f}, {y:.3f})")

jacobian = calculate_jacobian_2link(theta1, theta2, l1, l2)
print(f"Jacobian matrix:\n{jacobian}")
```

### 3D Robot Arm Example

For more complex robots, we use homogeneous transformation matrices:

```python
import numpy as np

def dh_transform(a, alpha, d, theta):
    """
    Create a Denavit-Hartenberg transformation matrix.

    Args:
        a: link length
        alpha: link twist
        d: link offset
        theta: joint angle

    Returns:
        4x4 transformation matrix
    """
    T = np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0, np.sin(alpha), np.cos(alpha), d],
        [0, 0, 0, 1]
    ])
    return T

def forward_kinematics_3d(joint_angles, dh_params):
    """
    Calculate forward kinematics for a 3D robot arm using DH parameters.

    Args:
        joint_angles: List of joint angles [theta1, theta2, ..., thetaN]
        dh_params: List of tuples (a, alpha, d, theta) for each joint

    Returns:
        4x4 transformation matrix representing end-effector pose
    """
    T_total = np.eye(4)  # Start with identity matrix

    for i, (a, alpha, d, theta_offset) in enumerate(dh_params):
        theta = joint_angles[i] + theta_offset
        T_i = dh_transform(a, alpha, d, theta)
        T_total = np.dot(T_total, T_i)

    return T_total

# Example: 3-DOF planar robot arm
dh_params = [
    (0.5, 0, 0, 0),    # Joint 1: a=0.5, alpha=0, d=0, theta_offset=0
    (0.4, 0, 0, 0),    # Joint 2: a=0.4, alpha=0, d=0, theta_offset=0
    (0.3, 0, 0, 0)     # Joint 3: a=0.3, alpha=0, d=0, theta_offset=0
]

joint_angles = [np.pi/4, np.pi/6, -np.pi/3]  # 3 joint angles
T_end_effector = forward_kinematics_3d(joint_angles, dh_params)

print("End-effector transformation matrix:")
print(T_end_effector)
print(f"End-effector position: ({T_end_effector[0,3]:.3f}, {T_end_effector[1,3]:.3f}, {T_end_effector[2,3]:.3f})")
```

## Inverse Kinematics

### Definition

Inverse kinematics is the process of calculating the joint angles required to achieve a desired end-effector position and orientation. It's called "inverse" because we're working backwards from Cartesian space to joint space.

### Analytical vs Numerical Solutions

#### Analytical Solutions

Analytical solutions provide exact mathematical formulas but are only possible for simple robots with specific geometries.

For a 2-link planar robot:

```python
def inverse_kinematics_2link(x, y, l1=1.0, l2=1.0):
    """
    Calculate inverse kinematics for a 2-link planar robot.

    Args:
        x: Desired x position
        y: Desired y position
        l1: Length of link 1
        l2: Length of link 2

    Returns:
        (theta1, theta2): Joint angles (or None if no solution exists)
    """
    # Calculate distance from origin to target
    r = math.sqrt(x**2 + y**2)

    # Check if target is reachable
    if r > l1 + l2:
        print("Target is out of reach")
        return None

    if r < abs(l1 - l2):
        print("Target is inside workspace")
        return None

    # Calculate theta2 using law of cosines
    cos_theta2 = (l1**2 + l2**2 - r**2) / (2 * l1 * l2)
    sin_theta2 = math.sqrt(1 - cos_theta2**2)
    theta2 = math.atan2(sin_theta2, cos_theta2)

    # Calculate theta1
    k1 = l1 + l2 * cos_theta2
    k2 = l2 * sin_theta2
    theta1 = math.atan2(y, x) - math.atan2(k2, k1)

    return theta1, theta2

# Example usage
target_x = 1.2
target_y = 0.8
result = inverse_kinematics_2link(target_x, target_y, 1.0, 0.8)

if result:
    theta1, theta2 = result
    print(f"Joint angles: theta1={theta1:.3f} rad, theta2={theta2:.3f} rad")

    # Verify with forward kinematics
    x_verify, y_verify = forward_kinematics_2link(theta1, theta2, 1.0, 0.8)
    print(f"Verification - Target: ({target_x}, {target_y}), Actual: ({x_verify:.3f}, {y_verify:.3f})")
```

#### Numerical Solutions

For complex robots, numerical methods are often used:

```python
from scipy.optimize import fsolve

def ik_error_function(joint_angles, target_pose, dh_params):
    """
    Error function for numerical inverse kinematics.

    Args:
        joint_angles: Current joint angles
        target_pose: Desired end-effector pose [x, y, z, rx, ry, rz]
        dh_params: DH parameters for the robot

    Returns:
        Error vector
    """
    # Calculate current end-effector pose
    T_current = forward_kinematics_3d(joint_angles, dh_params)

    # Extract position
    current_pos = T_current[:3, 3]
    target_pos = target_pose[:3]

    # Calculate position error
    error = target_pos - current_pos

    return error

def numerical_inverse_kinematics(target_pose, dh_params, initial_guess):
    """
    Solve inverse kinematics using numerical methods.

    Args:
        target_pose: Desired end-effector pose
        dh_params: DH parameters for the robot
        initial_guess: Initial joint angle guess

    Returns:
        Joint angles that achieve the target pose
    """
    solution = fsolve(ik_error_function, initial_guess, args=(target_pose, dh_params))
    return solution
```

## ROS 2 Integration

### Creating a Kinematics Service

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState
import numpy as np

class KinematicsNode(Node):
    def __init__(self):
        super().__init__('kinematics_node')

        # Publisher for end-effector position
        self.ee_publisher = self.create_publisher(Point, 'end_effector_position', 10)

        # Publisher for joint commands
        self.joint_command_publisher = self.create_publisher(
            Float64MultiArray, 'joint_commands', 10
        )

        # Subscriber for joint states
        self.joint_state_subscriber = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10
        )

        # Timer for periodic calculations
        self.timer = self.create_timer(0.1, self.kinematics_callback)

        self.current_joints = None
        self.link_lengths = [1.0, 0.8, 0.5]  # Example link lengths

    def joint_state_callback(self, msg):
        """Update current joint angles"""
        self.current_joints = list(msg.position)

    def forward_kinematics(self, joint_angles):
        """Calculate forward kinematics for 3-DOF arm"""
        if len(joint_angles) < 3:
            return None

        theta1, theta2, theta3 = joint_angles[:3]
        l1, l2, l3 = self.link_lengths

        # Calculate end-effector position
        x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2) + l3 * np.cos(theta1 + theta2 + theta3)
        y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2) + l3 * np.sin(theta1 + theta2 + theta3)
        z = 0  # For planar robot

        return np.array([x, y, z])

    def kinematics_callback(self):
        """Main kinematics calculation loop"""
        if self.current_joints is not None:
            # Calculate forward kinematics
            ee_pos = self.forward_kinematics(self.current_joints)

            if ee_pos is not None:
                # Publish end-effector position
                point_msg = Point()
                point_msg.x = float(ee_pos[0])
                point_msg.y = float(ee_pos[1])
                point_msg.z = float(ee_pos[2])

                self.ee_publisher.publish(point_msg)

                self.get_logger().info(f'EE Position: ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})')

def main(args=None):
    rclpy.init(args=args)
    node = KinematicsNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Jacobian Matrix

The Jacobian matrix relates joint velocities to end-effector velocities:

```
v_ee = J * q_dot
```

Where:
- v_ee is the end-effector velocity vector
- J is the Jacobian matrix
- q_dot is the joint velocity vector

### Jacobian Calculation

```python
def calculate_jacobian(robot_model, joint_angles):
    """
    Calculate the geometric Jacobian for a robot.

    Args:
        robot_model: Robot model with DH parameters
        joint_angles: Current joint angles

    Returns:
        Jacobian matrix
    """
    # This is a simplified version - in practice, you'd use more sophisticated methods
    # or libraries like KDL, PyKDL, or modern robotics libraries

    # For a 6-DOF manipulator, the Jacobian is 6x6 (linear + angular velocities)
    # For a 3-DOF planar manipulator, the Jacobian is 3x3 (x, y, theta)

    # Example for 3-DOF planar robot
    theta1, theta2, theta3 = joint_angles
    l1, l2, l3 = [1.0, 0.8, 0.5]  # Link lengths

    # Calculate partial derivatives (simplified)
    # dx/dtheta1
    dx_dt1 = -l1*np.sin(theta1) - l2*np.sin(theta1+theta2) - l3*np.sin(theta1+theta2+theta3)
    # dx/dtheta2
    dx_dt2 = -l2*np.sin(theta1+theta2) - l3*np.sin(theta1+theta2+theta3)
    # dx/dtheta3
    dx_dt3 = -l3*np.sin(theta1+theta2+theta3)

    # dy/dtheta1
    dy_dt1 = l1*np.cos(theta1) + l2*np.cos(theta1+theta2) + l3*np.cos(theta1+theta2+theta3)
    # dy/dtheta2
    dy_dt2 = l2*np.cos(theta1+theta2) + l3*np.cos(theta1+theta2+theta3)
    # dy/dtheta3
    dy_dt3 = l3*np.cos(theta1+theta2+theta3)

    jacobian = np.array([
        [dx_dt1, dx_dt2, dx_dt3],
        [dy_dt1, dy_dt2, dy_dt3],
        [1, 1, 1]  # Simplified angular velocity
    ])

    return jacobian
```

## Singularity Analysis

Singularities occur when the Jacobian matrix loses rank, meaning the robot loses one or more degrees of freedom.

```python
def check_singularities(jacobian):
    """
    Check if the robot is in a singular configuration.

    Args:
        jacobian: Current Jacobian matrix

    Returns:
        True if in singularity, False otherwise
    """
    # Calculate determinant (for square matrices)
    if jacobian.shape[0] == jacobian.shape[1]:
        det = np.linalg.det(jacobian)
        return abs(det) < 1e-6  # Threshold for singularity
    else:
        # For non-square matrices, use condition number
        cond = np.linalg.cond(jacobian)
        return cond > 1e6  # High condition number indicates near singularity

    return False

def handle_singularity(jacobian, desired_velocity):
    """
    Handle singular configurations using damped least squares.
    """
    damping = 0.01  # Damping factor
    I = np.eye(jacobian.shape[1])  # Identity matrix

    # Damped least squares solution
    jacobian_damped = np.linalg.inv(jacobian.T @ jacobian + damping**2 * I) @ jacobian.T
    joint_velocities = jacobian_damped @ desired_velocity

    return joint_velocities
```

## Practical Implementation: KDL (Kinematics and Dynamics Library)

### Using PyKDL for Complex Kinematics

```python
# Note: You may need to install python-orocos-kdl
# sudo apt install python3-orocos-kdl

try:
    import PyKDL as kdl

    def setup_kdl_chain():
        """Setup a KDL chain for kinematics calculations"""
        chain = kdl.Chain()

        # Add segments to the chain
        # Segment(joint, frame)
        chain.addSegment(kdl.Segment(
            kdl.Joint(kdl.Joint.RotZ),
            kdl.Frame(kdl.Vector(0.0, 0.0, 0.0))
        ))

        chain.addSegment(kdl.Segment(
            kdl.Joint(kdl.Joint.RotZ),
            kdl.Frame(kdl.Vector(1.0, 0.0, 0.0))
        ))

        return chain

    def kdl_forward_kinematics(chain, joint_angles):
        """Calculate forward kinematics using KDL"""
        fk_solver = kdl.ChainFkSolverPos_recursive(chain)

        # Create joint array
        q = kdl.JntArray(len(joint_angles))
        for i, angle in enumerate(joint_angles):
            q[i] = angle

        # Calculate end-effector position
        end_frame = kdl.Frame()
        result = fk_solver.JntToCart(q, end_frame)

        if result >= 0:
            pos = end_frame.p
            return [pos[0], pos[1], pos[2]]
        else:
            return None

except ImportError:
    print("PyKDL not available. Using custom implementation instead.")
```

## Visualization

### Visualizing Robot Kinematics

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_robot_2d(joint_angles, link_lengths, target_pos=None):
    """
    Visualize a 2D robot arm with its current configuration.

    Args:
        joint_angles: List of joint angles
        link_lengths: List of link lengths
        target_pos: Optional target position to show
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Calculate joint positions
    x_pos = [0]
    y_pos = [0]

    current_angle = 0
    for i, (angle, length) in enumerate(zip(joint_angles, link_lengths)):
        current_angle += angle
        x_pos.append(x_pos[-1] + length * np.cos(current_angle))
        y_pos.append(y_pos[-1] + length * np.sin(current_angle))

    # Plot robot links
    ax.plot(x_pos, y_pos, 'o-', linewidth=3, markersize=8, label='Robot Arm')

    # Mark joints
    for i, (x, y) in enumerate(zip(x_pos, y_pos)):
        ax.plot(x, y, 'ro', markersize=10)
        ax.text(x, y, f'J{i}', fontsize=12, ha='right')

    # Mark end-effector
    ax.plot(x_pos[-1], y_pos[-1], 'gs', markersize=12, label='End-Effector')

    # Show target if provided
    if target_pos is not None:
        ax.plot(target_pos[0], target_pos[1], 'r*', markersize=15, label='Target')

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend()
    ax.set_title('Robot Arm Configuration')

    plt.show()

# Example usage
joint_angles = [np.pi/4, np.pi/6, -np.pi/3]
link_lengths = [1.0, 0.8, 0.5]
target = [1.2, 0.8]

visualize_robot_2d(joint_angles, link_lengths, target)
```

## Workspace Analysis

Understanding the workspace is crucial for robot design and operation:

```python
def calculate_workspace(robot_params, resolution=0.1):
    """
    Calculate the reachable workspace of a robot.

    Args:
        robot_params: Robot parameters (link lengths, joint limits)
        resolution: Resolution for workspace calculation

    Returns:
        List of reachable points
    """
    link_lengths = robot_params['link_lengths']
    joint_limits = robot_params.get('joint_limits', [(-np.pi, np.pi)] * len(link_lengths))

    reachable_points = []

    # This is a simplified example - in practice, you'd use more sophisticated methods
    for theta1 in np.arange(joint_limits[0][0], joint_limits[0][1], resolution):
        for theta2 in np.arange(joint_limits[1][0], joint_limits[1][1], resolution):
            x, y = forward_kinematics_2link(theta1, theta2, link_lengths[0], link_lengths[1])
            reachable_points.append((x, y))

    return reachable_points

def visualize_workspace(robot_params):
    """Visualize the workspace of a robot"""
    points = calculate_workspace(robot_params)

    if points:
        x_coords, y_coords = zip(*points)

        plt.figure(figsize=(10, 10))
        plt.scatter(x_coords, y_coords, s=1, alpha=0.5)
        plt.axis('equal')
        plt.grid(True)
        plt.title('Robot Workspace')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.show()
```

## Applications in Robotics

### Path Planning

Kinematics is essential for path planning:

```python
def plan_cartesian_path(start_pos, end_pos, steps=10):
    """
    Plan a Cartesian path between two points and convert to joint space.

    Args:
        start_pos: Start position [x, y, z]
        end_pos: End position [x, y, z]
        steps: Number of steps in the path

    Returns:
        List of joint configurations for the path
    """
    path = []

    for i in range(steps + 1):
        t = i / steps
        current_pos = [
            start_pos[0] + t * (end_pos[0] - start_pos[0]),
            start_pos[1] + t * (end_pos[1] - start_pos[1]),
            start_pos[2] + t * (end_pos[2] - start_pos[2])
        ]

        # Calculate inverse kinematics for this position
        # (This would require a more sophisticated IK solver in practice)
        path.append(current_pos)

    return path
```

## Troubleshooting Common Issues

### 1. Multiple Solutions

Inverse kinematics often has multiple solutions:

```python
def get_ik_solutions_2link(x, y, l1, l2):
    """Get all possible solutions for 2-link robot"""
    solutions = []

    # Primary solution
    sol1 = inverse_kinematics_2link(x, y, l1, l2)
    if sol1:
        solutions.append(sol1)

    # Elbow-up solution (for robots with this configuration)
    # Additional logic would go here for more complex robots

    return solutions
```

### 2. Joint Limit Violations

Always check joint limits:

```python
def check_joint_limits(joint_angles, joint_limits):
    """Check if joint angles are within limits"""
    for angle, (min_limit, max_limit) in zip(joint_angles, joint_limits):
        if angle < min_limit or angle > max_limit:
            return False
    return True
```

### 3. Numerical Stability

Use appropriate numerical methods:

```python
def robust_inverse_kinematics(target_pose, current_joints, max_iterations=100, tolerance=1e-6):
    """Robust numerical inverse kinematics with convergence checking"""
    joints = np.array(current_joints)

    for i in range(max_iterations):
        # Calculate current pose
        current_pose = forward_kinematics_3d(joints, dh_params)

        # Calculate error
        error = target_pose - current_pose

        # Check convergence
        if np.linalg.norm(error) < tolerance:
            return joints

        # Calculate Jacobian
        jacobian = calculate_jacobian(robot_model, joints)

        # Use damped least squares to avoid singularities
        damping = 0.01
        J_damped = np.linalg.inv(jacobian.T @ jacobian + damping**2 * np.eye(len(joints))) @ jacobian.T
        delta_joints = J_damped @ error

        # Update joint angles
        joints = joints + delta_joints * 0.1  # Small step size for stability

    print("Warning: IK did not converge")
    return joints
```

## Next Steps

After mastering forward and inverse kinematics:

1. Continue to [Robot Dynamics Basics](./dynamics.md) to understand forces and motion
2. Practice implementing kinematics for different robot configurations
3. Explore advanced kinematics libraries and tools
4. Apply kinematics to path planning and motion control

Your understanding of kinematics is now fundamental for controlling robot motion and positioning in the Physical AI and Humanoid Robotics course!
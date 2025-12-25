---
sidebar_position: 5
---

# Sensor Fusion

This guide covers the fundamentals of sensor fusion in robotics, combining data from multiple sensors to create a more accurate and robust perception system. Sensor fusion is essential for reliable robot navigation, mapping, and control.

## Overview

Sensor fusion combines information from multiple sensors to improve the accuracy, reliability, and robustness of perception systems. Rather than relying on a single sensor, fusion leverages the complementary strengths of different sensor types while mitigating their individual weaknesses.

### Key Benefits

- **Improved Accuracy**: Combining sensors reduces overall measurement error
- **Increased Robustness**: Backup sensors maintain functionality when primary fails
- **Enhanced Reliability**: Consistent performance despite individual sensor limitations
- **Extended Capabilities**: Combined sensors enable capabilities not possible with single sensors

### Types of Sensor Fusion

#### By Data Processing Level

1. **Data-Level Fusion**: Combine raw sensor data before processing
2. **Feature-Level Fusion**: Combine extracted features from sensors
3. **Decision-Level Fusion**: Combine decisions from individual sensors
4. **Hybrid Fusion**: Combination of multiple levels

#### By Temporal Characteristics

1. **Temporal Fusion**: Combine measurements over time (filtering)
2. **Spatial Fusion**: Combine measurements from different spatial locations
3. **Multi-Modal Fusion**: Combine different sensor modalities

## Mathematical Foundations

### Probabilistic Framework

Sensor fusion typically operates in a probabilistic framework where sensor measurements are treated as probability distributions:

```python
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

class ProbabilisticSensorFusion:
    """Implement probabilistic sensor fusion methods"""

    def __init__(self):
        self.belief = None  # Current state belief (mean, covariance)

    def gaussian_fusion(self, measurements, uncertainties):
        """
        Fuse multiple Gaussian measurements
        measurements: list of measurement values
        uncertainties: list of standard deviations (or covariances)
        """
        # Convert to information form (precision = 1/variance)
        precisions = [1 / (unc**2) for unc in uncertainties]
        weights = [prec / sum(precisions) for prec in precisions]

        # Weighted average of measurements
        fused_mean = sum(w * m for w, m in zip(weights, measurements))

        # Combined uncertainty
        fused_precision = sum(precisions)
        fused_variance = 1 / fused_precision
        fused_std = np.sqrt(fused_variance)

        return fused_mean, fused_std

    def kalman_fusion(self, prediction_mean, prediction_cov,
                      measurement_means, measurement_covs, measurement_matrices):
        """
        Multi-sensor Kalman fusion
        prediction_mean: Prior state estimate
        prediction_cov: Prior state covariance
        measurement_means: List of measurement vectors
        measurement_covs: List of measurement covariances
        measurement_matrices: List of measurement matrices (H matrices)
        """
        # Convert to numpy arrays
        x_pred = np.array(prediction_mean).reshape(-1, 1)
        P_pred = np.array(prediction_cov)

        # Combined innovation and innovation covariance
        total_innovation = np.zeros_like(x_pred)
        total_gain = np.zeros((P_pred.shape[0], P_pred.shape[1]))

        # Process each sensor
        for i, (z_meas, R_meas, H_meas) in enumerate(zip(measurement_means, measurement_covs, measurement_matrices)):
            z_meas = np.array(z_meas).reshape(-1, 1)
            R_meas = np.array(R_meas)
            H_meas = np.array(H_meas)

            # Innovation
            innovation = z_meas - H_meas @ x_pred

            # Innovation covariance
            S = H_meas @ P_pred @ H_meas.T + R_meas

            # Kalman gain
            K = P_pred @ H_meas.T @ np.linalg.inv(S)

            # Update state estimate
            x_pred = x_pred + K @ innovation

            # Update covariance
            P_pred = (np.eye(P_pred.shape[0]) - K @ H_meas) @ P_pred

        return x_pred.flatten(), P_pred

    def particle_filter_fusion(self, particles, weights, measurements, measurement_models):
        """
        Particle filter based sensor fusion
        particles: Array of particles [n_particles, state_dim]
        weights: Array of particle weights [n_particles]
        measurements: List of measurements from different sensors
        measurement_models: List of functions that compute likelihood for each sensor
        """
        n_particles = len(particles)

        # Initialize combined weights
        combined_weights = np.ones(n_particles) / n_particles

        # Process each sensor
        for i, (measurement, meas_model) in enumerate(zip(measurements, measurement_models)):
            # Compute likelihood for each particle
            likelihoods = np.array([meas_model(particle, measurement) for particle in particles])

            # Multiply weights by likelihoods
            combined_weights *= likelihoods

        # Normalize weights
        combined_weights /= np.sum(combined_weights)

        # Resample if effective sample size is too low
        n_eff = 1.0 / np.sum(combined_weights**2)
        if n_eff < n_particles / 2:
            particles, combined_weights = self.resample(particles, combined_weights)

        return particles, combined_weights

    def resample(self, particles, weights):
        """Systematic resampling"""
        n_particles = len(particles)

        # Cumulative sum of weights
        cumulative_sum = np.cumsum(weights)

        # Uniform sampling
        u = np.random.uniform(0, 1.0/n_particles)
        indices = []
        i = 0

        for j in range(n_particles):
            while cumulative_sum[i] < u:
                i += 1
            indices.append(i)
            u += 1.0/n_particles

        # Resample particles and reset weights
        resampled_particles = particles[indices]
        resampled_weights = np.ones(n_particles) / n_particles

        return resampled_particles, resampled_weights

# Example usage
def demonstrate_probabilistic_fusion():
    """Demonstrate probabilistic fusion methods"""
    fusion = ProbabilisticSensorFusion()

    # Example: Fusing GPS and IMU position measurements
    gps_measurement = 10.2  # meters
    gps_uncertainty = 2.0   # meters

    imu_measurement = 9.8   # meters (integrated from velocity)
    imu_uncertainty = 0.5   # meters

    # Fuse measurements
    fused_pos, fused_unc = fusion.gaussian_fusion(
        [gps_measurement, imu_measurement],
        [gps_uncertainty, imu_uncertainty]
    )

    print(f"GPS measurement: {gps_measurement:.2f} ± {gps_uncertainty:.2f}")
    print(f"IMU measurement: {imu_measurement:.2f} ± {imu_uncertainty:.2f}")
    print(f"Fused measurement: {fused_pos:.2f} ± {fused_unc:.2f}")

    # The fused result should be closer to the more accurate IMU measurement
    # but still incorporates information from GPS
```

### Kalman Filters

Kalman filters are fundamental for sensor fusion, particularly for state estimation:

```python
class ExtendedKalmanFilter:
    """Extended Kalman Filter for nonlinear systems"""

    def __init__(self, state_dim, measurement_dim):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim

        # Initialize state vector (mean)
        self.x = np.zeros(state_dim)

        # Initialize covariance matrix
        self.P = np.eye(state_dim) * 1000  # High initial uncertainty

        # Process noise covariance
        self.Q = np.eye(state_dim) * 0.1

        # Measurement noise covariance (will be set per sensor)
        self.R = np.eye(measurement_dim)

        # State transition matrix (will be computed based on system model)
        self.F = np.eye(state_dim)

        # Measurement matrix (will be set based on sensor model)
        self.H = np.zeros((measurement_dim, state_dim))

    def predict(self, control_input=None):
        """Prediction step: predict next state"""
        # State transition (for a simple motion model)
        # This would be customized based on your specific system
        dt = 0.1  # Time step

        # Example: 2D position with velocity (x, y, vx, vy)
        # State: [x, y, vx, vy]
        F = np.array([
            [1, 0, dt, 0],   # x <- x + vx*dt
            [0, 1, 0, dt],   # y <- y + vy*dt
            [0, 0, 1, 0],   # vx <- vx (constant velocity model)
            [0, 0, 0, 1]    # vy <- vy
        ])

        # Apply control input if provided
        if control_input is not None:
            # This would add control effects to the state transition
            pass

        # Predict state
        self.x = F @ self.x

        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q

    def update(self, measurement, sensor_R=None, sensor_H=None):
        """Update step: incorporate measurement"""
        # Use provided sensor parameters or defaults
        R = sensor_R if sensor_R is not None else self.R
        H = sensor_H if sensor_H is not None else self.H

        # Innovation (measurement residual)
        innovation = measurement - H @ self.x

        # Innovation covariance
        S = H @ self.P @ H.T + R

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update state
        self.x = self.x + K @ innovation

        # Update covariance
        I = np.eye(len(self.x))
        self.P = (I - K @ H) @ self.P

    def get_state(self):
        """Get current state estimate"""
        return self.x.copy()

    def get_covariance(self):
        """Get current state covariance"""
        return self.P.copy()

class MultiSensorEKF:
    """Multi-sensor Extended Kalman Filter"""

    def __init__(self):
        # State: [x, y, z, vx, vy, vz, qx, qy, qz, qw] (position, velocity, orientation)
        self.state_dim = 10
        self.ekf = ExtendedKalmanFilter(self.state_dim, self.state_dim)

    def process_sensor_measurement(self, sensor_type, measurement, timestamp):
        """Process measurement from different sensor types"""
        # Define measurement matrix for each sensor type
        if sensor_type == 'imu':
            # IMU measures: angular velocity [wx, wy, wz], linear acceleration [ax, ay, az]
            # Map to state: angular velocity -> orientation change, acceleration -> velocity change
            H = np.zeros((6, self.state_dim))
            H[0:3, 6:9] = np.eye(3)  # Angular velocity maps to orientation rates
            # Acceleration affects velocity (simplified)

            R = np.diag([0.01, 0.01, 0.01, 0.1, 0.1, 0.1])  # IMU noise

        elif sensor_type == 'gps':
            # GPS measures: position [x, y, z]
            H = np.zeros((3, self.state_dim))
            H[0:3, 0:3] = np.eye(3)  # Position maps directly to state

            R = np.diag([2.0, 2.0, 3.0])  # GPS noise (typically 2-3m horizontally)

        elif sensor_type == 'lidar':
            # LiDAR measures: position relative to known landmarks
            # This is simplified - real LiDAR would need landmark association
            H = np.zeros((3, self.state_dim))
            H[0:3, 0:3] = np.eye(3)  # Position measurement

            R = np.diag([0.1, 0.1, 0.2])  # LiDAR noise (typically accurate for position)

        elif sensor_type == 'camera':
            # Camera measures: visual features that can be converted to position
            # This is simplified - real camera fusion would involve feature tracking
            H = np.zeros((2, self.state_dim))  # 2D image coordinates
            H[0:2, 0:2] = np.eye(2)  # Position affects image features

            R = np.diag([5.0, 5.0])  # Camera noise in pixels converted to position

        else:
            raise ValueError(f"Unknown sensor type: {sensor_type}")

        # Update EKF with measurement
        self.ekf.update(measurement, R, H)

    def predict_state(self, dt):
        """Predict state forward in time"""
        # This would update the state transition matrix based on dt
        self.ekf.predict()

    def get_fused_estimate(self):
        """Get the fused state estimate"""
        return self.ekf.get_state(), self.ekf.get_covariance()
```

## Common Sensor Fusion Techniques

### 1. Complementary Filter

Simple fusion technique that combines high-frequency and low-frequency signals:

```python
class ComplementaryFilter:
    """Complementary filter for fusing accelerometer and gyroscope data"""

    def __init__(self, alpha=0.98):
        """
        alpha: Weight for gyroscope (0-1)
               Higher alpha = trust gyroscope more
               Lower alpha = trust accelerometer more
        """
        self.alpha = alpha
        self.angle = 0.0
        self.bias = 0.0
        self.dt = 0.01  # Default time step

    def update(self, gyro_reading, accel_reading, dt=None):
        """Update filter with new measurements"""
        if dt is not None:
            self.dt = dt

        # Integrate gyroscope reading
        gyro_angle = self.angle + gyro_reading * self.dt

        # Use accelerometer to get angle (when not accelerating)
        accel_angle = np.arctan2(accel_reading[1], accel_reading[2])

        # Fuse using complementary filter
        self.angle = self.alpha * gyro_angle + (1 - self.alpha) * accel_angle

        return self.angle

# Example: Fusing IMU sensors
def example_imu_fusion():
    """Example of fusing IMU sensors"""
    cf = ComplementaryFilter(alpha=0.95)

    # Simulated sensor data
    gyro_data = [0.1, 0.15, 0.12, 0.08, 0.11]  # rad/s
    accel_data = [
        [0.1, 9.8, 0.2],
        [0.15, 9.75, 0.25],
        [0.08, 9.82, 0.18],
        [0.12, 9.78, 0.22],
        [0.09, 9.81, 0.21]
    ]

    for gyro, accel in zip(gyro_data, accel_data):
        angle = cf.update(gyro, accel)
        print(f"Fused angle: {np.degrees(angle):.2f}°")
```

### 2. Covariance Intersection

Method for fusing estimates when correlation between sensors is unknown:

```python
def covariance_intersection(estimates, covariances):
    """
    Fuse estimates using covariance intersection
    Handles unknown correlations between estimates
    """
    # Convert to information form
    n = len(estimates)
    if n != len(covariances):
        raise ValueError("Number of estimates must match number of covariances")

    # Calculate information matrices (inverse covariances)
    infos = [np.linalg.inv(cov) for cov in covariances]

    # Calculate total information
    total_info = sum(infos)

    # Calculate weights using covariance intersection
    # This is a simplified version - full CI requires iterative weight calculation
    weights = [1.0/n] * n  # Equal weighting for simplicity

    # Calculate fused estimate
    fused_estimate = np.zeros_like(estimates[0])
    for w, est in zip(weights, estimates):
        fused_estimate += w * est

    # Calculate fused covariance
    fused_info = total_info  # Simplified - full CI has more complex weight calculation
    fused_covariance = np.linalg.inv(fused_info)

    return fused_estimate, fused_covariance
```

### 3. Dempster-Shafer Theory

For handling uncertain and conflicting evidence:

```python
class DempsterShaferFusion:
    """Dempster-Shafer theory for uncertainty fusion"""

    def __init__(self, frame_of_discernment):
        """
        frame_of_discernment: Set of possible hypotheses
        e.g., ['obstacle', 'free_space', 'unknown']
        """
        self.theta = set(frame_of_discernment)
        self.mass_functions = []  # List of mass functions from different sensors

    def add_evidence(self, mass_function):
        """
        Add evidence from a sensor
        mass_function: dict with keys as subsets of theta and values as masses
        e.g., {'obstacle': 0.6, 'free_space': 0.3, 'obstacle,free_space': 0.1}
        """
        # Validate mass function
        total_mass = sum(mass_function.values())
        if abs(total_mass - 1.0) > 1e-6:
            raise ValueError(f"Mass function must sum to 1, got {total_mass}")

        self.mass_functions.append(mass_function)

    def combine_evidence(self):
        """Combine all evidence using Dempster's rule of combination"""
        if not self.mass_functions:
            return {}

        # Start with first mass function
        result = self.mass_functions[0].copy()

        # Combine with each subsequent mass function
        for i in range(1, len(self.mass_functions)):
            result = self.combine_two_mass_functions(result, self.mass_functions[i])

        return result

    def combine_two_mass_functions(self, m1, m2):
        """Combine two mass functions using Dempster's rule"""
        result = {}

        # Combine all pairs of focal elements
        for A, mass_a in m1.items():
            for B, mass_b in m2.items():
                intersection = self._intersect_sets(A, B)
                if intersection:  # Non-empty intersection
                    if intersection in result:
                        result[intersection] += mass_a * mass_b
                    else:
                        result[intersection] = mass_a * mass_b

        # Normalize (remove conflict)
        total = sum(result.values())
        if total > 0:
            result = {k: v/total for k, v in result.items()}

        return result

    def _intersect_sets(self, A, B):
        """Find intersection of two sets (represented as strings or sets)"""
        if isinstance(A, str):
            A = set(A.split(','))
        if isinstance(B, str):
            B = set(B.split(','))
        if isinstance(A, (list, tuple)):
            A = set(A)
        if isinstance(B, (list, tuple)):
            B = set(B)

        intersection = A.intersection(B)
        return ','.join(sorted(intersection)) if intersection else None

# Example usage for obstacle detection
def example_dempster_shafer():
    """Example of using Dempster-Shafer for obstacle detection"""
    # Frame of discernment: what we're trying to determine
    frame = {'obstacle', 'free_space', 'unknown'}

    ds_fusion = DempsterShaferFusion(frame)

    # Evidence from LiDAR (detects obstacle)
    lidar_evidence = {
        'obstacle': 0.7,
        'free_space': 0.2,
        'unknown': 0.1
    }

    # Evidence from camera (detects free space)
    camera_evidence = {
        'obstacle': 0.1,
        'free_space': 0.8,
        'unknown': 0.1
    }

    # Add evidence
    ds_fusion.add_evidence(lidar_evidence)
    ds_fusion.add_evidence(camera_evidence)

    # Combine evidence
    fused_result = ds_fusion.combine_evidence()
    print(f"Fused belief: {fused_result}")
```

## Multi-Sensor Integration Patterns

### Robot State Estimation

Combine multiple sensors for complete robot state estimation:

```python
class RobotStateEstimator:
    """Estimate complete robot state using multiple sensors"""

    def __init__(self):
        # State vector: [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
        # Position, orientation, linear velocity, angular velocity
        self.state_dim = 12
        self.ekf = ExtendedKalmanFilter(self.state_dim, self.state_dim)

        # Sensor timestamps for synchronization
        self.last_imu_time = None
        self.last_gps_time = None
        self.last_lidar_time = None

        # Sensor data buffers for interpolation
        self.imu_buffer = []
        self.gps_buffer = []
        self.odom_buffer = []

    def update_with_imu(self, imu_msg):
        """Process IMU measurement"""
        # Extract linear acceleration and angular velocity
        acc = [imu_msg.linear_acceleration.x,
               imu_msg.linear_acceleration.y,
               imu_msg.linear_acceleration.z]
        gyro = [imu_msg.angular_velocity.x,
                imu_msg.angular_velocity.y,
                imu_msg.angular_velocity.z]

        # Measurement vector: [ax, ay, az, wx, wy, wz]
        measurement = np.array(acc + gyro)

        # Measurement matrix for IMU
        H = np.zeros((6, self.state_dim))
        # Acceleration affects velocity (derivative)
        H[0:3, 6:9] = np.eye(3)  # Linear acceleration -> velocity change
        # Angular velocity maps directly to state
        H[3:6, 9:12] = np.eye(3)  # Angular velocity

        # IMU noise covariance
        R = np.diag([0.01, 0.01, 0.01,  # accelerometer noise
                     0.001, 0.001, 0.001])  # gyroscope noise

        self.ekf.update(measurement, R, H)
        self.last_imu_time = imu_msg.header.stamp

    def update_with_gps(self, gps_msg):
        """Process GPS measurement"""
        # Extract position
        pos = [gps_msg.latitude, gps_msg.longitude, gps_msg.altitude]

        # Convert to local coordinates (simplified)
        # In practice, you'd use proper coordinate transformation
        local_pos = self.gps_to_local_coordinates(pos)

        # Measurement vector: [x, y, z]
        measurement = np.array(local_pos)

        # Measurement matrix for GPS (position only)
        H = np.zeros((3, self.state_dim))
        H[0:3, 0:3] = np.eye(3)  # Position maps directly

        # GPS noise covariance (varies with satellite geometry)
        R = np.diag([2.0, 2.0, 5.0])  # 2m horizontal, 5m vertical

        self.ekf.update(measurement, R, H)
        self.last_gps_time = gps_msg.header.stamp

    def update_with_odometry(self, odom_msg):
        """Process odometry measurement"""
        # Extract pose and twist
        pose = odom_msg.pose.pose
        twist = odom_msg.twist.twist

        # Measurement vector: [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
        # For this example, we'll use position and linear velocity
        pos = [pose.position.x, pose.position.y, pose.position.z]
        lin_vel = [twist.linear.x, twist.linear.y, twist.linear.z]

        measurement = np.array(pos + lin_vel)

        # Measurement matrix for odometry
        H = np.zeros((6, self.state_dim))  # Using position and linear velocity
        H[0:3, 0:3] = np.eye(3)  # Position
        H[3:6, 6:9] = np.eye(3)  # Linear velocity

        # Odometry noise covariance
        R = np.diag([0.01, 0.01, 0.02,  # position noise
                     0.05, 0.05, 0.05])  # velocity noise

        self.ekf.update(measurement, R, H)

    def gps_to_local_coordinates(self, gps_coords):
        """Convert GPS coordinates to local Cartesian coordinates"""
        # This is a simplified conversion
        # In practice, use proper coordinate transformation libraries
        lat, lon, alt = gps_coords

        # Reference point (should be set to initial GPS position)
        ref_lat, ref_lon, ref_alt = getattr(self, 'ref_gps', (0, 0, 0))

        # Simple conversion (valid for small areas)
        # More accurate conversion needed for larger areas
        R = 6371000  # Earth radius in meters

        x = (lon - ref_lon) * (R * np.cos(np.radians(lat)))
        y = (lat - ref_lat) * (R)
        z = alt - ref_alt

        if not hasattr(self, 'ref_gps'):
            self.ref_gps = (lat, lon, alt)

        return [x, y, z]

    def get_robot_state(self):
        """Get current robot state estimate"""
        state, covariance = self.ekf.get_state(), self.ekf.get_covariance()

        # Extract components
        position = state[0:3]
        orientation_quat = self.state[3:7]  # This would need proper quaternion handling
        linear_velocity = state[7:10]
        angular_velocity = state[10:13]

        return {
            'position': position,
            'orientation': orientation_quat,
            'linear_velocity': linear_velocity,
            'angular_velocity': angular_velocity,
            'covariance': covariance
        }

    def predict_state(self, dt):
        """Predict state forward in time"""
        # This would implement the state transition model
        # For a motion model: integrate velocities to get new positions
        self.ekf.predict()
```

### Sensor Synchronization

Properly synchronize data from multiple sensors:

```python
from collections import deque
import threading
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

class SensorSynchronizer:
    """Synchronize multiple sensor streams"""

    def __init__(self, node, max_queue_size=10):
        self.node = node
        self.max_queue_size = max_queue_size
        self.sensors = {}
        self.synchronized_callback = None
        self.lock = threading.Lock()

    def add_sensor(self, topic, msg_type, qos_profile=None):
        """Add a sensor topic to synchronization"""
        if qos_profile is None:
            qos_profile = QoSProfile(
                depth=10,
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                history=QoSHistoryPolicy.KEEP_LAST
            )

        self.sensors[topic] = {
            'subscriber': Subscriber(
                self.node, msg_type, topic, qos_profile=qos_profile
            ),
            'buffer': deque(maxlen=self.max_queue_size),
            'timestamps': deque(maxlen=self.max_queue_size)
        }

    def set_synchronized_callback(self, callback):
        """Set callback for synchronized data"""
        self.synchronized_callback = callback

    def start_synchronization(self, time_window=0.1):
        """Start approximate time synchronization"""
        if len(self.sensors) < 2:
            raise ValueError("Need at least 2 sensors for synchronization")

        # Create subscribers list
        subscribers = [sensor_info['subscriber'] for sensor_info in self.sensors.values()]

        # Create synchronizer
        self.approx_sync = ApproximateTimeSynchronizer(
            subscribers,
            queue_size=10,
            slop=time_window  # Time tolerance in seconds
        )
        self.approx_sync.registerCallback(self._synchronized_callback)

    def _synchronized_callback(self, *msgs):
        """Internal callback when messages are synchronized"""
        if self.synchronized_callback:
            # Create dictionary of synchronized messages
            sync_data = {}
            for i, (topic, sensor_info) in enumerate(self.sensors.items()):
                sync_data[topic] = msgs[i]

            self.synchronized_callback(sync_data)

    def interpolate_sensor_data(self, topic, target_time):
        """Interpolate sensor data to target time"""
        with self.lock:
            sensor_info = self.sensors[topic]
            timestamps = list(sensor_info['timestamps'])
            data_buffer = list(sensor_info['buffer'])

            if len(timestamps) < 2:
                return None

            # Find nearest timestamps
            time_diffs = [abs(ts - target_time) for ts in timestamps]
            nearest_idx = time_diffs.index(min(time_diffs))

            # Simple interpolation between adjacent points
            if nearest_idx > 0 and nearest_idx < len(timestamps) - 1:
                # Interpolate between adjacent points
                t1, t2 = timestamps[nearest_idx-1], timestamps[nearest_idx]
                d1, d2 = data_buffer[nearest_idx-1], data_buffer[nearest_idx]

                if t1 != t2:
                    # Linear interpolation
                    alpha = (target_time - t1) / (t2 - t1)
                    interpolated_data = self.interpolate_data(d1, d2, alpha)
                    return interpolated_data

        return data_buffer[nearest_idx]

    def interpolate_data(self, data1, data2, alpha):
        """Interpolate between two sensor data messages"""
        # This would implement interpolation based on data type
        # For simplicity, return the later data point
        # In practice, implement proper interpolation for each sensor type
        return data2
```

## Implementation Example: ROS 2 Sensor Fusion Node

Create a complete ROS 2 node that fuses multiple sensors:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, NavSatFix, LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Header
import numpy as np
from scipy.spatial.transform import Rotation as R

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')

        # Create sensor fusion estimator
        self.estimator = RobotStateEstimator()

        # Create QoS profiles for different sensor types
        sensor_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST
        )

        # Subscribe to different sensor topics
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, sensor_qos
        )

        self.gps_sub = self.create_subscription(
            NavSatFix, '/gps/fix', self.gps_callback, sensor_qos
        )

        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, sensor_qos
        )

        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, sensor_qos
        )

        # Publisher for fused state
        self.fused_state_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/fused_pose', 10
        )

        # Timer for prediction and publishing
        self.timer = self.create_timer(0.05, self.prediction_callback)  # 20 Hz

        self.get_logger().info('Sensor fusion node initialized')

    def imu_callback(self, msg):
        """Process IMU data"""
        try:
            self.estimator.update_with_imu(msg)
        except Exception as e:
            self.get_logger().error(f'Error processing IMU: {e}')

    def gps_callback(self, msg):
        """Process GPS data"""
        try:
            self.estimator.update_with_gps(msg)
        except Exception as e:
            self.get_logger().error(f'Error processing GPS: {e}')

    def odom_callback(self, msg):
        """Process odometry data"""
        try:
            self.estimator.update_with_odometry(msg)
        except Exception as e:
            self.get_logger().error(f'Error processing odometry: {e}')

    def scan_callback(self, msg):
        """Process LiDAR scan data (for obstacle detection)"""
        try:
            # Process scan for obstacle detection
            obstacles = self.process_scan_for_obstacles(msg)

            # Update state estimator with obstacle information
            # This would involve more complex processing in practice
            self.update_obstacle_information(obstacles)

        except Exception as e:
            self.get_logger().error(f'Error processing scan: {e}')

    def process_scan_for_obstacles(self, scan_msg):
        """Process LiDAR scan to detect obstacles"""
        # Convert scan ranges to Cartesian coordinates
        angles = np.array([
            scan_msg.angle_min + i * scan_msg.angle_increment
            for i in range(len(scan_msg.ranges))
        ])
        ranges = np.array(scan_msg.ranges)

        # Filter valid ranges
        valid_mask = (ranges >= scan_msg.range_min) & (ranges <= scan_msg.range_max)
        valid_angles = angles[valid_mask]
        valid_ranges = ranges[valid_mask]

        # Convert to Cartesian
        x_coords = valid_ranges * np.cos(valid_angles)
        y_coords = valid_ranges * np.sin(valid_angles)

        # Simple obstacle clustering (in practice, use more sophisticated methods)
        points = np.column_stack((x_coords, y_coords))
        obstacles = self.cluster_obstacles(points)

        return obstacles

    def cluster_obstacles(self, points, min_distance=0.5):
        """Simple obstacle clustering"""
        if len(points) == 0:
            return []

        clusters = []
        visited = set()

        for i, point in enumerate(points):
            if i in visited:
                continue

            cluster = [point]
            visited.add(i)

            # Find nearby points
            for j, other_point in enumerate(points):
                if j in visited:
                    continue

                distance = np.linalg.norm(point - other_point)
                if distance < min_distance:
                    cluster.append(other_point)
                    visited.add(j)

            if len(cluster) > 2:  # Minimum points for valid obstacle
                cluster_center = np.mean(cluster, axis=0)
                cluster_size = np.std(cluster, axis=0)
                clusters.append({
                    'position': cluster_center,
                    'size': cluster_size,
                    'points': len(cluster)
                })

        return clusters

    def update_obstacle_information(self, obstacles):
        """Update state estimator with obstacle information"""
        # This would integrate obstacle information into the state estimation
        # For now, just log the obstacles
        for obstacle in obstacles:
            self.get_logger().info(
                f'Detected obstacle at ({obstacle["position"][0]:.2f}, {obstacle["position"][1]:.2f})'
            )

    def prediction_callback(self):
        """Prediction and publishing loop"""
        # Predict state forward in time
        dt = 0.05  # 20 Hz timer
        self.estimator.predict_state(dt)

        # Get fused state
        state_info = self.estimator.get_robot_state()

        if state_info:
            # Create and publish fused pose message
            pose_msg = PoseWithCovarianceStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = 'map'

            # Set position
            pose_msg.pose.pose.position.x = float(state_info['position'][0])
            pose_msg.pose.pose.position.y = float(state_info['position'][1])
            pose_msg.pose.pose.position.z = float(state_info['position'][2])

            # Set orientation (simplified - would need proper quaternion handling)
            # For now, set to identity
            pose_msg.pose.pose.orientation.w = 1.0

            # Set covariance
            pose_msg.pose.covariance = state_info['covariance'][:36].flatten().tolist()

            self.fused_state_publisher.publish(pose_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SensorFusionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down sensor fusion node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Advanced Fusion Techniques

### Particle Filtering for Non-linear Systems

For highly non-linear systems where Kalman filters aren't suitable:

```python
class ParticleFilter:
    """Particle filter for non-linear/non-Gaussian systems"""

    def __init__(self, state_dim, n_particles=1000):
        self.state_dim = state_dim
        self.n_particles = n_particles

        # Initialize particles randomly
        self.particles = np.random.randn(n_particles, state_dim) * 10
        self.weights = np.ones(n_particles) / n_particles

        # Process noise
        self.process_noise = np.eye(state_dim) * 0.1

    def predict(self, control_input=None):
        """Predict particle states forward in time"""
        for i in range(self.n_particles):
            # Apply motion model with noise
            self.particles[i] += np.random.multivariate_normal(
                np.zeros(self.state_dim), self.process_noise
            )

            # Apply control if provided
            if control_input is not None:
                self.particles[i] += control_input

    def update(self, measurement, measurement_function, measurement_noise):
        """Update particle weights based on measurement"""
        for i in range(self.n_particles):
            # Predict measurement for this particle
            predicted_measurement = measurement_function(self.particles[i])

            # Calculate likelihood of actual measurement given particle
            diff = measurement - predicted_measurement
            likelihood = self.gaussian_likelihood(diff, measurement_noise)

            # Update weight
            self.weights[i] *= likelihood

        # Normalize weights
        self.weights += 1e-300  # Avoid numerical issues
        self.weights /= np.sum(self.weights)

    def gaussian_likelihood(self, diff, cov):
        """Calculate Gaussian likelihood"""
        inv_cov = np.linalg.inv(cov)
        exponent = -0.5 * diff.T @ inv_cov @ diff
        norm = 1.0 / np.sqrt((2 * np.pi) ** len(diff) * np.linalg.det(cov))
        return norm * np.exp(exponent)

    def resample(self):
        """Resample particles based on weights"""
        # Effective sample size
        n_eff = 1.0 / np.sum(self.weights**2)

        # Resample if effective sample size is too low
        if n_eff < self.n_particles / 2:
            indices = np.random.choice(
                self.n_particles,
                size=self.n_particles,
                p=self.weights
            )

            self.particles = self.particles[indices]
            self.weights = np.ones(self.n_particles) / self.n_particles

    def estimate(self):
        """Get state estimate as weighted average of particles"""
        return np.average(self.particles, axis=0, weights=self.weights)

class MultiSensorParticleFilter:
    """Particle filter for multi-sensor fusion"""

    def __init__(self, state_dim, n_particles=1000):
        self.pf = ParticleFilter(state_dim, n_particles)

    def update_with_multiple_sensors(self, sensor_measurements, sensor_functions, sensor_noises):
        """
        Update with multiple sensor measurements
        sensor_measurements: list of measurements from different sensors
        sensor_functions: list of functions that map state to sensor measurement
        sensor_noises: list of noise covariances for each sensor
        """
        # For each particle, compute likelihood for all sensors
        for i in range(self.pf.n_particles):
            total_likelihood = 1.0

            for meas, func, noise in zip(sensor_measurements, sensor_functions, sensor_noises):
                predicted_meas = func(self.pf.particles[i])
                diff = meas - predicted_meas

                # Calculate likelihood for this sensor
                likelihood = self.pf.gaussian_likelihood(diff, noise)
                total_likelihood *= likelihood

            # Update particle weight with combined likelihood
            self.pf.weights[i] *= total_likelihood

        # Normalize weights
        self.pf.weights += 1e-300
        self.pf.weights /= np.sum(self.pf.weights)
```

### Deep Learning Fusion

Modern approaches use neural networks for sensor fusion:

```python
import torch
import torch.nn as nn

class DeepSensorFusion(nn.Module):
    """Neural network for sensor fusion"""

    def __init__(self, sensor_configs):
        """
        sensor_configs: dict with sensor names and their input dimensions
        e.g., {'imu': 6, 'lidar': 360, 'camera': 240*320*3}
        """
        super().__init__()

        self.sensor_configs = sensor_configs
        self.sensors = list(sensor_configs.keys())

        # Encoder for each sensor
        self.encoders = nn.ModuleDict()
        for sensor, dim in sensor_configs.items():
            self.encoders[sensor] = nn.Sequential(
                nn.Linear(dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32)
            )

        # Fusion layer
        total_encoded_dim = 32 * len(self.sensors)  # Assuming 32-dim encoding per sensor
        self.fusion = nn.Sequential(
            nn.Linear(total_encoded_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12)  # Output: [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
        )

    def forward(self, sensor_inputs):
        """
        sensor_inputs: dict with sensor names and their tensor values
        """
        encoded_features = []

        for sensor in self.sensors:
            if sensor in sensor_inputs:
                encoded = self.encoders[sensor](sensor_inputs[sensor])
                encoded_features.append(encoded)

        # Concatenate all encoded features
        if encoded_features:
            concatenated = torch.cat(encoded_features, dim=-1)
            fused_output = self.fusion(concatenated)
            return fused_output
        else:
            # Return zeros if no inputs
            return torch.zeros(1, 12)

# Training example (simplified)
def train_deep_fusion():
    """Example training loop for deep sensor fusion"""
    # Define sensor configurations
    sensor_configs = {
        'imu': 6,      # [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
        'lidar': 360,  # 360 range measurements
        'gps': 3       # [lat, lon, alt] (simplified)
    }

    # Create model
    model = DeepSensorFusion(sensor_configs)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Training loop (simplified)
    for epoch in range(100):
        # In practice, you'd load batches of sensor data and ground truth
        # For this example, we'll use dummy data

        # Dummy inputs (batch_size=1)
        batch_size = 1
        imu_data = torch.randn(batch_size, 6)
        lidar_data = torch.randn(batch_size, 360)
        gps_data = torch.randn(batch_size, 3)

        sensor_inputs = {
            'imu': imu_data,
            'lidar': lidar_data,
            'gps': gps_data
        }

        # Dummy ground truth
        ground_truth = torch.randn(batch_size, 12)  # [x, y, z, vx, vy, vz, ...]

        # Forward pass
        prediction = model(sensor_inputs)
        loss = criterion(prediction, ground_truth)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

    return model
```

## Quality Assurance and Validation

### Fusion Performance Metrics

```python
class FusionPerformanceEvaluator:
    """Evaluate sensor fusion performance"""

    def __init__(self):
        self.metrics = {
            'accuracy': [],
            'consistency': [],
            'robustness': [],
            'latency': [],
            'resource_usage': []
        }

    def evaluate_accuracy(self, fused_estimate, ground_truth):
        """Evaluate fusion accuracy"""
        if ground_truth is not None:
            error = np.linalg.norm(fused_estimate[:3] - ground_truth[:3])  # Position error
            self.metrics['accuracy'].append(error)
            return error
        return None

    def evaluate_consistency(self, fused_estimate, fused_covariance, measurements):
        """Evaluate consistency of estimates with measurements"""
        # Calculate normalized innovation squared (NIS) for consistency check
        innovations = []

        for meas, meas_func in measurements:  # meas_func maps state to measurement space
            predicted_meas = meas_func(fused_estimate)
            innovation = meas - predicted_meas

            # Get measurement covariance (simplified)
            meas_cov = np.eye(len(innovation)) * 0.1  # Placeholder

            # Calculate innovation covariance
            # In practice, you'd need the full Jacobian
            innovation_cov = meas_cov  # Simplified

            # NIS
            nis = innovation.T @ np.linalg.inv(innovation_cov) @ innovation
            innovations.append(nis)

        consistency_metric = np.mean(innovations) if innovations else 0
        self.metrics['consistency'].append(consistency_metric)

        return consistency_metric

    def evaluate_robustness(self, sensor_availability):
        """Evaluate robustness to sensor failures"""
        # Calculate percentage of time with at least minimum sensors
        min_required = 2  # Minimum sensors for basic functionality
        available_count = sum(1 for available in sensor_availability if available)

        robustness_score = min(available_count / min_required, 1.0)
        self.metrics['robustness'].append(robustness_score)

        return robustness_score

    def evaluate_latency(self, processing_times):
        """Evaluate processing latency"""
        avg_latency = np.mean(processing_times) if processing_times else 0
        max_latency = np.max(processing_times) if processing_times else 0

        self.metrics['latency'].append({
            'avg': avg_latency,
            'max': max_latency,
            'std': np.std(processing_times) if processing_times else 0
        })

        return avg_latency, max_latency

    def get_performance_report(self):
        """Generate performance report"""
        report = {}

        for metric_name, values in self.metrics.items():
            if values:
                if isinstance(values[0], dict):
                    # Handle dictionary metrics (like latency)
                    report[metric_name] = {
                        'recent': values[-1],
                        'history': values[-10:]  # Last 10 values
                    }
                else:
                    report[metric_name] = {
                        'current': values[-1],
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': min(values),
                        'max': max(values),
                        'count': len(values)
                    }

        return report

# Example usage
def validate_sensor_fusion():
    """Example of validating sensor fusion system"""
    evaluator = FusionPerformanceEvaluator()

    # Simulate fusion process with metrics
    for i in range(100):
        # Simulate sensor data
        imu_data = np.random.randn(6)
        gps_data = np.random.randn(3)
        lidar_data = np.random.randn(360)

        # Perform fusion (simplified)
        fused_estimate = np.random.randn(12)  # Simulated fused state

        # Evaluate accuracy (with simulated ground truth)
        ground_truth = np.random.randn(12)
        accuracy = evaluator.evaluate_accuracy(fused_estimate, ground_truth)

        # Evaluate consistency
        measurements = [(imu_data, lambda x: x[9:12]),  # IMU measures angular velocity
                       (gps_data, lambda x: x[0:3])]    # GPS measures position
        consistency = evaluator.evaluate_consistency(fused_estimate, np.eye(12), measurements)

        # Evaluate robustness
        sensor_status = [True, True, True]  # All sensors available
        robustness = evaluator.evaluate_robustness(sensor_status)

        # Evaluate latency (simulated)
        processing_time = [0.01 + np.random.randn() * 0.001]  # 10ms ± 1ms
        latency_avg, latency_max = evaluator.evaluate_latency(processing_time)

    # Generate report
    report = evaluator.get_performance_report()
    print("Sensor Fusion Performance Report:")
    for metric, values in report.items():
        if isinstance(values, dict) and 'mean' in values:
            print(f"  {metric}: mean={values['mean']:.3f}, std={values['std']:.3f}")
        else:
            print(f"  {metric}: {values}")
```

## Troubleshooting Common Issues

### Sensor Synchronization Issues

```python
def troubleshoot_sensor_sync():
    """Common sensor synchronization issues and solutions"""
    issues = {
        'timestamp_drift': {
            'problem': 'Sensors have different clock drifts causing temporal misalignment',
            'symptoms': 'Decreased fusion accuracy over time, inconsistent measurements',
            'solutions': [
                'Use hardware synchronization if available',
                'Implement software timestamp correction',
                'Use message_filters.ApproximateTimeSynchronizer with appropriate slop',
                'Check system clock synchronization (NTP)'
            ]
        },
        'frequency_mismatch': {
            'problem': 'Sensors operating at different frequencies causing data loss',
            'symptoms': 'Missing sensor data, irregular fusion updates',
            'solutions': [
                'Implement interpolation between measurements',
                'Use prediction models to estimate intermediate states',
                'Configure sensors for compatible frequencies',
                'Implement proper buffering strategies'
            ]
        },
        'coordinate_frame_mismatch': {
            'problem': 'Sensors in different coordinate frames causing spatial misalignment',
            'symptoms': 'Incorrect position estimates, poor obstacle detection',
            'solutions': [
                'Verify TF tree is properly configured',
                'Use tf2 for coordinate transformations',
                'Double-check sensor mounting positions and orientations',
                'Validate static transforms in URDF'
            ]
        },
        'data_association_errors': {
            'problem': 'Difficulty matching features/measurements across sensors',
            'symptoms': 'Spurious measurements, incorrect fusion results',
            'solutions': [
                'Implement robust data association algorithms',
                'Use feature descriptors for matching',
                'Validate sensor calibration',
                'Implement outlier rejection techniques'
            ]
        }
    }

    return issues

def validate_sensor_setup():
    """Validate sensor setup before fusion"""
    validation_checks = {
        'topic_availability': {
            'description': 'Check if all sensor topics are publishing',
            'command': 'ros2 topic list | grep -E "(imu|scan|gps|odom)"'
        },
        'message_rate': {
            'description': 'Verify sensor message rates are as expected',
            'command': 'ros2 topic hz /sensor_topic_name'
        },
        'tf_tree': {
            'description': 'Check TF tree for proper transformations',
            'command': 'ros2 run tf2_tools view_frames'
        },
        'calibration_files': {
            'description': 'Verify calibration files exist and are accessible',
            'command': 'ls -la $(ros2 pkg prefix sensor_package)/share/sensor_package/config/'
        }
    }

    print("Sensor Setup Validation Checklist:")
    for check_name, check_info in validation_checks.items():
        print(f"\n{check_name.replace('_', ' ').title()}:")
        print(f"  Description: {check_info['description']}")
        print(f"  Command: {check_info['command']}")
```

## Best Practices

### 1. Sensor Selection Best Practices

- **Complementary Sensors**: Choose sensors that complement each other's strengths/weaknesses
- **Redundancy**: Include backup sensors for critical functions
- **Cost-Benefit**: Balance performance improvements with cost increases
- **Power Requirements**: Consider power consumption for mobile robots
- **Environmental Suitability**: Match sensors to operating environment

### 2. Calibration Best Practices

- **Regular Recalibration**: Schedule periodic calibration verification
- **Multi-Sensor Calibration**: Calibrate sensors relative to each other
- **Environmental Factors**: Account for temperature, humidity, vibration
- **Validation Testing**: Test calibration with known reference objects
- **Documentation**: Maintain detailed calibration records

### 3. Fusion Algorithm Best Practices

- **Model Appropriateness**: Choose fusion algorithm suitable for your application
- **Computational Efficiency**: Optimize for target hardware constraints
- **Numerical Stability**: Implement proper numerical safeguards
- **Failure Modes**: Design for graceful degradation when sensors fail
- **Validation**: Continuously validate fusion results

### 4. Implementation Best Practices

- **Modular Design**: Create modular, testable fusion components
- **Real-time Capability**: Ensure algorithms meet timing requirements
- **Memory Management**: Efficiently handle large sensor data volumes
- **Error Handling**: Implement robust error handling and recovery
- **Logging**: Maintain detailed logs for debugging and analysis

### 5. Testing Best Practices

- **Unit Testing**: Test individual fusion components
- **Integration Testing**: Test complete fusion pipeline
- **Hardware-in-Loop**: Test with actual sensors when possible
- **Edge Case Testing**: Test failure scenarios and boundary conditions
- **Performance Testing**: Validate real-time performance requirements

## Advanced Topics

### Adaptive Fusion

```python
class AdaptiveSensorFusion:
    """Adaptive fusion that adjusts parameters based on conditions"""

    def __init__(self):
        self.base_weights = {'imu': 0.3, 'gps': 0.4, 'lidar': 0.3}
        self.context_weights = {}  # Context-dependent weights
        self.performance_monitors = {}  # Track sensor performance

    def update_weights_based_on_context(self, environment_context):
        """Adjust fusion weights based on environment"""
        # Example: In GPS-denied environments, increase IMU and LiDAR weights
        if environment_context.get('gps_denied', False):
            adjusted_weights = self.base_weights.copy()
            adjusted_weights['imu'] += 0.2
            adjusted_weights['lidar'] += 0.1
            adjusted_weights['gps'] = max(0.0, adjusted_weights['gps'] - 0.3)
        else:
            adjusted_weights = self.base_weights.copy()

        # Normalize weights
        total = sum(adjusted_weights.values())
        adjusted_weights = {k: v/total for k, v in adjusted_weights.items()}

        return adjusted_weights

    def monitor_sensor_performance(self, sensor_name, recent_performance):
        """Monitor individual sensor performance"""
        if sensor_name not in self.performance_monitors:
            self.performance_monitors[sensor_name] = []

        self.performance_monitors[sensor_name].append(recent_performance)

        # If performance degrades, reduce weight
        if len(self.performance_monitors[sensor_name]) > 10:
            recent_avg = np.mean(self.performance_monitors[sensor_name][-10:])
            historical_avg = np.mean(self.performance_monitors[sensor_name][:-10])

            if recent_avg < 0.8 * historical_avg:  # Performance degraded by 20%
                # Reduce weight for this sensor
                pass  # Implementation would adjust weights
```

### Multi-Model Fusion

```python
class MultiModelFusion:
    """Fusion using multiple models for different motion patterns"""

    def __init__(self):
        self.models = {
            'constant_velocity': self.constant_velocity_model,
            'constant_acceleration': self.constant_acceleration_model,
            'turning': self.turning_model
        }

        self.model_weights = {model: 1.0/len(self.models) for model in self.models}
        self.model_predictions = {}

    def constant_velocity_model(self, state, dt):
        """Constant velocity motion model"""
        x, y, z, vx, vy, vz = state[:6]
        return np.array([
            x + vx * dt,
            y + vy * dt,
            z + vz * dt,
            vx, vy, vz,  # velocities unchanged
            *state[6:]    # other state variables unchanged
        ])

    def constant_acceleration_model(self, state, dt):
        """Constant acceleration motion model"""
        x, y, z, vx, vy, vz, ax, ay, az = state[:9]
        return np.array([
            x + vx * dt + 0.5 * ax * dt**2,
            y + vy * dt + 0.5 * ay * dt**2,
            z + vz * dt + 0.5 * az * dt**2,
            vx + ax * dt,
            vy + ay * dt,
            vz + az * dt,
            ax, ay, az,  # accelerations unchanged
            *state[9:]    # other state variables unchanged
        ])

    def turning_model(self, state, dt):
        """Turning motion model with angular rates"""
        # More complex model for turning maneuvers
        pass

    def predict_with_multiple_models(self, state, dt):
        """Predict using multiple models"""
        predictions = {}

        for model_name, model_func in self.models.items():
            predictions[model_name] = model_func(state, dt)

        # Combine predictions based on model weights
        combined_prediction = np.zeros_like(state)
        for model_name, pred in predictions.items():
            combined_prediction += self.model_weights[model_name] * pred

        return combined_prediction, predictions
```

## Integration with Navigation and Perception

### Fusion for SLAM

```python
class FusionSLAM:
    """SLAM system using sensor fusion"""

    def __init__(self):
        self.map = {}  # Occupancy grid or point cloud map
        self.pose_estimator = RobotStateEstimator()
        self.loop_closure_detector = None  # For loop closure detection

    def process_sensor_data(self, sensor_data):
        """Process synchronized sensor data for SLAM"""
        # Update pose estimate using fusion
        self.pose_estimator.update_with_sensors(sensor_data)

        # Update map with LiDAR data
        if 'lidar' in sensor_data:
            self.update_map(sensor_data['lidar'], self.pose_estimator.get_position())

        # Detect loop closures using multiple sensors
        if self.should_check_for_loop_closure():
            self.detect_loop_closure(sensor_data)

    def update_map(self, lidar_scan, robot_pose):
        """Update occupancy grid with LiDAR data"""
        # Convert scan to world coordinates using robot pose
        # Update occupancy probabilities
        pass

    def detect_loop_closure(self, sensor_data):
        """Detect when robot returns to previously visited location"""
        # Use visual features, LiDAR signatures, and odometry
        # to detect loop closures
        pass
```

## Next Steps

After mastering sensor fusion:

1. Continue to [Computer Vision Integration](./computer-vision-integration.md) to learn about visual perception
2. Practice implementing fusion algorithms with your robot
3. Experiment with different fusion approaches for your specific application
4. Test fusion systems in both simulation and real environments

Your understanding of sensor fusion is now foundational for creating robust perception systems in the Physical AI and Humanoid Robotics course!
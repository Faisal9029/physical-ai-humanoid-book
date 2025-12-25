---
sidebar_position: 2
---

# Sensor Integration

This guide covers the fundamentals of sensor integration in robotics, including how to connect, configure, and process data from various sensor types. Sensor integration is critical for robot perception and navigation.

## Overview

Sensor integration involves connecting physical or simulated sensors to your robot and processing their data streams. Modern robots typically use multiple sensors to perceive their environment:

- **Cameras**: For visual perception and object recognition
- **LiDAR**: For 3D mapping and obstacle detection
- **IMU**: For orientation and motion tracking
- **Encoders**: For odometry and motion estimation
- **GPS**: For global positioning
- **Force/Torque Sensors**: For contact detection and manipulation

### Sensor Data Flow

The typical sensor data flow in a robotic system:

```
Raw Sensor Data → Preprocessing → Filtering → Sensor Fusion → Perception → Action
```

## Sensor Types and Characteristics

### Camera Sensors

Cameras provide rich visual information but require significant processing:

#### Types
- **RGB Cameras**: Provide color images
- **Depth Cameras**: Provide distance information
- **Stereo Cameras**: Provide 3D information from stereo vision
- **Thermal Cameras**: Provide heat signature information

#### Key Parameters
- **Resolution**: Image dimensions (e.g., 640x480, 1280x720)
- **Frame Rate**: Frames per second (e.g., 30 FPS)
- **Field of View**: Angular extent of the scene (e.g., 60°)
- **Bit Depth**: Color depth (e.g., 8-bit, 16-bit)

### LiDAR Sensors

LiDAR provides accurate 3D distance measurements:

#### Types
- **2D LiDAR**: Single plane scanning
- **3D LiDAR**: Multiple planes for full 3D point clouds
- **Solid-State**: No moving parts, more reliable

#### Key Parameters
- **Range**: Detection distance (e.g., 10m, 100m)
- **Resolution**: Angular resolution (e.g., 0.1°)
- **Scan Rate**: Scans per second (e.g., 10Hz)
- **Points per Scan**: Number of points per revolution

### IMU Sensors

IMUs measure acceleration and angular velocity:

#### Types
- **Accelerometers**: Measure linear acceleration
- **Gyroscopes**: Measure angular velocity
- **Magnetometers**: Measure magnetic field

#### Key Parameters
- **Sampling Rate**: Measurements per second (e.g., 100Hz, 1000Hz)
- **Noise Density**: Noise per square root of bandwidth
- **Bias Stability**: Long-term bias drift
- **Range**: Measurement range (e.g., ±16g for accelerometer)

### Encoders

Encoders measure joint positions and velocities:

#### Types
- **Incremental**: Measure relative motion
- **Absolute**: Measure absolute position
- **Optical**: Use light detection
- **Magnetic**: Use magnetic field detection

#### Key Parameters
- **Resolution**: Counts per revolution
- **Accuracy**: Position accuracy
- **Sampling Rate**: Update frequency

## ROS 2 Sensor Integration

### Sensor Message Types

ROS 2 provides standardized message types for different sensors:

```python
# Common sensor message types
from sensor_msgs.msg import (
    Image,           # Camera images
    CompressedImage, # Compressed camera images
    LaserScan,       # 2D LiDAR data
    PointCloud2,     # 3D point cloud data
    Imu,             # Inertial measurement unit
    JointState,      # Joint positions, velocities, efforts
    MagneticField,   # Magnetic field measurements
    Temperature,     # Temperature measurements
    FluidPressure    # Pressure measurements
)
```

### Camera Integration

#### Camera Publisher Example

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')

        # Publisher for camera images
        self.image_publisher = self.create_publisher(Image, 'camera/image_raw', 10)

        # Publisher for camera info
        self.info_publisher = self.create_publisher(CameraInfo, 'camera/camera_info', 10)

        # OpenCV bridge for converting between ROS and OpenCV formats
        self.bridge = CvBridge()

        # Timer to capture and publish images
        self.timer = self.create_timer(0.033, self.capture_and_publish)  # ~30 FPS

        # Initialize camera (using simulated camera in this example)
        self.cap = cv2.VideoCapture(0)  # Use 0 for default camera

        # If using simulation, you might subscribe to simulation topics instead
        self.get_logger().info('Camera publisher initialized')

    def capture_and_publish(self):
        """Capture image and publish to ROS topic"""
        ret, frame = self.cap.read()

        if ret:
            # Convert OpenCV image to ROS Image message
            image_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            image_msg.header.stamp = self.get_clock().now().to_msg()
            image_msg.header.frame_id = 'camera_link'

            # Publish image
            self.image_publisher.publish(image_msg)

            # Publish camera info
            self.publish_camera_info(image_msg.header)

    def publish_camera_info(self, header):
        """Publish camera calibration information"""
        info_msg = CameraInfo()
        info_msg.header = header
        info_msg.width = 640
        info_msg.height = 480

        # Camera intrinsic matrix (example values)
        info_msg.k = [
            525.0, 0.0, 320.0,  # fx, 0, cx
            0.0, 525.0, 240.0,  # 0, fy, cy
            0.0, 0.0, 1.0       # 0, 0, 1
        ]

        # Distortion coefficients (none in this example)
        info_msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]

        self.info_publisher.publish(info_msg)

def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down camera publisher')
    finally:
        node.cap.release()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Camera Subscriber Example

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class CameraSubscriber(Node):
    def __init__(self):
        super().__init__('camera_subscriber')

        # Subscriber for camera images
        self.subscription = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10
        )

        # OpenCV bridge for converting ROS to OpenCV format
        self.bridge = CvBridge()

        # Initialize OpenCV window
        cv2.namedWindow('Camera Feed', cv2.WINDOW_AUTOSIZE)

        self.get_logger().info('Camera subscriber initialized')

    def image_callback(self, msg):
        """Process incoming camera image"""
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

            # Process the image (example: apply Gaussian blur)
            processed_image = cv2.GaussianBlur(cv_image, (5, 5), 0)

            # Display the image
            cv2.imshow('Camera Feed', processed_image)
            cv2.waitKey(1)  # Allow OpenCV to process window events

            self.get_logger().info(f'Received image: {msg.width}x{msg.height}')

        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = CameraSubscriber()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down camera subscriber')
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### LiDAR Integration

#### LiDAR Publisher Example

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
from math import pi

class LidarPublisher(Node):
    def __init__(self):
        super().__init__('lidar_publisher')

        # Publisher for LiDAR data
        self.scan_publisher = self.create_publisher(LaserScan, 'scan', 10)

        # Timer for publishing scans
        self.timer = self.create_timer(0.1, self.publish_scan)  # 10 Hz

        # LiDAR parameters
        self.angle_min = -pi
        self.angle_max = pi
        self.angle_increment = 2*pi / 360  # 1 degree increments
        self.time_increment = 0.0
        self.scan_time = 0.1  # 100ms per scan
        self.range_min = 0.1
        self.range_max = 30.0

        self.get_logger().info('LiDAR publisher initialized')

    def publish_scan(self):
        """Generate and publish LiDAR scan data"""
        scan_msg = LaserScan()
        scan_msg.header.stamp = self.get_clock().now().to_msg()
        scan_msg.header.frame_id = 'laser_link'

        # Set scan parameters
        scan_msg.angle_min = self.angle_min
        scan_msg.angle_max = self.angle_max
        scan_msg.angle_increment = self.angle_increment
        scan_msg.time_increment = self.time_increment
        scan_msg.scan_time = self.scan_time
        scan_msg.range_min = self.range_min
        scan_msg.range_max = self.range_max

        # Generate example range data (simulating obstacles at various distances)
        num_readings = int((self.angle_max - self.angle_min) / self.angle_increment)

        # Create sample data - in real implementation, this would come from actual sensor
        ranges = []
        intensities = []

        for i in range(num_readings):
            angle = self.angle_min + i * self.angle_increment

            # Simulate some obstacles
            distance = self.range_max  # Default: no obstacle detected

            # Add some simulated obstacles
            if abs(angle) < 0.2:  # Front
                distance = 2.0
            elif abs(angle - pi/4) < 0.1:  # 45 degrees
                distance = 5.0
            elif abs(angle + pi/3) < 0.15:  # -60 degrees
                distance = 3.0

            ranges.append(distance)
            intensities.append(100.0)  # Constant intensity for simplicity

        scan_msg.ranges = ranges
        scan_msg.intensities = intensities

        # Publish the scan
        self.scan_publisher.publish(scan_msg)
        self.get_logger().info(f'Published LiDAR scan with {num_readings} readings')

def main(args=None):
    rclpy.init(args=args)
    node = LidarPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down LiDAR publisher')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### LiDAR Subscriber Example

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
from math import cos, sin

class LidarSubscriber(Node):
    def __init__(self):
        super().__init__('lidar_subscriber')

        # Subscriber for LiDAR data
        self.subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            10
        )

        self.get_logger().info('LiDAR subscriber initialized')

    def scan_callback(self, msg):
        """Process incoming LiDAR scan"""
        # Calculate angles for each range reading
        angles = []
        for i in range(len(msg.ranges)):
            angle = msg.angle_min + i * msg.angle_increment
            angles.append(angle)

        # Convert to Cartesian coordinates
        points = []
        for i, (angle, range_val) in enumerate(zip(angles, msg.ranges)):
            if msg.range_min <= range_val <= msg.range_max:
                x = range_val * cos(angle)
                y = range_val * sin(angle)
                points.append((x, y))

        # Analyze the scan data
        self.analyze_scan(points, msg)

    def analyze_scan(self, points, scan_msg):
        """Analyze LiDAR scan data"""
        if not points:
            self.get_logger().info('No valid range readings in scan')
            return

        # Calculate some statistics
        distances = [np.sqrt(x*x + y*y) for x, y in points]
        avg_distance = np.mean(distances) if distances else 0
        min_distance = np.min(distances) if distances else float('inf')

        # Check for obstacles in front of robot
        front_readings = []
        front_angle_range = 0.5  # 0.5 radians (~29 degrees) in front

        for i, angle in enumerate(scan_msg.angle_min + np.arange(len(scan_msg.ranges)) * scan_msg.angle_increment):
            if abs(angle) < front_angle_range:
                if scan_msg.range_min <= scan_msg.ranges[i] <= scan_msg.range_max:
                    front_readings.append(scan_msg.ranges[i])

        if front_readings:
            min_front_dist = min(front_readings)
            if min_front_dist < 1.0:  # Obstacle within 1 meter
                self.get_logger().warn(f'Obstacle detected {min_front_dist:.2f}m ahead!')

        self.get_logger().info(
            f'Scan analysis: Points={len(points)}, Avg dist={avg_distance:.2f}m, '
            f'Min dist={min_distance:.2f}m, Min front dist={min_front_dist:.2f}m'
        )

def main(args=None):
    rclpy.init(args=args)
    node = LidarSubscriber()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down LiDAR subscriber')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### IMU Integration

#### IMU Publisher Example

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3, Quaternion
import numpy as np
from math import sin, cos

class ImuPublisher(Node):
    def __init__(self):
        super().__init__('imu_publisher')

        # Publisher for IMU data
        self.imu_publisher = self.create_publisher(Imu, 'imu/data', 10)

        # Timer for publishing IMU data
        self.timer = self.create_timer(0.01, self.publish_imu)  # 100 Hz

        # Simulate IMU motion
        self.time = 0.0
        self.orientation = [0.0, 0.0, 0.0, 1.0]  # [x, y, z, w] quaternion
        self.angular_velocity = [0.1, 0.05, 0.02]  # [x, y, z] rad/s
        self.linear_acceleration = [0.0, 0.0, 9.81]  # [x, y, z] m/s²

        self.get_logger().info('IMU publisher initialized')

    def publish_imu(self):
        """Publish IMU data"""
        imu_msg = Imu()
        imu_msg.header.stamp = self.get_clock().now().to_msg()
        imu_msg.header.frame_id = 'imu_link'

        # Simulate changing orientation (simple rotation)
        self.time += 0.01  # 100 Hz
        rotation_angle = 0.1 * sin(self.time * 2)  # Oscillating rotation

        # Convert to quaternion (simplified - in practice use proper conversion)
        # For small rotations around z-axis: [0, 0, sin(θ/2), cos(θ/2)]
        half_angle = rotation_angle / 2.0
        self.orientation = [0.0, 0.0, sin(half_angle), cos(half_angle)]

        # Set orientation
        imu_msg.orientation = Quaternion(
            x=self.orientation[0],
            y=self.orientation[1],
            z=self.orientation[2],
            w=self.orientation[3]
        )

        # Add noise to orientation
        imu_msg.orientation_covariance = [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01]

        # Set angular velocity (with some noise)
        imu_msg.angular_velocity = Vector3(
            x=self.angular_velocity[0] + np.random.normal(0, 0.01),
            y=self.angular_velocity[1] + np.random.normal(0, 0.01),
            z=self.angular_velocity[2] + np.random.normal(0, 0.01)
        )

        # Set linear acceleration (with gravity and some motion)
        imu_msg.linear_acceleration = Vector3(
            x=self.linear_acceleration[0] + 0.1 * sin(self.time * 5) + np.random.normal(0, 0.05),
            y=self.linear_acceleration[1] + 0.1 * cos(self.time * 3) + np.random.normal(0, 0.05),
            z=self.linear_acceleration[2] + np.random.normal(0, 0.05)
        )

        # Publish IMU message
        self.imu_publisher.publish(imu_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ImuPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down IMU publisher')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Sensor Data Processing

### Data Preprocessing

Raw sensor data often needs preprocessing before use:

```python
import numpy as np
from scipy import ndimage
from scipy.spatial import KDTree

class SensorDataProcessor:
    """Process and clean sensor data"""

    def __init__(self):
        self.noise_threshold = 0.1
        self.outlier_threshold = 2.0

    def preprocess_lidar_data(self, scan_msg):
        """Clean and preprocess LiDAR scan data"""
        ranges = np.array(scan_msg.ranges)
        angles = np.array([scan_msg.angle_min + i * scan_msg.angle_increment
                          for i in range(len(ranges))])

        # Remove invalid readings
        valid_mask = (ranges >= scan_msg.range_min) & (ranges <= scan_msg.range_max)
        valid_ranges = ranges[valid_mask]
        valid_angles = angles[valid_mask]

        # Convert to Cartesian coordinates
        x = valid_ranges * np.cos(valid_angles)
        y = valid_ranges * np.sin(valid_angles)
        points = np.column_stack((x, y))

        # Remove outliers using statistical method
        cleaned_points = self.remove_outliers(points)

        return cleaned_points

    def preprocess_camera_data(self, image_msg, bridge):
        """Preprocess camera image data"""
        # Convert ROS image to OpenCV
        cv_image = bridge.imgmsg_to_cv2(image_msg, 'bgr8')

        # Apply noise reduction
        denoised_image = cv2.fastNlMeansDenoisingColored(cv_image, None, 10, 10, 7, 21)

        # Apply histogram equalization for better contrast
        # Convert to YUV for histogram equalization
        yuv_image = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2YUV)
        yuv_image[:,:,0] = cv2.equalizeHist(yuv_image[:,:,0])
        enhanced_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)

        return enhanced_image

    def remove_outliers(self, points, method='statistical'):
        """Remove outliers from point cloud data"""
        if method == 'statistical':
            # Statistical outlier removal
            kdtree = KDTree(points)
            distances, _ = kdtree.query(points, k=10)  # 10 nearest neighbors

            # Calculate mean distance to neighbors
            mean_distances = np.mean(distances[:, 1:], axis=1)  # Exclude self-distance

            # Remove points with high mean distance to neighbors
            threshold = np.mean(mean_distances) + self.outlier_threshold * np.std(mean_distances)
            valid_mask = mean_distances < threshold

            return points[valid_mask]

        elif method == 'radius':
            # Radius-based outlier removal
            kdtree = KDTree(points)
            indices = kdtree.query_ball_point(points, r=0.5)
            valid_mask = [len(idx) > 2 for idx in indices]  # Need at least 3 neighbors

            return points[valid_mask]

        return points

    def filter_imu_data(self, imu_msg, filter_type='moving_average', window_size=5):
        """Filter IMU data to reduce noise"""
        if filter_type == 'moving_average':
            # Simple moving average filter
            if not hasattr(self, 'imu_buffer'):
                self.imu_buffer = {'angular_velocity': [], 'linear_acceleration': []}

            # Add new data to buffers
            self.imu_buffer['angular_velocity'].append([
                imu_msg.angular_velocity.x,
                imu_msg.angular_velocity.y,
                imu_msg.angular_velocity.z
            ])

            self.imu_buffer['linear_acceleration'].append([
                imu_msg.linear_acceleration.x,
                imu_msg.linear_acceleration.y,
                imu_msg.linear_acceleration.z
            ])

            # Keep only recent values
            if len(self.imu_buffer['angular_velocity']) > window_size:
                self.imu_buffer['angular_velocity'].pop(0)
                self.imu_buffer['linear_acceleration'].pop(0)

            # Calculate averages if we have enough data
            if len(self.imu_buffer['angular_velocity']) >= window_size:
                avg_ang_vel = np.mean(self.imu_buffer['angular_velocity'], axis=0)
                avg_lin_acc = np.mean(self.imu_buffer['linear_acceleration'], axis=0)

                filtered_imu = Imu()
                filtered_imu.orientation = imu_msg.orientation
                filtered_imu.angular_velocity = Vector3(
                    x=avg_ang_vel[0], y=avg_ang_vel[1], z=avg_ang_vel[2]
                )
                filtered_imu.linear_acceleration = Vector3(
                    x=avg_lin_acc[0], y=avg_lin_acc[1], z=avg_lin_acc[2]
                )

                return filtered_imu

        return imu_msg
```

### Multi-Sensor Data Synchronization

Synchronizing data from multiple sensors is crucial:

```python
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from message_filters import ApproximateTimeSynchronizer, Subscriber
import threading

class MultiSensorSynchronizer:
    """Synchronize data from multiple sensors"""

    def __init__(self, node):
        self.node = node
        self.lock = threading.Lock()

        # Create subscribers for different sensors
        qos_profile = QoSProfile(depth=10)

        self.camera_sub = Subscriber(
            node, Image, 'camera/image_raw', qos_profile=qos_profile
        )
        self.lidar_sub = Subscriber(
            node, LaserScan, 'scan', qos_profile=qos_profile
        )
        self.imu_sub = Subscriber(
            node, Imu, 'imu/data', qos_profile=qos_profile
        )

        # Synchronize messages (approximate time synchronization)
        self.sync = ApproximateTimeSynchronizer(
            [self.camera_sub, self.lidar_sub, self.imu_sub],
            queue_size=10,
            slop=0.1  # 100ms tolerance
        )
        self.sync.registerCallback(self.multi_sensor_callback)

        self.processed_count = 0

    def multi_sensor_callback(self, camera_msg, lidar_msg, imu_msg):
        """Process synchronized sensor data"""
        with self.lock:
            self.processed_count += 1

            # Process synchronized data
            self.fuse_sensor_data(camera_msg, lidar_msg, imu_msg)

            self.node.get_logger().info(
                f'Processed synchronized data batch #{self.processed_count}'
            )

    def fuse_sensor_data(self, camera_msg, lidar_msg, imu_msg):
        """Fuse data from multiple sensors"""
        # Example: Combine camera image with LiDAR points projected to image
        # This would typically involve:
        # 1. Converting LiDAR points to camera frame
        # 2. Projecting 3D points to 2D image coordinates
        # 3. Combining with image data for enhanced perception

        # For now, just log that we have synchronized data
        self.node.get_logger().info(
            f'Synchronized: Camera({camera_msg.width}x{camera_msg.height}), '
            f'LiDAR({len(lidar_msg.ranges)} pts), '
            f'IMU(orient: {imu_msg.orientation.x:.3f})'
        )
```

## Sensor Integration with URDF

### Adding Sensors to Robot Description

Sensors need to be properly defined in your robot's URDF:

```xml
<?xml version="1.0"?>
<robot name="sensor_integration_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder radius="0.2" length="0.1"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.2" length="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.02"/>
    </inertial>
  </link>

  <!-- Camera link -->
  <link name="camera_link">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>

  <!-- Camera joint -->
  <joint name="camera_joint" type="fixed">
    <parent link="base_link"/>
    <child link="camera_link"/>
    <origin xyz="0.1 0 0.1" rpy="0 0 0"/>
  </joint>

  <!-- LiDAR link -->
  <link name="laser_link">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.03"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.03"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.0002" ixy="0" ixz="0" iyy="0.0002" iyz="0" izz="0.0004"/>
    </inertial>
  </link>

  <!-- LiDAR joint -->
  <joint name="laser_joint" type="fixed">
    <parent link="base_link"/>
    <child link="laser_link"/>
    <origin xyz="0 0 0.15" rpy="0 0 0"/>
  </joint>

  <!-- IMU link -->
  <link name="imu_link">
    <inertial>
      <mass value="0.01"/>
      <inertia ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6"/>
    </inertial>
  </link>

  <!-- IMU joint -->
  <joint name="imu_joint" type="fixed">
    <parent link="base_link"/>
    <child link="imu_link"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
  </joint>

  <!-- Gazebo plugins for sensors -->
  <gazebo reference="camera_link">
    <sensor name="camera" type="camera">
      <update_rate>30</update_rate>
      <camera name="head">
        <horizontal_fov>1.3962634</horizontal_fov> <!-- 80 degrees -->
        <image>
          <width>640</width>
          <height>480</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.1</near>
          <far>300</far>
        </clip>
      </camera>
      <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
        <frame_name>camera_link</frame_name>
        <min_depth>0.1</min_depth>
        <max_depth>300.0</max_depth>
      </plugin>
    </sensor>
  </gazebo>

  <gazebo reference="laser_link">
    <sensor name="laser" type="ray">
      <ray>
        <scan>
          <horizontal>
            <samples>720</samples>
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
      <plugin name="laser_controller" filename="libgazebo_ros_laser.so">
        <frame_name>laser_link</frame_name>
      </plugin>
    </sensor>
  </gazebo>

  <gazebo reference="imu_link">
    <sensor name="imu" type="imu">
      <always_on>true</always_on>
      <update_rate>100</update_rate>
      <plugin name="imu_controller" filename="libgazebo_ros_imu.so">
        <frame_name>imu_link</frame_name>
      </plugin>
    </sensor>
  </gazebo>
</robot>
```

## Sensor Calibration

### Camera Calibration

Camera calibration is essential for accurate computer vision:

```python
import cv2
import numpy as np
from cv2 import aruco

class CameraCalibrator:
    """Calibrate camera intrinsic and extrinsic parameters"""

    def __init__(self):
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None

    def calibrate_camera_chessboard(self, images, board_size=(9, 6), square_size=0.025):
        """Calibrate camera using chessboard pattern"""
        # Prepare object points (3D points in real world)
        objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * square_size

        # Arrays to store object points and image points
        obj_points = []  # 3D points in real world
        img_points = []  # 2D points in image plane

        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, board_size, None)

            if ret:
                obj_points.append(objp)
                # Improve corner accuracy
                refined_corners = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1),
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                )
                img_points.append(refined_corners)

        if len(obj_points) > 0:
            # Calibrate camera
            ret, self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs = cv2.calibrateCamera(
                obj_points, img_points, gray.shape[::-1], None, None
            )

            return ret, self.camera_matrix, self.dist_coeffs
        else:
            return False, None, None

    def calibrate_camera_aruco(self, images, aruco_dict=cv2.aruco.DICT_6X6_250, marker_size=0.05):
        """Calibrate camera using ArUco markers"""
        dictionary = cv2.aruco.Dictionary_get(aruco_dict)
        parameters = cv2.aruco.DetectorParameters_create()

        all_ids = []
        all_corners = []

        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect markers
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)

            if ids is not None:
                all_corners.extend(corners)
                all_ids.extend(ids.flatten())

        if len(all_ids) > 0:
            # Estimate pose of each marker
            marker_length = marker_size
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                all_corners, marker_length, self.camera_matrix, self.dist_coeffs
            )

            # Calibrate using marker poses
            # This is a simplified approach - full calibration would need more complex processing
            return True, self.camera_matrix, self.dist_coeffs

        return False, None, None

    def undistort_image(self, img):
        """Undistort image using calibration parameters"""
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            h, w = img.shape[:2]
            new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
            )

            undistorted = cv2.undistort(img, self.camera_matrix, self.dist_coeffs, None, new_camera_mtx)

            # Crop the image if needed
            x, y, w, h = roi
            if roi is not None:
                undistorted = undistorted[y:y+h, x:x+w]

            return undistorted

        return img

# Example usage
def calibrate_robot_camera():
    """Example of calibrating a robot's camera"""
    calibrator = CameraCalibrator()

    # Load calibration images (would come from robot's camera)
    # images = load_calibration_images_from_robot()

    # Calibrate using chessboard pattern
    # success, camera_matrix, dist_coeffs = calibrator.calibrate_camera_chessboard(images)

    # Use calibration for undistorting images
    # undistorted_img = calibrator.undistort_image(raw_image)

    print("Camera calibration example completed")
```

### LiDAR Calibration

LiDAR calibration involves aligning the sensor with the robot's coordinate system:

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

class LidarCalibrator:
    """Calibrate LiDAR sensor position and orientation"""

    def __init__(self):
        # LiDAR extrinsic calibration: position and orientation relative to base_link
        self.lidar_to_base = self.create_transform(
            translation=[0.0, 0.0, 0.15],  # 15cm above base
            rotation=[0, 0, 0]  # No rotation initially
        )

    def create_transform(self, translation, rotation):
        """Create a 4x4 transformation matrix"""
        # Create rotation matrix from Euler angles (roll, pitch, yaw)
        r = R.from_euler('xyz', rotation, degrees=False)
        rot_matrix = r.as_matrix()

        # Create 4x4 transformation matrix
        transform = np.eye(4)
        transform[0:3, 0:3] = rot_matrix
        transform[0:3, 3] = translation

        return transform

    def calibrate_lidar_to_camera(self, lidar_points, camera_image, correspondences):
        """
        Calibrate LiDAR to camera extrinsics using known correspondences
        This is a simplified approach - real calibration would be more complex
        """
        # This would typically involve:
        # 1. Detecting common features in both modalities
        # 2. Computing the transformation that minimizes reprojection error
        # 3. Using optimization techniques like ICP (Iterative Closest Point)

        # For now, return a sample transformation
        return self.lidar_to_base

    def transform_lidar_points(self, points, transform_matrix=None):
        """Transform LiDAR points to robot coordinate frame"""
        if transform_matrix is None:
            transform_matrix = self.lidar_to_base

        # Convert points to homogeneous coordinates
        points_homogeneous = np.hstack([
            points, np.ones((points.shape[0], 1))
        ])

        # Apply transformation
        transformed_points = (transform_matrix @ points_homogeneous.T).T

        # Return 3D coordinates
        return transformed_points[:, :3]

    def project_lidar_to_image(self, lidar_points, camera_matrix, dist_coeffs):
        """Project LiDAR points onto camera image"""
        # Transform points to camera frame (assuming camera and LiDAR are calibrated)
        # This would use the extrinsic calibration between sensors

        # Project 3D points to 2D image coordinates
        points_2d, _ = cv2.projectPoints(
            lidar_points.reshape(-1, 1, 3),
            np.zeros(3),  # rvec (would come from calibration)
            np.zeros(3),  # tvec (would come from calibration)
            camera_matrix,
            dist_coeffs
        )

        return points_2d.reshape(-1, 2)
```

## Sensor Quality Assessment

### Data Quality Metrics

Monitor sensor data quality to ensure reliable operation:

```python
import statistics
from collections import deque

class SensorQualityAssessor:
    """Assess quality of sensor data streams"""

    def __init__(self, window_size=100):
        self.window_size = window_size
        self.metrics = {
            'camera': {
                'brightness_history': deque(maxlen=window_size),
                'sharpness_history': deque(maxlen=window_size),
                'frame_rate_history': deque(maxlen=window_size)
            },
            'lidar': {
                'range_validity_ratio': deque(maxlen=window_size),
                'point_density': deque(maxlen=window_size),
                'intensity_variance': deque(maxlen=window_size)
            },
            'imu': {
                'acceleration_norm': deque(maxlen=window_size),
                'angular_velocity_norm': deque(maxlen=window_size),
                'bias_drift': deque(maxlen=window_size)
            }
        }

    def assess_camera_quality(self, image_msg, bridge):
        """Assess camera data quality"""
        try:
            cv_image = bridge.imgmsg_to_cv2(image_msg, 'bgr8')

            # Calculate brightness (average pixel intensity)
            brightness = np.mean(cv_image)
            self.metrics['camera']['brightness_history'].append(brightness)

            # Calculate sharpness (Laplacian variance)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            self.metrics['camera']['sharpness_history'].append(laplacian_var)

            # Check if image is too dark or blurry
            quality_issues = []
            if brightness < 50:  # Too dark
                quality_issues.append("Image too dark")
            if laplacian_var < 100:  # Too blurry
                quality_issues.append("Image too blurry")

            return {
                'brightness': brightness,
                'sharpness': laplacian_var,
                'quality_issues': quality_issues,
                'timestamp': image_msg.header.stamp
            }

        except Exception as e:
            return {'error': str(e)}

    def assess_lidar_quality(self, scan_msg):
        """Assess LiDAR data quality"""
        # Calculate ratio of valid to invalid readings
        valid_ranges = [r for r in scan_msg.ranges if scan_msg.range_min <= r <= scan_msg.range_max]
        validity_ratio = len(valid_ranges) / len(scan_msg.ranges) if scan_msg.ranges else 0

        self.metrics['lidar']['range_validity_ratio'].append(validity_ratio)

        # Calculate average intensity if available
        if len(scan_msg.intensities) > 0:
            intensity_variance = np.var(scan_msg.intensities)
            self.metrics['lidar']['intensity_variance'].append(intensity_variance)

        # Quality assessment
        quality_issues = []
        if validity_ratio < 0.5:  # More than 50% invalid readings
            quality_issues.append("High invalid reading ratio")
        if len(valid_ranges) < 10:  # Too few valid readings
            quality_issues.append("Too few valid readings")

        return {
            'validity_ratio': validity_ratio,
            'total_readings': len(scan_msg.ranges),
            'valid_readings': len(valid_ranges),
            'quality_issues': quality_issues,
            'timestamp': scan_msg.header.stamp
        }

    def assess_imu_quality(self, imu_msg):
        """Assess IMU data quality"""
        # Calculate magnitude of acceleration (should be ~9.81 m/s² when stationary)
        acc_mag = np.sqrt(
            imu_msg.linear_acceleration.x**2 +
            imu_msg.linear_acceleration.y**2 +
            imu_msg.linear_acceleration.z**2
        )
        self.metrics['imu']['acceleration_norm'].append(acc_mag)

        # Calculate magnitude of angular velocity
        ang_vel_mag = np.sqrt(
            imu_msg.angular_velocity.x**2 +
            imu_msg.angular_velocity.y**2 +
            imu_msg.angular_velocity.z**2
        )
        self.metrics['imu']['angular_velocity_norm'].append(ang_vel_mag)

        # Quality assessment
        quality_issues = []
        if abs(acc_mag - 9.81) > 2.0:  # Acceleration far from gravity
            quality_issues.append("Acceleration magnitude unusual")
        if ang_vel_mag > 5.0:  # High angular velocity (might be spinning)
            quality_issues.append("High angular velocity detected")

        return {
            'acceleration_norm': acc_mag,
            'angular_velocity_norm': ang_vel_mag,
            'orientation': imu_msg.orientation,
            'quality_issues': quality_issues,
            'timestamp': imu_msg.header.stamp
        }

    def get_overall_quality_report(self):
        """Get overall quality metrics for all sensors"""
        report = {}

        for sensor_type, metrics in self.metrics.items():
            report[sensor_type] = {}
            for metric_name, history in metrics.items():
                if history:
                    report[sensor_type][metric_name] = {
                        'current': history[-1],
                        'mean': np.mean(history),
                        'std': np.std(history),
                        'min': min(history),
                        'max': max(history)
                    }

        return report

# Example usage
def monitor_sensor_quality():
    """Example of monitoring sensor quality"""
    quality_assessor = SensorQualityAssessor(window_size=50)

    # This would be called in sensor callbacks
    # camera_quality = quality_assessor.assess_camera_quality(camera_msg, bridge)
    # lidar_quality = quality_assessor.assess_lidar_quality(lidar_msg)
    # imu_quality = quality_assessor.assess_imu_quality(imu_msg)

    # Get overall report
    # quality_report = quality_assessor.get_overall_quality_report()

    print("Sensor quality monitoring example completed")
```

## Troubleshooting Common Issues

### Sensor Data Issues

#### Data Synchronization Problems

```python
def troubleshoot_sensor_sync():
    """Diagnose sensor synchronization issues"""
    print("Sensor Synchronization Troubleshooting:")
    print("1. Check timestamps - ensure sensors have synchronized clocks")
    print("2. Verify QoS settings match between publishers and subscribers")
    print("3. Check network bandwidth if using remote sensors")
    print("4. Monitor message rates to identify bottlenecks")
    print("5. Use 'ros2 topic hz' to check topic frequencies")
    print("6. Consider using message_filters for temporal alignment")
```

#### Calibration Problems

```python
def troubleshoot_calibration():
    """Diagnose sensor calibration issues"""
    print("Sensor Calibration Troubleshooting:")
    print("1. Verify sensor mounting is secure and hasn't shifted")
    print("2. Check that calibration patterns are visible and well-lit")
    print("3. Ensure sufficient feature points are visible during calibration")
    print("4. Verify coordinate frame relationships in TF tree")
    print("5. Test calibration with known objects of known size")
    print("6. Check for lens distortion in camera systems")
```

## Best Practices

### 1. Sensor Integration

- **Standardized Interfaces**: Use ROS 2 standard message types
- **Error Handling**: Implement robust error handling for sensor failures
- **Redundancy**: Use multiple sensors when reliability is critical
- **Validation**: Continuously validate sensor data quality

### 2. Data Processing

- **Real-time Processing**: Optimize algorithms for real-time performance
- **Memory Management**: Efficiently handle large data streams
- **Filtering**: Apply appropriate filters to reduce noise
- **Synchronization**: Properly synchronize multi-sensor data

### 3. Quality Assurance

- **Continuous Monitoring**: Monitor sensor health continuously
- **Calibration Maintenance**: Regularly verify and update calibrations
- **Environmental Adaptation**: Adjust parameters for changing conditions
- **Fault Detection**: Implement automatic fault detection and recovery

### 4. Performance Optimization

- **Computational Efficiency**: Optimize algorithms for the target platform
- **Bandwidth Management**: Compress data when necessary
- **Multi-threading**: Use appropriate threading models
- **Hardware Acceleration**: Leverage GPU/TPU when available

## Next Steps

After mastering sensor integration:

1. Continue to [Camera Setup and Calibration](./camera-setup.md) to learn about computer vision sensors
2. Practice integrating different sensor types with your robot
3. Implement sensor fusion techniques for enhanced perception
4. Test sensor systems in both simulation and real environments

Your understanding of sensor integration is now foundational for creating perceptive robots in the Physical AI and Humanoid Robotics course!
---
sidebar_position: 3
---

# Camera Setup and Calibration

This guide covers the fundamentals of camera setup and calibration for robotics applications. Proper camera configuration and calibration are essential for accurate computer vision and perception tasks.

## Overview

Camera setup and calibration is a critical step in robotics perception. A well-calibrated camera system enables:

- **Accurate measurements**: Real-world distances from image pixels
- **Reliable feature detection**: Consistent identification of objects and landmarks
- **3D reconstruction**: Conversion of 2D images to 3D spatial information
- **Sensor fusion**: Integration with other sensors like LiDAR
- **Navigation**: Visual odometry and SLAM capabilities

### Key Camera Parameters

#### Intrinsic Parameters
- **Focal Length**: Distance from optical center to image plane (fx, fy in pixels)
- **Principal Point**: Optical center in image coordinates (cx, cy)
- **Skew Coefficient**: Pixel axis alignment (usually 0)
- **Distortion Coefficients**: Corrections for lens distortions (k1, k2, p1, p2, k3)

#### Extrinsic Parameters
- **Rotation Matrix**: Orientation of camera relative to robot
- **Translation Vector**: Position of camera relative to robot

## Camera Types and Selection

### RGB Cameras

RGB cameras capture color images and are the most common type for robotics:

#### Key Specifications
- **Resolution**: Image dimensions (e.g., 640×480, 1280×720, 1920×1080)
- **Frame Rate**: Captures per second (e.g., 30, 60, 120 FPS)
- **Field of View**: Angular coverage (e.g., 60°, 90°, 120°)
- **Lens Type**: Fixed or adjustable focal length
- **Mounting**: C-mount, CS-mount, or proprietary mounts

#### Common Formats
- **USB Cameras**: Easy integration with computers
- **GigE Cameras**: Long-distance transmission capabilities
- **CSI Cameras**: Direct connection to embedded systems (like Raspberry Pi)
- **Ethernet Cameras**: Network-based streaming

### Depth Cameras

Depth cameras provide 3D information along with color:

#### Types
- **Stereo Cameras**: Two cameras for triangulation
- **Structured Light**: Projected patterns for depth calculation
- **Time-of-Flight (ToF)**: Measure light travel time

#### Key Specifications
- **Depth Range**: Minimum and maximum measurable distances
- **Depth Accuracy**: Precision of depth measurements
- **Depth Resolution**: Number of depth pixels
- **IR Capability**: Infrared operation for low-light conditions

### Thermal Cameras

Thermal cameras detect heat signatures:

#### Key Specifications
- **Spectral Range**: Wavelength sensitivity (typically 8-14 μm)
- **Thermal Sensitivity**: Minimum detectable temperature difference
- **Temperature Range**: Operational temperature range
- **Spatial Resolution**: Thermal pixel resolution

## Camera Hardware Setup

### Mounting Considerations

Proper camera mounting affects both image quality and usability:

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

class CameraMounting:
    """Calculate optimal camera mounting parameters"""

    def __init__(self):
        self.mounting_parameters = {
            'height': 1.0,  # meters above ground
            'pitch': 0.0,   # degrees from horizontal
            'yaw': 0.0,     # degrees from forward
            'roll': 0.0     # degrees from level
        }

    def calculate_field_of_view_coverage(self, camera_fov, mounting_height, pitch_angle):
        """
        Calculate ground coverage based on mounting parameters
        """
        fov_half = np.radians(camera_fov / 2)
        pitch_rad = np.radians(pitch_angle)

        # Calculate distances to ground
        near_distance = mounting_height / np.tan(pitch_rad + fov_half)
        far_distance = mounting_height / np.tan(pitch_rad - fov_half)

        # Calculate width coverage
        width = 2 * mounting_height * np.tan(fov_half)

        return {
            'near_distance': near_distance,
            'far_distance': far_distance,
            'width_coverage': width,
            'area_coverage': width * (far_distance - near_distance)
        }

    def optimize_mounting_for_task(self, task_requirements):
        """
        Optimize camera mounting based on task requirements
        """
        # Example: Optimize for navigation
        if task_requirements.get('task') == 'navigation':
            # Lower mounting for ground obstacle detection
            self.mounting_parameters['height'] = 0.8  # Lower for ground coverage
            self.mounting_parameters['pitch'] = -5    # Downward tilt for ground view

        # Example: Optimize for object detection
        elif task_requirements.get('task') == 'object_detection':
            # Higher mounting for broader view
            self.mounting_parameters['height'] = 1.2  # Higher for broader view
            self.mounting_parameters['pitch'] = 0     # Forward view

        return self.mounting_parameters

# Example usage
def setup_navigation_camera():
    """Set up camera for navigation tasks"""
    mount = CameraMounting()

    # Calculate coverage for navigation camera
    coverage = mount.calculate_field_of_view_coverage(
        camera_fov=70,      # 70 degree FOV
        mounting_height=1.0, # 1 meter height
        pitch_angle=-5      # 5 degrees downward
    )

    print(f"Ground coverage: {coverage['width_coverage']:.2f}m wide × {coverage['far_distance'] - coverage['near_distance']:.2f}m deep")

    # Optimize for navigation
    nav_params = mount.optimize_mounting_for_task({'task': 'navigation'})
    print(f"Optimized mounting: {nav_params}")
```

### Camera Placement Strategy

```python
class CameraPlacementStrategy:
    """Determine optimal camera placement for different scenarios"""

    def __init__(self):
        self.strategies = {
            'front_facing': self.front_facing_placement,
            'surround_view': self.surround_view_placement,
            'top_down': self.top_down_placement,
            'stereo_pair': self.stereo_pair_placement
        }

    def front_facing_placement(self, robot_dimensions):
        """Place camera for forward-facing perception"""
        return {
            'position': [robot_dimensions['length']/2, 0, robot_dimensions['height']/2],
            'orientation': [0, 0, 0],  # Looking straight ahead
            'fov_horizontal': 70,
            'fov_vertical': 50
        }

    def surround_view_placement(self, robot_radius):
        """Place multiple cameras for 360-degree view"""
        cameras = []

        # Place cameras at 90-degree intervals
        for i, angle in enumerate([0, 90, 180, 270]):
            camera = {
                'id': f'camera_{i}',
                'position': [
                    robot_radius * np.cos(np.radians(angle)),
                    robot_radius * np.sin(np.radians(angle)),
                    0.5  # 0.5m above ground
                ],
                'orientation': [0, 0, np.radians(angle)],
                'fov_horizontal': 90,
                'fov_vertical': 60
            }
            cameras.append(camera)

        return cameras

    def stereo_pair_placement(self, baseline_distance=0.12):
        """Place stereo camera pair"""
        return [
            {
                'id': 'left_camera',
                'position': [-baseline_distance/2, 0, 0.5],
                'orientation': [0, 0, 0],
                'fov_horizontal': 60,
                'fov_vertical': 45
            },
            {
                'id': 'right_camera',
                'position': [baseline_distance/2, 0, 0.5],
                'orientation': [0, 0, 0],
                'fov_horizontal': 60,
                'fov_vertical': 45
            }
        ]

# Example usage
def setup_robot_cameras():
    """Set up cameras for a robot platform"""
    placement_strategy = CameraPlacementStrategy()

    robot_dims = {'length': 0.8, 'width': 0.6, 'height': 0.5}

    # Front-facing camera for navigation
    front_cam = placement_strategy.front_facing_placement(robot_dims)
    print(f"Front camera: {front_cam}")

    # Stereo pair for depth perception
    stereo_pair = placement_strategy.stereo_pair_placement(0.12)  # 12cm baseline
    print(f"Stereo pair: {stereo_pair}")
```

## Camera Calibration Process

### Intrinsic Calibration

Intrinsic calibration determines internal camera parameters:

```python
import cv2
import numpy as np
import yaml
from pathlib import Path

class IntrinsicCalibrator:
    """Perform intrinsic camera calibration"""

    def __init__(self, board_size=(9, 6), square_size=0.025):
        """
        Initialize calibrator
        board_size: Number of internal corners (width, height)
        square_size: Size of chessboard squares in meters
        """
        self.board_size = board_size
        self.square_size = square_size

        # Prepare object points (3D points in real world)
        self.objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * square_size

        # Arrays to store object points and image points
        self.obj_points = []  # 3D points in real world space
        self.img_points = []  # 2D points in image plane

    def find_chessboard_corners(self, image):
        """Find chessboard corners in an image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(
            gray, self.board_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        )

        if ret:
            # Improve corner accuracy
            refined_corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )

            return ret, refined_corners, gray
        else:
            return ret, None, gray

    def capture_calibration_images(self, camera_index=0, num_images=20, save_dir='calibration_images'):
        """Capture images for calibration"""
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_index}")

        Path(save_dir).mkdir(exist_ok=True)
        captured_images = []

        print(f"Capturing {num_images} calibration images...")
        print("Press SPACE to capture image, ESC to quit")

        while len(captured_images) < num_images:
            ret, frame = cap.read()
            if not ret:
                break

            # Display frame with instructions
            cv2.putText(frame, f"Captured: {len(captured_images)}/{num_images}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press SPACE to capture, ESC to quit",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Try to find chessboard
            ret, corners, gray = self.find_chessboard_corners(frame)
            if ret:
                # Draw corners
                cv2.drawChessboardCorners(frame, self.board_size, corners, ret)
                cv2.putText(frame, "Chessboard detected!",
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Calibration', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord(' ') and ret:  # SPACE and chessboard detected
                # Save image
                img_path = f"{save_dir}/calib_{len(captured_images):03d}.jpg"
                cv2.imwrite(img_path, frame)

                # Add to calibration points
                self.obj_points.append(self.objp)
                self.img_points.append(corners)
                captured_images.append(img_path)

                print(f"Captured image {len(captured_images)}: {img_path}")

        cap.release()
        cv2.destroyAllWindows()

        print(f"Captured {len(captured_images)} calibration images")
        return captured_images

    def calibrate_from_images(self, image_paths):
        """Calibrate camera from a set of images"""
        for img_path in image_paths:
            img = cv2.imread(img_path)
            ret, corners, gray = self.find_chessboard_corners(img)

            if ret:
                self.obj_points.append(self.objp)
                self.img_points.append(corners)
            else:
                print(f"Could not find chessboard in {img_path}")

        if len(self.obj_points) < 10:
            raise RuntimeError(f"Not enough valid images for calibration. Need at least 10, got {len(self.obj_points)}")

        # Calibrate camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.obj_points, self.img_points, gray.shape[::-1], None, None
        )

        if not ret:
            raise RuntimeError("Camera calibration failed")

        # Calculate reprojection error
        tot_error = 0
        for i in range(len(self.obj_points)):
            img_points2, _ = cv2.projectPoints(
                self.obj_points[i], rvecs[i], tvecs[i], mtx, dist
            )
            error = cv2.norm(self.img_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
            tot_error += error

        mean_error = tot_error / len(self.obj_points)

        calibration_result = {
            'camera_matrix': mtx,
            'distortion_coefficients': dist,
            'rotation_vectors': rvecs,
            'translation_vectors': tvecs,
            'reprojection_error': mean_error,
            'valid_images_used': len(self.obj_points)
        }

        return calibration_result

    def save_calibration(self, calibration_result, filename='camera_calibration.yaml'):
        """Save calibration parameters to file"""
        # Convert numpy arrays to lists for YAML serialization
        data = {
            'image_width': 640,  # You'll need to adjust based on your camera
            'image_height': 480,
            'camera_matrix': {
                'rows': 3,
                'cols': 3,
                'data': calibration_result['camera_matrix'].flatten().tolist()
            },
            'distortion_coefficients': {
                'rows': 1,
                'cols': 5,
                'data': calibration_result['distortion_coefficients'].flatten().tolist()
            },
            'rectification_matrix': {
                'rows': 3,
                'cols': 3,
                'data': [1, 0, 0, 0, 1, 0, 0, 0, 1]  # Identity matrix for monocular
            },
            'projection_matrix': {
                'rows': 3,
                'cols': 4,
                'data': np.hstack([
                    calibration_result['camera_matrix'],
                    np.array([[0], [0], [0]])
                ]).flatten().tolist()
            },
            'avg_reprojection_error': float(calibration_result['reprojection_error'])
        }

        with open(filename, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

        print(f"Calibration saved to {filename}")
        return filename

    def load_calibration(self, filename='camera_calibration.yaml'):
        """Load calibration parameters from file"""
        with open(filename, 'r') as f:
            data = yaml.safe_load(f)

        camera_matrix = np.array(data['camera_matrix']['data']).reshape(3, 3)
        dist_coeffs = np.array(data['distortion_coefficients']['data'])

        return {
            'camera_matrix': camera_matrix,
            'distortion_coefficients': dist_coeffs,
            'image_width': data['image_width'],
            'image_height': data['image_height']
        }

# Example usage
def perform_camera_calibration():
    """Example of performing camera calibration"""
    calibrator = IntrinsicCalibrator(board_size=(9, 6), square_size=0.025)

    # Option 1: Capture new images
    # images = calibrator.capture_calibration_images(num_images=20)

    # Option 2: Use existing images
    # images = ['calibration_images/calib_000.jpg', 'calibration_images/calib_001.jpg', ...]

    # For demonstration, let's assume we have images
    # calibration_result = calibrator.calibrate_from_images(images)
    # calibrator.save_calibration(calibration_result)

    print("Camera calibration example completed")
```

### Extrinsic Calibration

Extrinsic calibration determines the camera's position and orientation relative to the robot:

```python
class ExtrinsicCalibrator:
    """Calibrate camera position relative to robot coordinate frame"""

    def __init__(self):
        self.robot_to_camera = np.eye(4)  # 4x4 transformation matrix

    def calibrate_using_checkerboard(self, checkerboard_size, square_size,
                                   robot_pose, checkerboard_pose):
        """
        Calibrate extrinsic parameters using known checkerboard pose
        """
        # This is a simplified approach - real calibration would use multiple observations
        # and optimization techniques

        # Transform from checkerboard to camera
        camera_to_checkerboard = self.estimate_pose(checkerboard_size, square_size)

        # Robot to checkerboard (known from robot localization)
        robot_to_checkerboard = self.pose_to_transform(robot_pose)

        # Robot to camera = (Robot to checkerboard) * (Checkerboard to camera)
        self.robot_to_camera = robot_to_checkerboard @ np.linalg.inv(camera_to_checkerboard)

        return self.robot_to_camera

    def estimate_pose(self, image, object_points, image_points, camera_matrix, dist_coeffs):
        """Estimate pose of object relative to camera"""
        ret, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)

        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)

        # Create 4x4 transformation matrix
        transform = np.eye(4)
        transform[0:3, 0:3] = R
        transform[0:3, 3] = tvec.flatten()

        return transform

    def pose_to_transform(self, pose):
        """Convert pose (position + orientation) to 4x4 transformation matrix"""
        # pose should contain position (x, y, z) and orientation (quaternion or euler)
        # This is a simplified version
        transform = np.eye(4)
        # Fill in the transformation matrix based on pose
        return transform

    def calibrate_using_april_tags(self, tag_size, tag_poses, camera_poses):
        """
        Calibrate using AprilTag markers with known positions
        """
        # AprilTags provide precise pose information
        # This would involve multiple tag detections and optimization
        pass

    def calibrate_using_hand_eye(self, robot_poses, camera_poses):
        """
        Hand-eye calibration: AX=XB method
        A: robot motion, B: camera motion, X: camera-to-robot transform
        """
        # This is a complex calibration procedure
        # Would use libraries like sciapy for optimization
        pass

# Example of multi-camera calibration
class MultiCameraCalibrator:
    """Calibrate multiple cameras relative to each other"""

    def __init__(self):
        self.camera_pairs = []

    def calibrate_stereo_pair(self, left_images, right_images,
                             left_camera_matrix, right_camera_matrix,
                             left_dist_coeffs, right_dist_coeffs):
        """Calibrate stereo camera pair"""
        # Find chessboard corners in both cameras simultaneously
        obj_points = []
        left_img_points = []
        right_img_points = []

        for left_img, right_img in zip(left_images, right_images):
            ret_left, corners_left, _ = self.find_chessboard_corners(left_img)
            ret_right, corners_right, _ = self.find_chessboard_corners(right_img)

            if ret_left and ret_right:
                obj_points.append(self.objp)
                left_img_points.append(corners_left)
                right_img_points.append(corners_right)

        if len(obj_points) < 10:
            raise RuntimeError("Not enough valid stereo pairs for calibration")

        # Stereo calibration
        ret, camera_matrix_l, dist_l, camera_matrix_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
            obj_points, left_img_points, right_img_points,
            left_camera_matrix, left_dist_coeffs,
            right_camera_matrix, right_dist_coeffs,
            left_img.shape[::-1], flags=cv2.CALIB_USE_INTRINSIC_GUESS
        )

        # Rectification
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            camera_matrix_l, dist_l, camera_matrix_r, dist_r,
            left_img.shape[::-1], R, T
        )

        return {
            'rotation': R,
            'translation': T,
            'essential_matrix': E,
            'fundamental_matrix': F,
            'rectification_transforms': (R1, R2),
            'projection_matrices': (P1, P2),
            'disparity_to_depth': Q
        }
```

## Camera Configuration for ROS 2

### Camera Launch Files

Create launch files for camera configuration:

```python
# camera_launch.py
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Declare launch arguments
    camera_type = DeclareLaunchArgument(
        'camera_type',
        default_value='usb',
        description='Type of camera (usb, realsense, zed, etc.)'
    )

    camera_name = DeclareLaunchArgument(
        'camera_name',
        default_value='camera',
        description='Name of the camera'
    )

    # USB Camera Configuration
    usb_camera_node = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name=[LaunchConfiguration('camera_name'), '_usb'],
        parameters=[
            {
                'video_device': '/dev/video0',
                'framerate': 30.0,
                'image_width': 640,
                'image_height': 480,
                'pixel_format': 'yuyv',
                'camera_name': LaunchConfiguration('camera_name'),
                'camera_info_url': 'package://robot_description/config/camera_calibration.yaml',
                'io_method': 'mmap'
            }
        ],
        remappings=[
            ('image_raw', [LaunchConfiguration('camera_name'), '/image_raw']),
            ('camera_info', [LaunchConfiguration('camera_name'), '/camera_info'])
        ]
    )

    # Image processing nodes
    image_proc_node = Node(
        package='image_proc',
        executable='image_proc',
        name=[LaunchConfiguration('camera_name'), '_proc'],
        remappings=[
            ('image', [LaunchConfiguration('camera_name'), '/image_raw']),
            ('camera_info', [LaunchConfiguration('camera_name'), '/camera_info']),
            ('image_rect', [LaunchConfiguration('camera_name'), '/image_rect']),
            ('image_rect_color', [LaunchConfiguration('camera_name'), '/image_rect_color'])
        ]
    )

    return LaunchDescription([
        camera_type,
        camera_name,
        usb_camera_node,
        image_proc_node
    ])
```

### Camera Info Configuration

Create camera info configuration files:

```yaml
# config/camera_calibration.yaml
image_width: 640
image_height: 480
camera_name: camera
camera_matrix:
  rows: 3
  cols: 3
  data: [525.0, 0.0, 320.0, 0.0, 525.0, 240.0, 0.0, 0.0, 1.0]
distortion_model: plumb_bob
distortion_coefficients:
  rows: 1
  cols: 5
  data: [0.1, -0.2, 0.001, 0.002, 0.0]
rectification_matrix:
  rows: 3
  cols: 3
  data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
projection_matrix:
  rows: 3
  cols: 4
  data: [525.0, 0.0, 320.0, 0.0, 0.0, 525.0, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0]
```

## Camera Data Processing

### Image Preprocessing

```python
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class ImagePreprocessor:
    """Preprocess camera images for computer vision tasks"""

    def __init__(self, camera_info):
        self.bridge = CvBridge()
        self.camera_matrix = np.array(camera_info['camera_matrix']['data']).reshape(3, 3)
        self.dist_coeffs = np.array(camera_info['distortion_coefficients']['data'])

        # Compute undistortion maps
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix, self.dist_coeffs, None, self.camera_matrix,
            (camera_info['image_width'], camera_info['image_height']), cv2.CV_32FC1
        )

    def undistort_image(self, image):
        """Remove lens distortion from image"""
        return cv2.remap(image, self.map1, self.map2, cv2.INTER_LINEAR)

    def enhance_image(self, image):
        """Enhance image quality for better feature detection"""
        # Convert to LAB color space for illumination adjustment
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)

        # Merge channels and convert back to BGR
        enhanced_lab = cv2.merge([l, a, b])
        enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        return enhanced_image

    def normalize_brightness(self, image, target_mean=128):
        """Normalize image brightness"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate current mean
        current_mean = np.mean(gray)

        # Calculate adjustment factor
        factor = target_mean / current_mean

        # Apply normalization
        normalized = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        return normalized

    def filter_noise(self, image, method='bilateral'):
        """Apply noise reduction"""
        if method == 'bilateral':
            # Bilateral filter preserves edges while smoothing
            return cv2.bilateralFilter(image, 9, 75, 75)
        elif method == 'nl_means':
            # Non-local means denoising
            return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        elif method == 'gaussian':
            # Gaussian blur
            return cv2.GaussianBlur(image, (5, 5), 0)
        else:
            return image

    def process_image_pipeline(self, image):
        """Complete image processing pipeline"""
        # 1. Undistort image
        undistorted = self.undistort_image(image)

        # 2. Enhance image
        enhanced = self.enhance_image(undistorted)

        # 3. Normalize brightness
        normalized = self.normalize_brightness(enhanced)

        # 4. Reduce noise
        denoised = self.filter_noise(normalized, method='bilateral')

        return denoised

# Example usage in a ROS 2 node
class CameraProcessorNode:
    def __init__(self):
        self.preprocessor = None
        self.bridge = CvBridge()

    def image_callback(self, msg):
        """Process incoming camera image"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

            # Process image if preprocessor is initialized
            if self.preprocessor:
                processed_image = self.preprocessor.process_image_pipeline(cv_image)

                # Further processing (feature detection, object recognition, etc.)
                self.process_features(processed_image)

        except Exception as e:
            print(f"Error processing image: {e}")

    def process_features(self, image):
        """Process image features"""
        # Example: Detect edges
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Process contours (could be for obstacle detection, etc.)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Filter small contours
                # Calculate bounding box
                x, y, w, h = cv2.boundingRect(contour)

                # Could send to perception pipeline
                self.publish_detection(x, y, w, h, area)

    def publish_detection(self, x, y, w, h, area):
        """Publish detection results"""
        # This would publish to ROS topics
        pass
```

### Real-time Image Processing

```python
import threading
import queue
from collections import deque

class RealTimeImageProcessor:
    """Handle real-time image processing with threading"""

    def __init__(self, camera_info, max_queue_size=10):
        self.preprocessor = ImagePreprocessor(camera_info)
        self.input_queue = queue.Queue(maxsize=max_queue_size)
        self.output_queue = queue.Queue(maxsize=max_queue_size)

        # Processing statistics
        self.processing_times = deque(maxlen=100)
        self.fps_history = deque(maxlen=100)

        # Threading
        self.processing_thread = threading.Thread(target=self.process_loop, daemon=True)
        self.running = True

        # Start processing thread
        self.processing_thread.start()

    def add_image(self, image):
        """Add image to processing queue"""
        try:
            self.input_queue.put_nowait(image)
        except queue.Full:
            # Drop oldest image if queue is full
            try:
                self.input_queue.get_nowait()
                self.input_queue.put_nowait(image)
            except queue.Empty:
                pass  # Queue is empty, just add the new image

    def get_processed_image(self):
        """Get processed image from output queue"""
        try:
            return self.output_queue.get_nowait()
        except queue.Empty:
            return None

    def process_loop(self):
        """Main processing loop running in separate thread"""
        while self.running:
            try:
                # Get image from input queue (with timeout to allow graceful shutdown)
                image = self.input_queue.get(timeout=0.1)

                # Process image
                start_time = cv2.getTickCount()
                processed = self.preprocessor.process_image_pipeline(image)
                end_time = cv2.getTickCount()

                # Calculate processing time
                processing_time = (end_time - start_time) / cv2.getTickFrequency()
                self.processing_times.append(processing_time)

                # Add processed image to output queue
                try:
                    self.output_queue.put_nowait(processed)
                except queue.Full:
                    # Drop oldest processed image
                    try:
                        self.output_queue.get_nowait()
                        self.output_queue.put_nowait(processed)
                    except queue.Empty:
                        self.output_queue.put_nowait(processed)

            except queue.Empty:
                continue  # Timeout occurred, continue loop
            except Exception as e:
                print(f"Error in processing loop: {e}")
                continue

    def get_performance_stats(self):
        """Get processing performance statistics"""
        if self.processing_times:
            avg_processing_time = sum(self.processing_times) / len(self.processing_times)
            avg_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
        else:
            avg_processing_time = 0
            avg_fps = 0

        return {
            'avg_processing_time': avg_processing_time,
            'avg_fps': avg_fps,
            'queue_size': self.input_queue.qsize(),
            'processed_count': len(self.processing_times)
        }

    def stop(self):
        """Stop the processing thread"""
        self.running = False
        self.processing_thread.join()
```

## Camera Integration with Simulation

### Gazebo Camera Integration

Add camera sensors to your robot's URDF:

```xml
<?xml version="1.0"?>
<robot name="camera_integration_robot">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder radius="0.2" length="0.1"/>
      </geometry>
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

  <!-- Gazebo camera plugin -->
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
        <noise>
          <type>gaussian</type>
          <mean>0.0</mean>
          <stddev>0.007</stddev>
        </noise>
      </camera>
      <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
        <frame_name>camera_link</frame_name>
        <min_depth>0.1</min_depth>
        <max_depth>300.0</max_depth>
        <update_rate>30.0</update_rate>
      </plugin>
    </sensor>
  </gazebo>
</robot>
```

### Isaac Sim Camera Integration

For Isaac Sim, create camera components in USD:

```python
# Isaac Sim camera setup
import omni
from pxr import UsdGeom, Gf, Sdf

def add_camera_to_robot(robot_prim_path, camera_config):
    """Add a camera to robot in Isaac Sim"""
    stage = omni.usd.get_context().get_stage()

    # Create camera prim
    camera_path = f"{robot_prim_path}/{camera_config['name']}"
    camera_prim = UsdGeom.Camera.Define(stage, camera_path)

    # Set camera properties
    camera_prim.GetHorizontalApertureAttr().Set(camera_config['horizontal_aperture'])
    camera_prim.GetVerticalApertureAttr().Set(camera_config['vertical_aperture'])
    camera_prim.GetFocalLengthAttr().Set(camera_config['focal_length'])
    camera_prim.GetNearClippingRangeAttr().Set(camera_config['near_clip'])
    camera_prim.GetFarClippingRangeAttr().Set(camera_config['far_clip'])

    # Set position and orientation
    xform = UsdGeom.Xformable(camera_prim)
    xform.AddTranslateOp().Set(Gf.Vec3d(*camera_config['position']))
    xform.AddRotateXYZOp().Set(Gf.Vec3d(*camera_config['rotation']))

    return camera_prim

# Example camera configuration
def setup_isaac_sim_camera():
    """Set up camera for Isaac Sim"""
    camera_config = {
        'name': 'front_camera',
        'position': [0.1, 0.0, 0.1],  # 10cm forward, 10cm up
        'rotation': [0.0, 0.0, 0.0],  # Looking forward
        'horizontal_aperture': 20.955,  # 640px @ 525px focal length
        'vertical_aperture': 15.716,    # 480px @ 525px focal length
        'focal_length': 525.0,
        'near_clip': 0.1,
        'far_clip': 300.0
    }

    # This would be called with actual robot prim path
    # camera_prim = add_camera_to_robot('/World/Robot', camera_config)

    return camera_config
```

## Quality Assurance and Validation

### Calibration Validation

```python
class CalibrationValidator:
    """Validate camera calibration quality"""

    def __init__(self, camera_matrix, dist_coeffs):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

    def validate_reprojection_accuracy(self, obj_points, img_points, rvec, tvec):
        """Validate calibration using reprojection error"""
        # Project 3D points to 2D
        projected_points, _ = cv2.projectPoints(
            obj_points, rvec, tvec, self.camera_matrix, self.dist_coeffs
        )

        # Calculate reprojection error
        errors = []
        for i in range(len(img_points)):
            error = cv2.norm(img_points[i], projected_points[i], cv2.NORM_L2)
            errors.append(error)

        mean_error = np.mean(errors)
        std_error = np.std(errors)

        return {
            'mean_error': mean_error,
            'std_error': std_error,
            'max_error': max(errors),
            'min_error': min(errors),
            'errors': errors
        }

    def validate_calibration_grid(self, image_shape, num_points=10):
        """Validate calibration across image field"""
        # Create grid of points across image
        h, w = image_shape[:2]

        # Sample points across the image
        y_coords = np.linspace(0, h-1, num_points)
        x_coords = np.linspace(0, w-1, num_points)

        validation_results = []

        for y in y_coords:
            for x in x_coords:
                # This would test the calibration at each point
                # by checking if rectified coordinates are consistent
                pass

        return validation_results

    def check_calibration_consistency(self, multiple_calibrations):
        """Check consistency across multiple calibration runs"""
        camera_matrices = [cal['camera_matrix'] for cal in multiple_calibrations]
        dist_coeffs = [cal['distortion_coefficients'] for cal in multiple_calibrations]

        # Calculate variations
        cm_mean = np.mean(camera_matrices, axis=0)
        cm_std = np.std(camera_matrices, axis=0)

        dc_mean = np.mean(dist_coeffs, axis=0)
        dc_std = np.std(dist_coeffs, axis=0)

        # Check if variations are within acceptable bounds
        cm_variation = np.max(cm_std) / np.max(cm_mean)
        dc_variation = np.max(dc_std) / np.max(dc_mean)

        return {
            'camera_matrix_variation': cm_variation,
            'distortion_coeff_variation': dc_variation,
            'consistent': cm_variation < 0.01 and dc_variation < 0.05  # 1% and 5% thresholds
        }
```

### Runtime Calibration Monitoring

```python
class RuntimeCalibrationMonitor:
    """Monitor camera calibration during operation"""

    def __init__(self, initial_calibration):
        self.initial_calibration = initial_calibration
        self.calibration_drift_threshold = 0.05  # 5% drift threshold
        self.feature_tracking = True
        self.tracked_features = []

    def detect_feature_drift(self, current_image, reference_features):
        """Detect if camera calibration has drifted"""
        # Detect features in current image
        current_features = self.extract_features(current_image)

        # Match with reference features
        matches = self.match_features(reference_features, current_features)

        # Calculate transformation between matched features
        if len(matches) > 10:  # Need minimum matches for reliable estimation
            src_pts = np.float32([reference_features[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([current_features[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Estimate homography
            homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # Analyze homography to detect calibration drift
            drift_detected = self.analyze_homography_for_drift(homography)

            return drift_detected, homography
        else:
            return False, None

    def extract_features(self, image):
        """Extract features from image for tracking"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Use ORB detector for fast feature extraction
        orb = cv2.ORB_create(nfeatures=500)
        keypoints, descriptors = orb.detectAndCompute(gray, None)

        return keypoints

    def match_features(self, desc1, desc2):
        """Match features between two descriptor sets"""
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)

        return matches

    def analyze_homography_for_drift(self, homography):
        """Analyze homography to detect calibration drift"""
        if homography is None:
            return True  # Cannot estimate, assume drift

        # Check if homography is close to identity (no motion)
        identity = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        diff = np.sum(np.abs(homography - identity))

        # If difference is too large, it might indicate calibration drift
        # rather than just camera motion
        return diff > self.calibration_drift_threshold
```

## Troubleshooting Common Issues

### Calibration Issues

```python
def troubleshoot_calibration_issues():
    """Common camera calibration troubleshooting"""
    issues = {
        'high_reprojection_error': {
            'cause': 'Poor calibration pattern detection or insufficient images',
            'solution': 'Use high contrast pattern, ensure good lighting, capture 20+ diverse images'
        },
        'radial_distortion_not_corrected': {
            'cause': 'Incorrect distortion model or coefficients',
            'solution': 'Verify lens type, try different distortion models (plumb_bob vs rational)'
        },
        'tangential_distortion_present': {
            'cause': 'Misaligned lens elements',
            'solution': 'Include tangential distortion coefficients (p1, p2) in calibration'
        },
        'image_resolution_changed': {
            'cause': 'Using calibration from different resolution',
            'solution': 'Recalibrate or adjust intrinsics for new resolution'
        },
        'camera_moved_since_calibration': {
            'cause': 'Physical displacement of camera',
            'solution': 'Recalibrate or use runtime monitoring to detect drift'
        }
    }

    return issues
```

### Performance Issues

```python
def optimize_camera_performance():
    """Camera performance optimization tips"""
    optimizations = {
        'bandwidth_management': {
            'tip': 'Use appropriate compression for image transmission',
            'example': 'JPEG compression for visual inspection, raw for processing'
        },
        'processing_pipeline': {
            'tip': 'Optimize image processing pipeline for real-time performance',
            'example': 'Use threading, reduce unnecessary processing steps'
        },
        'memory_usage': {
            'tip': 'Manage image buffer sizes to prevent memory overflow',
            'example': 'Use fixed-size queues, implement image dropping when overloaded'
        },
        'cpu_utilization': {
            'tip': 'Offload intensive processing to GPU when possible',
            'example': 'Use CUDA/OpenCL for image filtering and feature detection'
        }
    }

    return optimizations
```

## Best Practices

### 1. Calibration Best Practices

- **Use High-Quality Patterns**: Use printed calibration patterns with high contrast
- **Diverse Poses**: Capture calibration images from various angles and distances
- **Good Lighting**: Ensure consistent, adequate lighting during calibration
- **Pattern Coverage**: Cover the entire field of view with the calibration pattern
- **Multiple Sessions**: Perform calibration multiple times and average results

### 2. Camera Setup Best Practices

- **Secure Mounting**: Ensure camera is securely mounted and won't shift
- **Appropriate FOV**: Choose field of view appropriate for the task
- **Minimize Vibration**: Isolate camera from robot vibrations when possible
- **Environmental Protection**: Protect camera from dust, water, and impacts
- **Regular Maintenance**: Check and clean lens regularly

### 3. Data Processing Best Practices

- **Real-time Constraints**: Optimize algorithms for real-time performance
- **Error Handling**: Implement robust error handling for sensor failures
- **Data Validation**: Continuously validate image quality and calibration
- **Resource Management**: Efficiently manage memory and computational resources
- **Modular Design**: Create modular, reusable image processing components

### 4. Quality Assurance Best Practices

- **Continuous Monitoring**: Monitor calibration quality during operation
- **Performance Metrics**: Track processing time, frame rates, and accuracy
- **Validation Tests**: Regularly validate calibration with known objects
- **Backup Plans**: Implement fallback strategies for sensor failures
- **Documentation**: Maintain detailed records of calibration procedures

## Advanced Topics

### Multi-Camera Systems

```python
class MultiCameraSystem:
    """Manage multiple synchronized cameras"""

    def __init__(self, camera_configs):
        self.cameras = {}
        self.synchronizer = None
        self.extrinsics = {}  # Camera-to-camera transformations

        for config in camera_configs:
            self.cameras[config['name']] = self.setup_camera(config)

    def setup_camera(self, config):
        """Setup individual camera"""
        # This would initialize camera with specific parameters
        pass

    def synchronize_cameras(self):
        """Synchronize multiple cameras"""
        # Implement hardware or software synchronization
        pass

    def perform_multiview_calibration(self):
        """Calibrate multiple cameras relative to each other"""
        # This would perform stereo or multi-view calibration
        pass

    def triangulate_3d_points(self, feature_correspondences):
        """Triangulate 3D points from multiple camera views"""
        # Use camera matrices and feature correspondences to compute 3D positions
        pass
```

### Dynamic Calibration

```python
class DynamicCalibrator:
    """Perform online calibration updates"""

    def __init__(self, initial_calibration):
        self.calibration = initial_calibration
        self.update_threshold = 0.01  # Update when drift exceeds this

    def update_calibration_online(self, current_data):
        """Update calibration using recent data"""
        # Use recent feature matches to refine calibration
        # Only update if improvement is significant
        pass

    def detect_calibration_drift(self, feature_matches):
        """Detect when calibration needs updating"""
        # Analyze feature matches for signs of calibration drift
        pass
```

## Next Steps

After mastering camera setup and calibration:

1. Continue to [LiDAR Integration](./lidar-integration.md) to learn about 3D sensing
2. Practice calibrating cameras on your robot platform
3. Implement real-time image processing pipelines
4. Test camera systems in both simulation and real environments

Your understanding of camera setup and calibration is now foundational for creating visual perception systems in the Physical AI and Humanoid Robotics course!
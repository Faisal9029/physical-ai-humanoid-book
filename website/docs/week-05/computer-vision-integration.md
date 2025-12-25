---
sidebar_position: 5
---

# Computer Vision Integration

This guide covers the integration of computer vision systems in robotics applications. Computer vision enables robots to perceive and understand their environment through visual information, making it essential for navigation, manipulation, and interaction tasks.

## Overview

Computer vision in robotics involves processing visual data from cameras to extract meaningful information for robot decision-making and control. This includes:

- **Object Detection**: Identifying and localizing objects in the environment
- **Semantic Segmentation**: Understanding pixel-level scene composition
- **Depth Estimation**: Extracting 3D information from 2D images
- **Visual SLAM**: Simultaneous localization and mapping using visual features
- **Pose Estimation**: Determining object and robot poses from images
- **Visual Servoing**: Controlling robot motion based on visual feedback

### Key Challenges

- **Real-time Processing**: Meeting computational constraints for real-time operation
- **Variable Lighting**: Handling different lighting conditions and shadows
- **Motion Blur**: Dealing with camera and object motion
- **Occlusions**: Handling partially visible objects
- **Scale Variation**: Recognizing objects at different distances
- **Viewpoint Changes**: Robust recognition across different viewing angles

## Vision System Architecture

### Processing Pipeline

The typical computer vision pipeline for robotics:

```
Raw Image → Preprocessing → Feature Extraction → Recognition → Action
```

Each stage serves a specific purpose:

1. **Preprocessing**: Image enhancement, noise reduction, geometric corrections
2. **Feature Extraction**: Extracting relevant visual features
3. **Recognition**: Identifying objects, scenes, or patterns
4. **Action**: Converting visual information to robot actions

### System Components

#### Camera Systems

```python
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class CameraParameters:
    """Camera intrinsic and extrinsic parameters"""
    # Intrinsic parameters
    fx: float  # Focal length in x (pixels)
    fy: float  # Focal length in y (pixels)
    cx: float  # Principal point x (pixels)
    cy: float  # Principal point y (pixels)

    # Distortion coefficients
    k1: float = 0.0  # Radial distortion coefficient 1
    k2: float = 0.0  # Radial distortion coefficient 2
    p1: float = 0.0  # Tangential distortion coefficient 1
    p2: float = 0.0  # Tangential distortion coefficient 2
    k3: float = 0.0  # Radial distortion coefficient 3

    # Extrinsic parameters (relative to robot base)
    rotation: Tuple[float, float, float, float] = (0, 0, 0, 1)  # Quaternion (x, y, z, w)
    translation: Tuple[float, float, float] = (0, 0, 0)  # Position (x, y, z)

    # Image properties
    width: int = 640
    height: int = 480
    fps: float = 30.0

class CameraSystem:
    """Interface for camera systems in robotics"""

    def __init__(self, params: CameraParameters):
        self.params = params
        self.camera_matrix = np.array([
            [params.fx, 0, params.cx],
            [0, params.fy, params.cy],
            [0, 0, 1]
        ])

        self.dist_coeffs = np.array([
            params.k1, params.k2, params.p1, params.p2, params.k3
        ])

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """Remove lens distortion from image"""
        return cv2.undistort(image, self.camera_matrix, self.dist_coeffs)

    def rectify_image(self, image: np.ndarray) -> np.ndarray:
        """Rectify image using camera parameters"""
        # Create rectification maps
        map1, map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix, self.dist_coeffs, None, self.camera_matrix,
            (self.params.width, self.params.height), cv2.CV_32FC1
        )

        # Apply rectification
        return cv2.remap(image, map1, map2, cv2.INTER_LINEAR)

    def project_3d_to_2d(self, points_3d: np.ndarray) -> np.ndarray:
        """Project 3D points to 2D image coordinates"""
        # Apply extrinsic transformation (if needed)
        # Then project using intrinsic matrix
        points_2d, _ = cv2.projectPoints(
            points_3d.reshape(-1, 1, 3),
            np.zeros(3),  # rvec (rotation vector)
            np.zeros(3),  # tvec (translation vector)
            self.camera_matrix,
            self.dist_coeffs
        )

        return points_2d.reshape(-1, 2)

    def project_2d_to_3d(self, points_2d: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Project 2D image coordinates to 3D points using depth"""
        # Convert to normalized coordinates
        x_norm = (points_2d[:, 0] - self.params.cx) / self.params.fx
        y_norm = (points_2d[:, 1] - self.params.cy) / self.params.fy

        # Create 3D points
        points_3d = np.column_stack([
            x_norm * depth,
            y_norm * depth,
            depth
        ])

        return points_3d
```

#### Image Processing Pipeline

```python
class VisionPipeline:
    """Modular vision processing pipeline"""

    def __init__(self):
        self.preprocessing_steps = []
        self.feature_extractors = []
        self.recognition_modules = []
        self.postprocessors = []

    def add_preprocessing_step(self, step_func):
        """Add a preprocessing step to the pipeline"""
        self.preprocessing_steps.append(step_func)

    def add_feature_extractor(self, extractor_func):
        """Add a feature extraction step"""
        self.feature_extractors.append(extractor_func)

    def add_recognition_module(self, recognition_func):
        """Add a recognition module"""
        self.recognition_modules.append(recognition_func)

    def add_postprocessor(self, postproc_func):
        """Add a post-processing step"""
        self.postprocessors.append(postproc_func)

    def process(self, image: np.ndarray) -> dict:
        """Process image through the complete pipeline"""
        result = {'original_image': image}

        # Preprocessing
        processed_image = image.copy()
        for step in self.preprocessing_steps:
            processed_image = step(processed_image)

        result['preprocessed_image'] = processed_image

        # Feature extraction
        features = {}
        for extractor in self.feature_extractors:
            feature_result = extractor(processed_image)
            features.update(feature_result)

        result['features'] = features

        # Recognition
        recognition_result = {}
        for recognizer in self.recognition_modules:
            rec_result = recognizer(processed_image, features)
            recognition_result.update(rec_result)

        result['recognition'] = recognition_result

        # Post-processing
        for postproc in self.postprocessors:
            result = postproc(result)

        return result

# Example pipeline for object detection
def create_object_detection_pipeline():
    """Create a pipeline for object detection"""
    pipeline = VisionPipeline()

    # Preprocessing: enhance image quality
    def enhance_image(image):
        # Convert to LAB for illumination normalization
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        # Merge and convert back
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    pipeline.add_preprocessing_step(enhance_image)

    # Feature extraction: extract visual features
    def extract_features(image):
        # Extract SIFT features
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)

        return {
            'keypoints': keypoints,
            'descriptors': descriptors,
            'feature_count': len(keypoints) if keypoints else 0
        }

    pipeline.add_feature_extractor(extract_features)

    # Recognition: perform object detection
    def detect_objects(image, features):
        # This would typically use a deep learning model
        # For now, we'll simulate detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Simple blob detection as example
        detector = cv2.SimpleBlobDetector_create()
        keypoints = detector.detect(gray)

        # Convert to detection format
        detections = []
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            size = int(kp.size)
            detections.append({
                'bbox': [x - size//2, y - size//2, x + size//2, y + size//2],
                'confidence': 0.8,  # Simulated confidence
                'class': 'object'
            })

        return {'detections': detections, 'count': len(detections)}

    pipeline.add_recognition_module(detect_objects)

    # Post-processing: filter and refine detections
    def filter_detections(result):
        # Filter detections based on confidence
        if 'detections' in result['recognition']:
            filtered_detections = [
                det for det in result['recognition']['detections']
                if det['confidence'] > 0.5
            ]
            result['recognition']['filtered_detections'] = filtered_detections

        return result

    pipeline.add_postprocessor(filter_detections)

    return pipeline
```

## Deep Learning for Computer Vision

### Vision Models in Robotics

Deep learning has revolutionized computer vision in robotics. Key models include:

#### Object Detection

```python
import torch
import torchvision
from torchvision import transforms
import numpy as np

class RoboticObjectDetector:
    """Object detection for robotics applications"""

    def __init__(self, model_name='yolov5s', confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load pre-trained model
        if model_name == 'yolov5s':
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        elif model_name == 'yolo_fastest':
            # Alternative lightweight model
            self.model = self.load_yolo_fastest()
        else:
            raise ValueError(f"Unknown model: {model_name}")

        self.model.to(self.device)
        self.model.eval()

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
        ])

    def detect_objects(self, image: np.ndarray) -> dict:
        """Detect objects in image"""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Preprocess image
        input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            results = self.model(input_tensor)

        # Process results
        detections = self.process_yolo_results(results, image.shape)

        return detections

    def process_yolo_results(self, results, image_shape):
        """Process YOLO detection results"""
        # Get predictions
        preds = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, conf, cls]

        # Filter by confidence
        high_conf_preds = preds[preds[:, 4] >= self.confidence_threshold]

        # Convert to standard format
        detections = []
        for pred in high_conf_preds:
            x1, y1, x2, y2, conf, cls = pred
            width = x2 - x1
            height = y2 - y1

            detection = {
                'bbox': [int(x1), int(y1), int(width), int(height)],
                'confidence': float(conf),
                'class_id': int(cls),
                'class_name': self.model.names[int(cls)],
                'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)]
            }
            detections.append(detection)

        return {
            'detections': detections,
            'count': len(detections),
            'image_shape': image_shape
        }

    def detect_and_track_objects(self, image: np.ndarray, previous_detections=None):
        """Detect objects and associate with previous detections for tracking"""
        current_detections = self.detect_objects(image)

        if previous_detections is not None:
            # Associate current detections with previous ones using IoU
            tracked_detections = self.associate_detections(
                current_detections['detections'],
                previous_detections.get('detections', [])
            )
            current_detections['tracked_detections'] = tracked_detections

        return current_detections

    def associate_detections(self, current_dets, previous_dets, iou_threshold=0.3):
        """Associate current detections with previous detections"""
        associations = []

        for curr_det in current_dets:
            best_match = None
            best_iou = 0

            for prev_det in previous_dets:
                iou = self.calculate_iou(curr_det['bbox'], prev_det['bbox'])
                if iou > best_iou and iou > iou_threshold:
                    best_iou = iou
                    best_match = prev_det

            if best_match:
                # Update with tracking information
                curr_det['track_id'] = best_match.get('track_id', len(associations))
                curr_det['velocity'] = self.estimate_velocity(
                    curr_det['center'], best_match.get('center', curr_det['center']),
                    time_delta=1.0/30.0  # Assuming 30 FPS
                )
            else:
                # New track
                curr_det['track_id'] = len(associations) + len(previous_dets)
                curr_det['velocity'] = [0, 0]

            associations.append(curr_det)

        return associations

    def calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Convert to x1, y1, x2, y2 format
        x1_1, y1_1, x2_1, y2_1 = x1, y1, x1 + w1, y1 + h1
        x1_2, y1_2, x2_2, y2_2 = x2, y2, x2 + w2, y2 + h2

        # Calculate intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        if xi2 < xi1 or yi2 < yi1:
            return 0

        intersection = (xi2 - xi1) * (yi2 - yi1)
        union = (x2_1 - x1_1) * (y2_1 - y1_1) + (x2_2 - x1_2) * (y2_2 - y1_2) - intersection

        return intersection / union if union > 0 else 0

    def estimate_velocity(self, current_center, previous_center, time_delta):
        """Estimate velocity from position change"""
        if previous_center is None:
            return [0, 0]

        dx = current_center[0] - previous_center[0]
        dy = current_center[1] - previous_center[1]

        vx = dx / time_delta
        vy = dy / time_delta

        return [vx, vy]
```

#### Semantic Segmentation

```python
class SemanticSegmentation:
    """Semantic segmentation for scene understanding"""

    def __init__(self, model_name='deeplabv3_resnet50'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if model_name == 'deeplabv3_resnet50':
            self.model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
        elif model_name == 'fcn_resnet50':
            self.model = torchvision.models.segmentation.fcn_resnet50(pretrained=True)
        else:
            raise ValueError(f"Unknown segmentation model: {model_name}")

        self.model.to(self.device)
        self.model.eval()

        # COCO dataset class names
        self.class_names = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
            'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
            'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
            'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
            'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

        # Preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((520, 520)),  # DeeplabV3 expects 520x520
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def segment_image(self, image: np.ndarray) -> dict:
        """Perform semantic segmentation on image"""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Preprocess
        input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]
            output_predictions = output.argmax(0).cpu().numpy()

        # Create colored segmentation map
        segmentation_map = self.colorize_segmentation(output_predictions)

        # Extract object masks and properties
        objects = self.extract_objects_from_segmentation(output_predictions, image_rgb)

        return {
            'segmentation_map': segmentation_map,
            'class_predictions': output_predictions,
            'objects': objects,
            'image_shape': image.shape
        }

    def colorize_segmentation(self, segmentation):
        """Colorize segmentation map for visualization"""
        # Generate random colors for each class
        np.random.seed(42)  # For consistent colors
        colors = np.random.randint(0, 255, size=(len(self.class_names), 3), dtype=np.uint8)

        # Create color map
        colored_map = np.zeros((segmentation.shape[0], segmentation.shape[1], 3), dtype=np.uint8)
        for class_id in np.unique(segmentation):
            mask = segmentation == class_id
            colored_map[mask] = colors[class_id]

        return colored_map

    def extract_objects_from_segmentation(self, segmentation, original_image):
        """Extract object information from segmentation"""
        objects = []

        for class_id in np.unique(segmentation):
            if class_id == 0:  # Skip background
                continue

            # Create mask for this class
            mask = (segmentation == class_id).astype(np.uint8)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) > 100:  # Filter small contours
                    # Calculate bounding box
                    x, y, w, h = cv2.boundingRect(contour)

                    # Calculate centroid
                    moments = cv2.moments(contour)
                    if moments['m00'] != 0:
                        cx = int(moments['m10'] / moments['m00'])
                        cy = int(moments['m01'] / moments['m00'])
                    else:
                        cx, cy = x + w//2, y + h//2

                    # Calculate area
                    area = cv2.contourArea(contour)

                    # Calculate compactness (ratio of area to perimeter squared)
                    perimeter = cv2.arcLength(contour, True)
                    compactness = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

                    # Get region properties
                    region_mask = mask[y:y+h, x:x+w]
                    region_pixels = original_image[y:y+h, x:x+w][region_mask == class_id]

                    # Calculate average color in the region
                    avg_color = np.mean(region_pixels, axis=0) if len(region_pixels) > 0 else [0, 0, 0]

                    object_info = {
                        'class_id': int(class_id),
                        'class_name': self.class_names[class_id] if class_id < len(self.class_names) else f'unknown_{class_id}',
                        'bbox': [int(x), int(y), int(w), int(h)],
                        'center': [int(cx), int(cy)],
                        'area': int(area),
                        'compactness': float(compactness),
                        'contour': contour,
                        'avg_color': avg_color.tolist()
                    }

                    objects.append(object_info)

        return objects

    def get_class_statistics(self, segmentation):
        """Get statistics about class distribution in segmentation"""
        unique, counts = np.unique(segmentation, return_counts=True)
        total_pixels = segmentation.size

        stats = {}
        for class_id, count in zip(unique, counts):
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f'unknown_{class_id}'
            stats[class_name] = {
                'pixel_count': int(count),
                'percentage': float(count / total_pixels * 100)
            }

        return stats
```

#### Depth Estimation

```python
class DepthEstimator:
    """Monocular depth estimation for robotics"""

    def __init__(self, model_name='midas_small'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if model_name == 'midas_small':
            # Using MiDaS for monocular depth estimation
            self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=True)
        elif model_name == 'dpt_beit':
            # DPT model for better accuracy
            self.model = torch.hub.load("intel-isl/MiDaS", "DPT_BEiT_L_384", pretrained=True)
        else:
            raise ValueError(f"Unknown depth model: {model_name}")

        self.model.to(self.device)
        self.model.eval()

        # Normalization transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def estimate_depth(self, image: np.ndarray) -> dict:
        """Estimate depth from single image"""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Preprocess
        input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            prediction = self.model(input_tensor)

        # Resize to original image size
        depth_map = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        # Convert to numpy
        depth_np = depth_map.cpu().numpy()

        # Normalize for visualization
        depth_normalized = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX)
        depth_colormap = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)

        return {
            'depth_map': depth_np,
            'depth_visualization': depth_colormap,
            'min_depth': float(np.min(depth_np)),
            'max_depth': float(np.max(depth_np)),
            'mean_depth': float(np.mean(depth_np)),
            'image_shape': image.shape
        }

    def create_point_cloud_from_depth(self, depth_map, camera_params: CameraParameters):
        """Create 3D point cloud from depth map and camera parameters"""
        h, w = depth_map.shape

        # Create coordinate grids
        x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))

        # Convert to normalized coordinates
        x_norm = (x_coords - camera_params.cx) / camera_params.fx
        y_norm = (y_coords - camera_params.cy) / camera_params.fy

        # Calculate 3D coordinates
        z_coords = depth_map  # Depth values
        x_coords_3d = x_norm * z_coords
        y_coords_3d = y_norm * z_coords

        # Stack to create point cloud
        points_3d = np.stack([x_coords_3d, y_coords_3d, z_coords], axis=-1)

        # Flatten for easier processing
        points_flat = points_3d.reshape(-1, 3)
        valid_mask = (depth_map.flatten() > 0) & (depth_map.flatten() < 1000)  # Filter invalid depths

        return {
            'points': points_flat[valid_mask],
            'colors': np.tile([128, 128, 128], (np.sum(valid_mask), 1)),  # Default gray colors
            'valid_count': np.sum(valid_mask),
            'invalid_count': len(valid_mask) - np.sum(valid_mask)
        }

    def detect_obstacles_from_depth(self, depth_map, min_distance=0.3, max_distance=5.0):
        """Detect obstacles using depth information"""
        # Filter depth map to valid range
        valid_depths = (depth_map >= min_distance) & (depth_map <= max_distance)
        filtered_depth = np.where(valid_depths, depth_map, np.inf)

        # Create obstacle map (areas with depth below threshold are obstacles)
        obstacle_threshold = min_distance + 0.5  # Consider as obstacle if closer than this
        obstacle_map = (filtered_depth < obstacle_threshold) & (filtered_depth != np.inf)

        # Find obstacle regions
        obstacle_regions = self.find_obstacle_regions(obstacle_map)

        return {
            'obstacle_map': obstacle_map.astype(np.uint8) * 255,
            'obstacle_regions': obstacle_regions,
            'obstacle_count': len(obstacle_regions)
        }

    def find_obstacle_regions(self, obstacle_map):
        """Find connected obstacle regions"""
        # Find contours of obstacle regions
        contours, _ = cv2.findContours(
            obstacle_map.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        regions = []
        for contour in contours:
            if cv2.contourArea(contour) > 50:  # Filter small regions
                # Calculate bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)

                # Calculate centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = x + w//2, y + h//2

                # Calculate average depth in region
                mask = np.zeros(obstacle_map.shape, dtype=np.uint8)
                cv2.fillPoly(mask, [contour], 1)
                avg_depth = np.mean(depth_map[mask == 1])

                region_info = {
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'center': [int(cx), int(cy)],
                    'area': int(cv2.contourArea(contour)),
                    'avg_depth': float(avg_depth)
                }

                regions.append(region_info)

        return regions
```

## Vision-Based Navigation

### Visual SLAM Integration

```python
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d

class VisualSLAM:
    """Visual SLAM system for robot localization and mapping"""

    def __init__(self):
        # Feature detector and descriptor
        self.detector = cv2.SIFT_create()

        # FLANN matcher for efficient feature matching
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

        # Map representation
        self.map_points = []  # 3D points in global map
        self.keyframes = []   # Key camera poses
        self.localizer = PoseEstimator()

        # Tracking state
        self.current_pose = np.eye(4)  # 4x4 transformation matrix
        self.reference_keyframe = None
        self.tracking_lost = False

        # Parameters
        self.min_matches = 20
        self.reprojection_threshold = 3.0
        self.keyframe_threshold = 50  # Minimum distance for new keyframe

    def process_frame(self, image: np.ndarray, camera_params: CameraParameters):
        """Process a new frame for SLAM"""
        # Extract features from current frame
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)

        if descriptors is None or len(descriptors) < self.min_matches:
            return {'status': 'insufficient_features', 'pose': self.current_pose}

        # If this is the first frame, create initial keyframe
        if not self.keyframes:
            self.create_initial_keyframe(image, keypoints, descriptors, camera_params)
            return {'status': 'initialized', 'pose': self.current_pose}

        # Match features with reference keyframe
        if self.reference_keyframe is not None:
            matches = self.matcher.knnMatch(
                descriptors, self.reference_keyframe['descriptors'], k=2
            )

            # Apply Lowe's ratio test for good matches
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)

            if len(good_matches) >= self.min_matches:
                # Estimate pose using matched features
                ref_points = np.float32([self.reference_keyframe['keypoints'][m.trainIdx].pt
                                        for m in good_matches]).reshape(-1, 1, 2)
                curr_points = np.float32([keypoints[m.queryIdx].pt
                                         for m in good_matches]).reshape(-1, 1, 2)

                # Estimate essential matrix
                E, mask = cv2.findEssentialMat(
                    ref_points, curr_points,
                    camera_params.fx, (camera_params.cx, camera_params.cy),
                    method=cv2.RANSAC, threshold=self.reprojection_threshold
                )

                if E is not None:
                    # Recover pose
                    _, R, t, _ = cv2.recoverPose(E, ref_points, curr_points)

                    # Create transformation matrix
                    T = np.eye(4)
                    T[:3, :3] = R
                    T[:3, 3] = t.flatten()

                    # Update current pose
                    self.current_pose = self.current_pose @ T

                    # Check if we need a new keyframe
                    if self.should_create_keyframe():
                        self.create_keyframe(image, keypoints, descriptors, self.current_pose)

                    return {
                        'status': 'tracking_success',
                        'pose': self.current_pose,
                        'matches': len(good_matches)
                    }
                else:
                    return {'status': 'pose_estimation_failed', 'pose': self.current_pose}
            else:
                return {'status': 'insufficient_matches', 'pose': self.current_pose}
        else:
            return {'status': 'no_reference_keyframe', 'pose': self.current_pose}

    def create_initial_keyframe(self, image, keypoints, descriptors, camera_params):
        """Create the initial keyframe"""
        initial_keyframe = {
            'image': image,
            'keypoints': keypoints,
            'descriptors': descriptors,
            'pose': np.eye(4),  # Initial pose at origin
            'timestamp': cv2.getTickCount(),
            'features': len(keypoints)
        }

        self.keyframes.append(initial_keyframe)
        self.reference_keyframe = initial_keyframe

    def create_keyframe(self, image, keypoints, descriptors, pose):
        """Create a new keyframe when sufficient motion detected"""
        keyframe = {
            'image': image,
            'keypoints': keypoints,
            'descriptors': descriptors,
            'pose': pose.copy(),
            'timestamp': cv2.getTickCount(),
            'features': len(keypoints)
        }

        self.keyframes.append(keyframe)
        self.reference_keyframe = keyframe

    def should_create_keyframe(self):
        """Determine if a new keyframe should be created"""
        if not self.keyframes:
            return True

        # Calculate distance from last keyframe
        last_keyframe_pose = self.keyframes[-1]['pose']
        current_position = self.current_pose[:3, 3]
        last_position = last_keyframe_pose[:3, 3]

        distance = np.linalg.norm(current_position - last_position)

        return distance > self.keyframe_threshold

    def triangulate_points(self, keyframe1, keyframe2, matches):
        """Triangulate 3D points from stereo keyframes"""
        # Get matched points
        pts1 = np.float32([keyframe1['keypoints'][m.queryIdx].pt for m in matches]).reshape(-1, 2)
        pts2 = np.float32([keyframe2['keypoints'][m.trainIdx].pt for m in matches]).reshape(-1, 2)

        # Get camera matrices (assuming same camera for both keyframes)
        K = np.array([
            [self.camera_params.fx, 0, self.camera_params.cx],
            [0, self.camera_params.fy, self.camera_params.cy],
            [0, 0, 1]
        ])

        # Get relative pose between keyframes
        rel_pose = np.linalg.inv(keyframe1['pose']) @ keyframe2['pose']
        R_rel = rel_pose[:3, :3]
        t_rel = rel_pose[:3, 3]

        # Triangulate points
        points_4d = cv2.triangulatePoints(
            K @ np.eye(4)[:3],  # Projection matrix for first camera
            K @ np.hstack([R_rel, t_rel.reshape(-1, 1)])[:3],  # Projection matrix for second camera
            pts1.T, pts2.T
        )

        # Convert from homogeneous to Euclidean coordinates
        points_3d = (points_4d[:3] / points_4d[3]).T

        return points_3d

    def get_global_map(self):
        """Get the global map of 3D points"""
        if not self.map_points:
            return np.empty((0, 3))

        return np.array(self.map_points)

    def get_current_pose(self):
        """Get current estimated pose"""
        return self.current_pose.copy()

    def reset(self):
        """Reset SLAM system"""
        self.map_points = []
        self.keyframes = []
        self.current_pose = np.eye(4)
        self.reference_keyframe = None
        self.tracking_lost = False
```

### Visual Servoing

```python
class VisualServoing:
    """Visual servoing for robot control based on visual feedback"""

    def __init__(self, camera_params: CameraParameters):
        self.camera_params = camera_params
        self.feature_tracker = cv2.ORB_create(nfeatures=500)
        self.target_features = None
        self.current_features = None
        self.feature_matches = []

        # Control parameters
        self.kp = 1.0  # Proportional gain
        self.ki = 0.1  # Integral gain
        self.kd = 0.05  # Derivative gain

        # Error accumulation for integral term
        self.error_integral = np.zeros(6)
        self.previous_error = np.zeros(6)

    def set_target(self, target_image: np.ndarray):
        """Set the target image for visual servoing"""
        self.target_image = target_image
        self.target_keypoints, self.target_descriptors = self.feature_tracker.detectAndCompute(
            cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY), None
        )

    def compute_servo_control(self, current_image: np.ndarray):
        """Compute visual servoing control commands"""
        current_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
        current_keypoints, current_descriptors = self.feature_tracker.detectAndCompute(
            current_gray, None
        )

        if (self.target_descriptors is None or current_descriptors is None or
            len(self.target_descriptors) < 10 or len(current_descriptors) < 10):
            return np.zeros(6), {'status': 'insufficient_features'}

        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(self.target_descriptors, current_descriptors)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) < 10:
            return np.zeros(6), {'status': 'insufficient_matches'}

        # Calculate error based on feature displacement
        error = self.calculate_feature_error(matches, current_keypoints)

        # Apply PID control
        control_command = self.pid_control(error)

        return control_command, {
            'status': 'success',
            'matches': len(matches),
            'error': error.tolist()
        }

    def calculate_feature_error(self, matches, current_keypoints):
        """Calculate error based on feature displacement"""
        if not matches:
            return np.zeros(6)

        # Calculate average displacement of matched features
        displacements = []
        for match in matches[:20]:  # Use top 20 matches
            target_pt = self.target_keypoints[match.queryIdx].pt
            current_pt = current_keypoints[match.trainIdx].pt

            displacement = np.array([current_pt[0] - target_pt[0], current_pt[1] - target_pt[1]])
            displacements.append(displacement)

        avg_displacement = np.mean(displacements, axis=0)

        # Convert pixel error to 3D error using camera parameters
        # This is a simplified approach - real implementation would use depth
        pixel_to_meters = 0.001  # Approximate conversion factor
        error_3d = np.array([
            avg_displacement[0] * pixel_to_meters,  # x translation
            avg_displacement[1] * pixel_to_meters,  # y translation
            0,  # z translation (would need depth)
            0,  # roll
            0,  # pitch
            0   # yaw (would need rotation estimation)
        ])

        return error_3d

    def pid_control(self, error):
        """PID controller for visual servoing"""
        # Proportional term
        proportional = self.kp * error

        # Integral term
        self.error_integral += error
        integral = self.ki * self.error_integral

        # Derivative term
        derivative = self.kd * (error - self.previous_error)
        self.previous_error = error

        control_output = proportional + integral + derivative

        return control_output

    def compute_image_jacobian(self, point_2d, depth):
        """Compute image Jacobian for visual servoing"""
        # Point in 2D image coordinates [u, v]
        u, v = point_2d

        # Camera parameters
        fx, fy = self.camera_params.fx, self.camera_params.fy
        cx, cy = self.camera_params.cx, self.camera_params.cy

        # Image Jacobian matrix (2x6 for 2D point, 6 DOF motion)
        L = np.zeros((2, 6))

        # Translation terms
        L[0, 0] = 1.0 / depth  # du/dx
        L[0, 1] = 0.0          # du/dy
        L[0, 2] = -u / depth   # du/dz

        L[1, 0] = 0.0          # dv/dx
        L[1, 1] = 1.0 / depth  # dv/dy
        L[1, 2] = -v / depth   # dv/dz

        # Rotation terms
        L[0, 3] = u * v / depth  # du/drx
        L[0, 4] = -(fx + u*u/fx) / depth  # du/dry
        L[0, 5] = v / fx  # du/drz

        L[1, 3] = (fy + v*v/fy) / depth  # dv/drx
        L[1, 4] = -u * v / depth  # dv/dry
        L[1, 5] = -u / fy  # dv/drz

        return L

    def ibvs_control(self, current_features, target_features):
        """Image-based visual servoing control"""
        if len(current_features) != len(target_features):
            return np.zeros(6), {'status': 'feature_count_mismatch'}

        # Calculate image error
        error = np.zeros(2 * len(current_features))
        for i, (curr_feat, target_feat) in enumerate(zip(current_features, target_features)):
            error[2*i:2*i+2] = curr_feat.pt - target_feat.pt

        # Compute interaction matrix (simplified)
        interaction_matrix = np.eye(2 * len(current_features))

        # Control law: v = -λ * L† * e
        lambda_gain = 0.5  # Control gain
        control_velocity = -lambda_gain * np.linalg.pinv(interaction_matrix) @ error

        # Convert to 6-DOF robot velocity
        robot_velocity = np.zeros(6)
        # Simplified mapping - in practice, this would be more complex
        robot_velocity[0:2] = control_velocity[0:2]  # Map first 2 to x, y

        return robot_velocity, {'status': 'success', 'error_norm': np.linalg.norm(error)}
```

## Performance Optimization

### Real-time Processing

```python
import threading
import queue
from collections import deque
import time

class RealTimeVisionProcessor:
    """Real-time vision processing with threading and queuing"""

    def __init__(self, max_queue_size=10):
        self.max_queue_size = max_queue_size
        self.input_queue = queue.Queue(maxsize=max_queue_size)
        self.output_queue = queue.Queue(maxsize=max_queue_size)
        self.processing_thread = None
        self.running = False

        # Performance metrics
        self.processing_times = deque(maxlen=100)
        self.fps_counter = 0
        self.last_fps_time = time.time()

    def start_processing(self, processing_func):
        """Start the real-time processing thread"""
        self.processing_func = processing_func
        self.running = True
        self.processing_thread = threading.Thread(target=self.processing_loop, daemon=True)
        self.processing_thread.start()

    def processing_loop(self):
        """Main processing loop running in separate thread"""
        while self.running:
            try:
                # Get image from input queue
                image = self.input_queue.get(timeout=0.1)

                # Process image
                start_time = time.time()
                result = self.processing_func(image)
                end_time = time.time()

                # Record processing time
                self.processing_times.append(end_time - start_time)

                # Put result in output queue
                try:
                    self.output_queue.put_nowait(result)
                except queue.Full:
                    # Drop oldest result if output queue is full
                    try:
                        self.output_queue.get_nowait()
                        self.output_queue.put_nowait(result)
                    except queue.Empty:
                        self.output_queue.put_nowait(result)

                # Update FPS counter
                self.fps_counter += 1
                current_time = time.time()
                if current_time - self.last_fps_time >= 1.0:
                    self.current_fps = self.fps_counter
                    self.fps_counter = 0
                    self.last_fps_time = current_time

            except queue.Empty:
                continue  # Timeout occurred, continue loop
            except Exception as e:
                print(f"Error in processing loop: {e}")
                continue

    def submit_image(self, image):
        """Submit an image for processing"""
        try:
            self.input_queue.put_nowait(image)
            return True
        except queue.Full:
            # Queue is full, drop oldest image
            try:
                self.input_queue.get_nowait()
                self.input_queue.put_nowait(image)
                return True
            except queue.Empty:
                self.input_queue.put_nowait(image)
                return True

    def get_result(self):
        """Get processed result"""
        try:
            return self.output_queue.get_nowait()
        except queue.Empty:
            return None

    def stop_processing(self):
        """Stop the processing thread"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join()

    def get_performance_metrics(self):
        """Get performance metrics"""
        if self.processing_times:
            avg_time = sum(self.processing_times) / len(self.processing_times)
            min_time = min(self.processing_times)
            max_time = max(self.processing_times)
        else:
            avg_time = min_time = max_time = 0

        return {
            'avg_processing_time': avg_time,
            'min_processing_time': min_time,
            'max_processing_time': max_time,
            'avg_fps': 1.0 / avg_time if avg_time > 0 else 0,
            'current_fps': getattr(self, 'current_fps', 0),
            'input_queue_size': self.input_queue.qsize(),
            'output_queue_size': self.output_queue.qsize()
        }

# Example usage
def example_real_time_processing():
    """Example of real-time vision processing"""
    # Create vision processor
    vision_processor = RealTimeVisionProcessor(max_queue_size=5)

    # Create object detector
    detector = RoboticObjectDetector(model_name='yolov5s')

    def process_function(image):
        """Processing function for the pipeline"""
        return detector.detect_objects(image)

    # Start processing
    vision_processor.start_processing(process_function)

    # Simulate image submission (in real application, this would come from camera)
    for i in range(100):
        # Create a dummy image (in real app, this comes from camera)
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Submit for processing
        vision_processor.submit_image(dummy_image)

        # Get results occasionally
        result = vision_processor.get_result()
        if result:
            print(f"Processed frame with {result['count']} detections")

        # Small delay to simulate real-time constraints
        time.sleep(0.033)  # ~30 FPS

    # Get performance metrics
    metrics = vision_processor.get_performance_metrics()
    print(f"Performance: {metrics}")

    # Stop processing
    vision_processor.stop_processing()
```

## Integration with ROS 2

### Vision Nodes

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import numpy as np

class VisionNode(Node):
    """ROS 2 node for computer vision processing"""

    def __init__(self):
        super().__init__('vision_node')

        # CV bridge for image conversion
        self.bridge = CvBridge()

        # Create subscriptions
        self.image_subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            '/camera/camera_info',
            self.camera_info_callback,
            10
        )

        # Create publishers
        self.detection_publisher = self.create_publisher(
            Detection2DArray,
            '/vision/detections',
            10
        )

        self.segmentation_publisher = self.create_publisher(
            Image,
            '/vision/segmentation_map',
            10
        )

        # Initialize vision components
        self.object_detector = RoboticObjectDetector()
        self.segmenter = SemanticSegmentation()
        self.depth_estimator = DepthEstimator()

        # Camera parameters
        self.camera_params = None
        self.camera_info_received = False

        # Processing parameters
        self.processing_rate = 10  # Hz
        self.processing_timer = self.create_timer(1.0/self.processing_rate, self.process_callback)

        # Frame counter for processing every N frames
        self.frame_counter = 0
        self.process_every_n_frames = 3  # Process every 3rd frame to reduce CPU load

        self.get_logger().info('Vision node initialized')

    def camera_info_callback(self, msg):
        """Process camera info message"""
        if not self.camera_info_received:
            # Extract camera parameters from CameraInfo message
            self.camera_params = CameraParameters(
                fx=msg.k[0],  # K[0,0]
                fy=msg.k[4],  # K[1,1]
                cx=msg.k[2],  # K[0,2]
                cy=msg.k[5],  # K[1,2]
                width=msg.width,
                height=msg.height
            )

            # Extract distortion coefficients
            if len(msg.d) >= 5:
                self.camera_params.k1 = msg.d[0]
                self.camera_params.k2 = msg.d[1]
                self.camera_params.p1 = msg.d[2]
                self.camera_params.p2 = msg.d[3]
                self.camera_params.k3 = msg.d[4]

            self.camera_info_received = True
            self.get_logger().info('Camera parameters received')

    def image_callback(self, msg):
        """Process incoming image message"""
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Store image for processing
            self.latest_image = cv_image
            self.latest_image_header = msg.header

        except Exception as e:
            self.get_logger().error(f'Error converting image: {e}')

    def process_callback(self):
        """Process images at specified rate"""
        if hasattr(self, 'latest_image'):
            # Process every Nth frame to reduce CPU load
            self.frame_counter += 1
            if self.frame_counter % self.process_every_n_frames != 0:
                return

            image = self.latest_image

            # Perform object detection
            try:
                detections = self.object_detector.detect_objects(image)

                # Publish detections
                self.publish_detections(detections)

                # Perform semantic segmentation
                segmentation_result = self.segmenter.segment_image(image)

                # Publish segmentation map
                self.publish_segmentation(segmentation_result['segmentation_map'])

                # Estimate depth (if applicable)
                if hasattr(self, 'depth_estimator'):
                    depth_result = self.depth_estimator.estimate_depth(image)

                    # Detect obstacles from depth
                    obstacle_result = self.depth_estimator.detect_obstacles_from_depth(
                        depth_result['depth_map']
                    )

                    # Publish obstacle information
                    self.publish_obstacles(obstacle_result)

                self.get_logger().info(f'Processed frame: {detections["count"]} objects detected')

            except Exception as e:
                self.get_logger().error(f'Error processing image: {e}')

    def publish_detections(self, detections_result):
        """Publish object detections"""
        detection_array = Detection2DArray()
        detection_array.header = self.latest_image_header

        for detection in detections_result['detections']:
            vision_detection = Detection2D()

            # Set bounding box
            bbox = detection['bbox']
            vision_detection.bbox.center.x = float(bbox[0] + bbox[2]/2)  # center x
            vision_detection.bbox.center.y = float(bbox[1] + bbox[3]/2)  # center y
            vision_detection.bbox.size_x = float(bbox[2])  # width
            vision_detection.bbox.size_y = float(bbox[3])  # height

            # Set detection results
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = str(detection['class_name'])
            hypothesis.hypothesis.score = float(detection['confidence'])

            vision_detection.results.append(hypothesis)

            detection_array.detections.append(vision_detection)

        self.detection_publisher.publish(detection_array)

    def publish_segmentation(self, segmentation_map):
        """Publish segmentation map"""
        try:
            segmentation_msg = self.bridge.cv2_to_imgmsg(segmentation_map, encoding='bgr8')
            segmentation_msg.header = self.latest_image_header
            self.segmentation_publisher.publish(segmentation_msg)
        except Exception as e:
            self.get_logger().error(f'Error publishing segmentation: {e}')

    def publish_obstacles(self, obstacle_result):
        """Publish obstacle information"""
        # This would typically publish to a separate obstacle topic
        # For now, just log the information
        self.get_logger().info(f'Detected {obstacle_result["obstacle_count"]} obstacles')

def main(args=None):
    rclpy.init(args=args)
    vision_node = VisionNode()

    try:
        rclpy.spin(vision_node)
    except KeyboardInterrupt:
        vision_node.get_logger().info('Shutting down vision node')
    finally:
        vision_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Quality Assurance and Validation

### Vision System Validation

```python
class VisionSystemValidator:
    """Validate vision system performance and accuracy"""

    def __init__(self):
        self.metrics = {
            'detection_accuracy': [],
            'processing_time': [],
            'fps': [],
            'repeatability': [],
            'robustness': []
        }

    def validate_detection_accuracy(self, ground_truth, detections, iou_threshold=0.5):
        """Validate detection accuracy against ground truth"""
        # Calculate precision, recall, and F1-score
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        # Create copies to mark matched detections
        gt_copy = ground_truth.copy()
        det_copy = detections.copy()

        # Match detections to ground truth
        for det in detections:
            best_iou = 0
            best_gt_idx = -1

            for i, gt in enumerate(ground_truth):
                iou = self.calculate_bbox_iou(det['bbox'], gt['bbox'])
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_gt_idx = i

            if best_gt_idx >= 0:
                # True positive: matched to ground truth
                true_positives += 1
                del gt_copy[best_gt_idx]  # Remove matched ground truth
                det_copy.remove(det)      # Remove matched detection
            else:
                # False positive: detection with no matching ground truth
                false_positives += 1

        # Remaining ground truths are false negatives
        false_negatives = len(gt_copy)

        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }

    def validate_processing_performance(self, processing_times):
        """Validate processing performance metrics"""
        if not processing_times:
            return {'error': 'No processing times provided'}

        avg_time = np.mean(processing_times)
        std_time = np.std(processing_times)
        min_time = np.min(processing_times)
        max_time = np.max(processing_times)

        # Calculate FPS
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0

        # Check if performance meets requirements
        meets_real_time = avg_time < 0.1  # < 100ms for 10Hz processing
        meets_throughput = avg_fps >= 10  # >= 10 FPS

        return {
            'avg_processing_time': avg_time,
            'std_processing_time': std_time,
            'min_processing_time': min_time,
            'max_processing_time': max_time,
            'avg_fps': avg_fps,
            'meets_real_time_requirement': meets_real_time,
            'meets_throughput_requirement': meets_throughput,
            'processing_times': processing_times
        }

    def validate_repeatability(self, repeated_measurements):
        """Validate repeatability of vision system"""
        if len(repeated_measurements) < 2:
            return {'error': 'Need at least 2 measurements for repeatability'}

        # Calculate consistency across repeated measurements
        measurements = np.array(repeated_measurements)
        mean_measurement = np.mean(measurements, axis=0)
        std_measurement = np.std(measurements, axis=0)
        cv = std_measurement / mean_measurement if np.all(mean_measurement != 0) else np.zeros_like(std_measurement)

        return {
            'mean': mean_measurement.tolist(),
            'std': std_measurement.tolist(),
            'cv': cv.tolist(),  # Coefficient of variation
            'repeatability_score': float(np.mean(cv)),  # Lower is better
            'measurements_count': len(repeated_measurements)
        }

    def validate_robustness(self, performance_under_conditions):
        """Validate robustness across different conditions"""
        conditions_performance = {}

        for condition, performance in performance_under_conditions.items():
            # Performance could be accuracy, processing time, etc.
            avg_performance = np.mean(performance) if performance else 0
            std_performance = np.std(performance) if len(performance) > 1 else 0

            conditions_performance[condition] = {
                'avg': avg_performance,
                'std': std_performance,
                'count': len(performance) if performance else 0
            }

        # Calculate robustness score (lower variance indicates higher robustness)
        variances = [cond['std'] for cond in conditions_performance.values() if cond['std'] is not None]
        robustness_score = np.mean(variances) if variances else float('inf')

        return {
            'conditions_performance': conditions_performance,
            'robustness_score': robustness_score,
            'most_sensitive_condition': min(conditions_performance.keys(),
                                          key=lambda k: conditions_performance[k]['avg']) if conditions_performance else None
        }

    def calculate_bbox_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union of two bounding boxes"""
        # bbox format: [x, y, width, height]
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        if xi2 < xi1 or yi2 < yi1:
            return 0

        intersection = (xi2 - xi1) * (yi2 - yi1)
        union = w1 * h1 + w2 * h2 - intersection

        return intersection / union if union > 0 else 0

    def run_comprehensive_validation(self, test_data):
        """Run comprehensive validation of vision system"""
        results = {}

        # Validate detection accuracy
        if 'ground_truth' in test_data and 'detections' in test_data:
            results['detection_accuracy'] = self.validate_detection_accuracy(
                test_data['ground_truth'], test_data['detections']
            )

        # Validate processing performance
        if 'processing_times' in test_data:
            results['processing_performance'] = self.validate_processing_performance(
                test_data['processing_times']
            )

        # Validate repeatability
        if 'repeatability_measurements' in test_data:
            results['repeatability'] = self.validate_repeatability(
                test_data['repeatability_measurements']
            )

        # Validate robustness
        if 'robustness_tests' in test_data:
            results['robustness'] = self.validate_robustness(
                test_data['robustness_tests']
            )

        # Generate summary report
        summary = self.generate_validation_summary(results)
        results['summary'] = summary

        return results

    def generate_validation_summary(self, validation_results):
        """Generate summary of validation results"""
        summary = {
            'overall_score': 0.0,
            'passing_tests': 0,
            'total_tests': 0,
            'critical_issues': [],
            'recommendations': []
        }

        # Calculate overall score based on key metrics
        score_components = []

        if 'detection_accuracy' in validation_results:
            acc = validation_results['detection_accuracy']
            f1_score = acc.get('f1_score', 0)
            score_components.append(('detection_f1', f1_score, 0.7))  # 70% target

        if 'processing_performance' in validation_results:
            perf = validation_results['processing_performance']
            meets_rt = perf.get('meets_real_time_requirement', False)
            score_components.append(('real_time', 1.0 if meets_rt else 0.0, 1.0))

        # Calculate weighted average
        if score_components:
            weighted_score = sum(score * weight for _, score, weight in score_components) / len(score_components)
            summary['overall_score'] = weighted_score

        # Identify critical issues
        if 'detection_accuracy' in validation_results:
            acc = validation_results['detection_accuracy']
            if acc.get('f1_score', 0) < 0.5:
                summary['critical_issues'].append('Detection accuracy below acceptable threshold (F1 < 0.5)')

        if 'processing_performance' in validation_results:
            perf = validation_results['processing_performance']
            if not perf.get('meets_real_time_requirement', False):
                summary['critical_issues'].append('Processing time exceeds real-time requirements')

        # Generate recommendations
        if summary['overall_score'] < 0.7:
            summary['recommendations'].append('Performance optimization needed')
        if 'critical_issues' in summary and summary['critical_issues']:
            summary['recommendations'].append('Address critical issues before deployment')

        return summary

# Example usage
def validate_vision_system():
    """Example of validating a vision system"""
    validator = VisionSystemValidator()

    # Simulated test data
    test_data = {
        'ground_truth': [
            {'bbox': [100, 100, 50, 50], 'class': 'person'},
            {'bbox': [200, 150, 30, 30], 'class': 'object'}
        ],
        'detections': [
            {'bbox': [95, 98, 52, 55], 'class': 'person', 'confidence': 0.85},
            {'bbox': [205, 148, 28, 32], 'class': 'object', 'confidence': 0.78}
        ],
        'processing_times': [0.08, 0.09, 0.07, 0.10, 0.085],
        'repeatability_measurements': [
            [100.5, 100.2, 100.8, 100.3],
            [100.4, 100.1, 100.9, 100.2],
            [100.6, 100.3, 100.7, 100.4]
        ],
        'robustness_tests': {
            'bright_lighting': [0.85, 0.82, 0.87],
            'dim_lighting': [0.72, 0.68, 0.75],
            'motion_blur': [0.65, 0.60, 0.68]
        }
    }

    results = validator.run_comprehensive_validation(test_data)

    print("Vision System Validation Results:")
    print(f"Overall Score: {results['summary']['overall_score']:.2f}")
    print(f"Critical Issues: {len(results['summary']['critical_issues'])}")
    print(f"Recommendations: {len(results['summary']['recommendations'])}")

    return results
```

## Troubleshooting Common Issues

### Vision Processing Issues

```python
def troubleshoot_vision_issues():
    """Common vision system troubleshooting guide"""
    issues = {
        'poor_detection_accuracy': {
            'symptoms': ['Low precision/recall', 'Many false positives/negatives', 'Poor confidence scores'],
            'possible_causes': [
                'Insufficient training data',
                'Poor lighting conditions',
                'Objects not in training dataset',
                'Model not suitable for domain'
            ],
            'solutions': [
                'Collect more training data with domain-specific examples',
                'Improve lighting or use IR illumination',
                'Fine-tune model on domain-specific data',
                'Try different model architecture'
            ]
        },
        'slow_processing_performance': {
            'symptoms': ['Low FPS', 'High CPU/GPU usage', 'Dropped frames'],
            'possible_causes': [
                'Model too complex for hardware',
                'Inefficient implementation',
                'Large input resolution',
                'Memory bottlenecks'
            ],
            'solutions': [
                'Use lighter model (YOLOv5s instead of YOLOv5x)',
                'Optimize implementation with threading',
                'Reduce input resolution',
                'Use GPU acceleration'
            ]
        },
        'calibration_drift': {
            'symptoms': ['Inaccurate 3D reconstructions', 'Poor AR overlay alignment', 'Drifting measurements'],
            'possible_causes': [
                'Camera moved since calibration',
                'Temperature changes affecting lens',
                'Mechanical vibration',
                'Improper mounting'
            ],
            'solutions': [
                'Recalibrate camera',
                'Use on-demand calibration',
                'Improve mounting stability',
                'Monitor calibration parameters'
            ]
        },
        'occlusion_handling': {
            'symptoms': ['Lost object tracking', 'Inconsistent detections', 'False negatives'],
            'possible_causes': [
                'Heavy occlusions',
                'Partial visibility',
                'Similar object appearances',
                'Fast motion'
            ],
            'solutions': [
                'Use temporal consistency tracking',
                'Implement partial matching',
                'Use 3D information for tracking',
                'Increase frame rate'
            ]
        },
        'lighting_variations': {
            'symptoms': ['Inconsistent performance', 'False detections in shadows', 'Poor night performance'],
            'possible_causes': [
                'High contrast scenes',
                'Changing lighting conditions',
                'Glare and reflections',
                'Low light conditions'
            ],
            'solutions': [
                'Use adaptive exposure',
                'Implement illumination normalization',
                'Add IR illumination',
                'Use multiple lighting conditions in training'
            ]
        }
    }

    return issues

def performance_optimization_guide():
    """Guide for optimizing vision system performance"""
    optimizations = {
        'model_optimization': {
            'techniques': [
                'Model quantization (INT8 instead of FP32)',
                'Model pruning (removing redundant weights)',
                'Knowledge distillation (student-teacher models)',
                'Architecture optimization (EfficientNet, MobileNet)'
            ],
            'expected_improvement': '2-5x speedup with minimal accuracy loss'
        },
        'implementation_optimization': {
            'techniques': [
                'Batch processing',
                'Multi-threading for I/O operations',
                'GPU acceleration',
                'Memory pooling',
                'Asynchronous processing'
            ],
            'expected_improvement': 'Significant performance gains with proper implementation'
        },
        'pipeline_optimization': {
            'techniques': [
                'Early rejection of unlikely regions',
                'Multi-scale processing',
                'ROI-based processing',
                'Temporal consistency exploitation'
            ],
            'expected_improvement': 'Reduced computation without accuracy loss'
        }
    }

    return optimizations
```

## Best Practices

### 1. Model Selection Best Practices

- **Start Simple**: Begin with simpler models and increase complexity as needed
- **Consider Hardware**: Match model complexity to available hardware
- **Domain Adaptation**: Fine-tune general models on domain-specific data
- **Efficiency vs Accuracy**: Balance computational requirements with performance needs

### 2. Data Quality Best Practices

- **Diverse Training Data**: Include varied lighting, angles, and conditions
- **Balanced Classes**: Ensure equal representation of all object classes
- **High-Quality Annotations**: Use precise bounding boxes and segmentation masks
- **Validation Sets**: Maintain separate validation sets that mirror deployment conditions

### 3. Deployment Best Practices

- **Real-time Constraints**: Optimize for target frame rate requirements
- **Resource Management**: Monitor and manage computational resources
- **Error Handling**: Implement robust error handling and fallback mechanisms
- **Continuous Monitoring**: Track performance metrics in deployed systems

### 4. Integration Best Practices

- **Modular Design**: Create independent, testable vision components
- **Standard Interfaces**: Use ROS 2 message types for interoperability
- **Configuration Management**: Support runtime parameter adjustment
- **Logging and Debugging**: Implement comprehensive logging for debugging

## Advanced Topics

### Multi-modal Fusion

```python
class MultiModalFusion:
    """Fuse vision data with other sensor modalities"""

    def __init__(self):
        self.vision_processor = None
        self.lidar_processor = None
        self.imu_processor = None
        self.fusion_algorithm = None

    def fuse_vision_lidar(self, vision_data, lidar_data, camera_extrinsics):
        """Fuse vision and LiDAR data"""
        # Project LiDAR points to camera frame
        camera_points = self.transform_lidar_to_camera(lidar_data, camera_extrinsics)

        # Associate vision detections with LiDAR points
        fused_detections = self.associate_detections_with_points(
            vision_data['detections'], camera_points
        )

        # Enhance detections with depth information
        enhanced_detections = self.enhance_detections_with_depth(
            fused_detections, camera_points
        )

        return enhanced_detections

    def transform_lidar_to_camera(self, lidar_points, extrinsics_matrix):
        """Transform LiDAR points to camera coordinate frame"""
        # Apply transformation matrix
        points_homo = np.hstack([lidar_points, np.ones((len(lidar_points), 1))])
        camera_points_homo = (extrinsics_matrix @ points_homo.T).T
        camera_points = camera_points_homo[:, :3] / camera_points_homo[:, 3:]

        return camera_points

    def associate_detections_with_points(self, detections, camera_points):
        """Associate vision detections with corresponding LiDAR points"""
        associated_detections = []

        for detection in detections:
            bbox = detection['bbox']
            x, y, w, h = bbox

            # Find LiDAR points within bounding box
            mask = (
                (camera_points[:, 0] >= x) & (camera_points[:, 0] <= x + w) &
                (camera_points[:, 1] >= y) & (camera_points[:, 1] <= y + h)
            )

            points_in_bbox = camera_points[mask]

            # Add point cloud information to detection
            detection['points_in_roi'] = points_in_bbox
            detection['roi_point_count'] = len(points_in_bbox)

            if len(points_in_bbox) > 0:
                # Calculate depth statistics
                detection['depth_mean'] = float(np.mean(points_in_bbox[:, 2]))
                detection['depth_std'] = float(np.std(points_in_bbox[:, 2]))
                detection['depth_median'] = float(np.median(points_in_bbox[:, 2]))

            associated_detections.append(detection)

        return associated_detections

    def enhance_detections_with_depth(self, detections, camera_points):
        """Enhance detections with depth information"""
        enhanced_detections = []

        for detection in detections:
            if detection.get('roi_point_count', 0) > 0:
                # Calculate 3D bounding box from points
                points_in_roi = detection['points_in_roi']

                # Calculate 3D bounding box
                x_min, y_min, z_min = np.min(points_in_roi, axis=0)
                x_max, y_max, z_max = np.max(points_in_roi, axis=0)

                detection['bbox_3d'] = {
                    'center': [(x_min + x_max)/2, (y_min + y_max)/2, (z_min + z_max)/2],
                    'size': [x_max - x_min, y_max - y_min, z_max - z_min]
                }

                # Calculate distance to object
                detection['distance'] = float(np.mean(points_in_roi[:, 2]))

                # Estimate object size in 3D
                detection['estimated_size'] = {
                    'width': float(x_max - x_min),
                    'height': float(y_max - y_min),
                    'depth': float(z_max - z_min)
                }

            enhanced_detections.append(detection)

        return enhanced_detections
```

### Online Learning and Adaptation

```python
class OnlineLearningAdapter:
    """Adapt vision models to new environments online"""

    def __init__(self, base_model):
        self.base_model = base_model
        self.adaptation_buffer = []
        self.adaptation_threshold = 100  # Number of samples before adaptation
        self.performance_monitor = None

    def monitor_performance(self, current_performance):
        """Monitor current performance for adaptation triggers"""
        if self.performance_monitor is None:
            self.performance_monitor = {
                'baseline_performance': current_performance,
                'current_performance': current_performance,
                'samples_collected': 0,
                'adaptation_needed': False
            }
        else:
            self.performance_monitor['current_performance'] = current_performance
            self.performance_monitor['samples_collected'] += 1

            # Check if performance has degraded significantly
            baseline = self.performance_monitor['baseline_performance']
            current = self.performance_monitor['current_performance']

            if current < baseline * 0.8:  # 20% degradation
                self.performance_monitor['adaptation_needed'] = True

    def collect_adaptation_data(self, image, annotations):
        """Collect data for online adaptation"""
        self.adaptation_buffer.append({
            'image': image,
            'annotations': annotations,
            'timestamp': time.time()
        })

        # Keep buffer size manageable
        if len(self.adaptation_buffer) > 1000:
            self.adaptation_buffer = self.adaptation_buffer[-500:]  # Keep last 500 samples

    def trigger_adaptation(self):
        """Trigger online model adaptation"""
        if len(self.adaptation_buffer) >= self.adaptation_threshold:
            # Perform online fine-tuning
            adaptation_data = self.adaptation_buffer[-self.adaptation_threshold:]

            # Simplified adaptation process
            # In practice, this would involve actual model fine-tuning
            adapted_model = self.perform_domain_adaptation(adaptation_data)

            if adapted_model:
                self.base_model = adapted_model
                self.adaptation_buffer = []  # Clear buffer after adaptation
                print("Model successfully adapted to new domain")
                return True

        return False

    def perform_domain_adaptation(self, adaptation_data):
        """Perform domain adaptation on the model"""
        # This is a simplified placeholder
        # Real implementation would involve:
        # - Unsupervised domain adaptation techniques
        # - Few-shot learning
        # - Model fine-tuning with new data
        # - Transfer learning techniques
        pass
```

## Next Steps

After mastering computer vision integration:

1. Continue to [Sensor Fusion](./sensor-fusion.md) to learn about combining multiple sensor modalities
2. Practice implementing vision systems on your robot platform
3. Experiment with different vision models and optimization techniques
4. Test vision systems in both simulation and real environments

Your understanding of computer vision integration is now foundational for creating perception systems in the Physical AI and Humanoid Robotics course!
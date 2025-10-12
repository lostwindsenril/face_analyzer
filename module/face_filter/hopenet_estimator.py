#!/usr/bin/env python3
"""
HopeNet head pose estimator module.
"""


# Estimate head pose
# adjusted_bbox = draw_face_detection_result(
#     frame, face_info, x_offset, y_offset
# )
# pose_angles = self.hopenet_estimator.estimate_head_pose(frame, adjusted_bbox)
# deviation = self.hopenet_estimator.calculate_face_deviation(pose_angles)

# Draw full visualization results
# draw_complete_result(
#     frame, face_info, x_offset, y_offset, is_valid_quad, nose_inside,
#     inference_time, attempt_count, geometry_score, pose_angles, deviation
# )

import sys
import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
import torchvision
import numpy as np
import cv2
from PIL import Image
import logging

# Create logger
logger = logging.getLogger(__name__)

from config.settings import Config


class HopeNetEstimator:
    """HopeNet head pose estimator."""

    def __init__(self, enable=None, weights_path=None, device=None):
        """
        Initialize the HopeNet estimator.

        Args:
            enable (bool): Whether to enable HopeNet
            weights_path (str): Weights file path
            device (str): Device
        """
        self.enable = enable if enable is not None else Config.ENABLE_HOPENET
        self.weights_path = weights_path or Config.HOPENET_WEIGHTS_PATH
        self.device = device or Config.DEVICE

        self.model = None
        self.transformations = None
        self.idx_tensor = None
        self.available = False
        self.hopenet_module = None  # Keep hopenet module reference

        if self.enable:
            self.available = self._initialize_hopenet()
        else:
            pass

    def _initialize_hopenet(self):
        """Initialize the HopeNet model."""
        # Save current sys.path
        original_path = sys.path.copy()

        try:
            # Temporarily prepend HopeNet path
            sys.path.insert(0, Config.DEEP_HEAD_POSE_PATH)

            # Clear potential module cache to avoid conflicts
            modules_to_clear = ['utils', 'hopenet']
            for module_name in modules_to_clear:
                if module_name in sys.modules:
                    del sys.modules[module_name]

            # Use importlib for explicit imports
            import importlib.util

            # Import hopenet module
            hopenet_spec = importlib.util.spec_from_file_location(
                "hopenet", os.path.join(Config.DEEP_HEAD_POSE_PATH, "hopenet.py")
            )
            hopenet = importlib.util.module_from_spec(hopenet_spec)
            hopenet_spec.loader.exec_module(hopenet)

            # Import utils module
            utils_spec = importlib.util.spec_from_file_location(
                "hopenet_utils", os.path.join(Config.DEEP_HEAD_POSE_PATH, "utils.py")
            )
            utils = importlib.util.module_from_spec(utils_spec)
            utils_spec.loader.exec_module(utils)

            # Save module reference
            self.hopenet_module = hopenet

        finally:
            # Restore original sys.path
            sys.path = original_path


        # Initialize Hopenet model
        self.model = self.hopenet_module.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

        # Load model weights
        if os.path.exists(self.weights_path):
            saved_state_dict = torch.load(self.weights_path, map_location=self.device)
            self.model.load_state_dict(saved_state_dict)
        else:
            logger.warning("Hopenet权重文件未找到，使用随机权重")

        # Move model to device
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()

        # Set transformations
        self.transformations = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Set index tensor for pose calculation
        idx_list = [idx for idx in range(66)]
        if torch.cuda.is_available():
            self.idx_tensor = torch.FloatTensor(idx_list).cuda()
        else:
            self.idx_tensor = torch.FloatTensor(idx_list)

        return True

    def expand_bbox_dockerface_style(self, x_min, y_min, x_max, y_max, img_width, img_height, expansion=50):
        """Expand the bounding box using dockerface style."""
        # Apply fixed-pixel expansion
        x_min -= expansion
        x_max += expansion
        y_min -= expansion
        y_max += int(expansion * 0.6)  # Smaller bottom expansion

        # Keep coordinates within image bounds
        x_min = max(x_min, 0)
        y_min = max(y_min, 0)
        x_max = min(img_width, x_max)
        y_max = min(img_height, y_max)

        return x_min, y_min, x_max, y_max

    def estimate_head_pose(self, image, bbox):
        """
        Estimate head pose.

        Args:
            image: Input image
            bbox: Face bounding box [x1, y1, x2, y2]

        Returns:
            dict: Pose angles {'yaw': float, 'pitch': float, 'roll': float}
        """
        if not self.enable or not self.available:
            return None

        if self.model is None or self.transformations is None:
            return None

        try:
            x_min, y_min, x_max, y_max = map(int, bbox)
            img_height, img_width = image.shape[:2]

            # Expand bounding box
            expanded_x_min, expanded_y_min, expanded_x_max, expanded_y_max = self.expand_bbox_dockerface_style(
                x_min, y_min, x_max, y_max, img_width, img_height
            )

            # Crop face region
            face_img = image[expanded_y_min:expanded_y_max, expanded_x_min:expanded_x_max]

            if face_img.size == 0:
                return None

            # Convert to RGB for PIL
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_rgb)

            # Apply transforms
            transformed_img = self.transformations(face_pil)
            img_shape = transformed_img.size()
            transformed_img = transformed_img.view(1, img_shape[0], img_shape[1], img_shape[2])

            # Move to device
            if torch.cuda.is_available():
                transformed_img = Variable(transformed_img).cuda()
            else:
                transformed_img = Variable(transformed_img)

            # Forward pass through Hopenet
            with torch.no_grad():
                yaw, pitch, roll = self.model(transformed_img)

                # Apply softmax and get predictions
                yaw_predicted = F.softmax(yaw, dim=1)
                pitch_predicted = F.softmax(pitch, dim=1)
                roll_predicted = F.softmax(roll, dim=1)

                # Get continuous predictions (degrees)
                yaw_predicted = torch.sum(yaw_predicted.data[0] * self.idx_tensor) * 3 - 99
                pitch_predicted = torch.sum(pitch_predicted.data[0] * self.idx_tensor) * 3 - 99
                roll_predicted = torch.sum(roll_predicted.data[0] * self.idx_tensor) * 3 - 99

                # Convert to floats
                yaw_deg = float(yaw_predicted)
                pitch_deg = float(pitch_predicted)
                roll_deg = float(roll_predicted)

                return {
                    'yaw': yaw_deg,
                    'pitch': pitch_deg,
                    'roll': roll_deg
                }

        except Exception as e:
            logger.error(f"模型推理错误: 头部姿态估计失败: {e}")
            return None

    def calculate_face_deviation(self, pose_angles):
        """
        Calculate face deviation angle from screen normal.

        Args:
            pose_angles (dict): Pose angles dict

        Returns:
            float: Deviation angle
        """
        if not self.enable or pose_angles is None:
            return None

        yaw = pose_angles['yaw']
        pitch = pose_angles['pitch']
        roll = pose_angles['roll']

        # Compute deviation from screen normal (Z axis).
        # For a frontal face: yaw≈0, pitch≈0, roll≈0
        deviation = np.sqrt(yaw**2 + pitch**2 + roll**2)

        # Clamp to a reasonable range
        deviation = min(deviation, 90.0)

        return deviation

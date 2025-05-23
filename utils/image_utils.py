#!/usr/bin/env python3
"""
Image processing utilities.
"""

import cv2
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


def parse_resolution(resolution: str) -> Tuple[int, int]:
    """
    Parse a resolution string.

    Args:
        resolution: Resolution string like "640x480" or "1920x1080"

    Returns:
        tuple: (width, height)

    Raises:
        ValueError: If the resolution format is invalid
    """
    try:
        parts = resolution.lower().split('x')
        if len(parts) != 2:
            raise ValueError("分辨率格式应为 'WIDTHxHEIGHT'")

        width = int(parts[0])
        height = int(parts[1])

        if width <= 0 or height <= 0:
            raise ValueError("宽度和高度必须大于0")

        if width > 7680 or height > 4320:  # 8K limit
            raise ValueError("分辨率不能超过8K (7680x4320)")

        return width, height
    except (ValueError, IndexError) as e:
        raise ValueError(f"无效的分辨率格式 '{resolution}': {e}")


def crop_frame_to_aspect_ratio(frame: np.ndarray, target_aspect_ratio: float) -> np.ndarray:
    """
    Crop a frame to a target aspect ratio (center crop).

    Args:
        frame: Input frame
        target_aspect_ratio: Target aspect ratio

    Returns:
        np.ndarray: Cropped frame
    """
    current_height, current_width = frame.shape[:2]
    current_aspect_ratio = current_width / current_height

    # If aspect ratio already matches, return directly
    if abs(current_aspect_ratio - target_aspect_ratio) < 0.001:
        return frame

    if current_aspect_ratio > target_aspect_ratio:
        # Frame is wider; crop width
        new_width = int(current_height * target_aspect_ratio)
        new_height = current_height
        x_offset = (current_width - new_width) // 2
        y_offset = 0
    else:
        # Frame is taller; crop height
        new_width = current_width
        new_height = int(current_width / target_aspect_ratio)
        x_offset = 0
        y_offset = (current_height - new_height) // 2

    # Perform center crop
    cropped_frame = frame[y_offset:y_offset + new_height, x_offset:x_offset + new_width]

    return cropped_frame


def resize_frame(frame: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """
    Resize a frame.

    Args:
        frame: Input frame
        target_width: Target width
        target_height: Target height

    Returns:
        np.ndarray: Resized frame
    """
    current_height, current_width = frame.shape[:2]

    # If target size matches current, return
    if current_width == target_width and current_height == target_height:
        return frame

    # Use high-quality interpolation
    if target_width * target_height > current_width * current_height:
        # Upscale with CUBIC interpolation
        interpolation = cv2.INTER_CUBIC
    else:
        # Downscale with AREA interpolation
        interpolation = cv2.INTER_AREA

    resized_frame = cv2.resize(frame, (target_width, target_height), interpolation=interpolation)

    return resized_frame


def resize_frame_with_aspect_ratio(frame: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """
    Resize a frame; center crop first if aspect ratio differs.

    Args:
        frame: Input frame
        target_width: Target width
        target_height: Target height

    Returns:
        np.ndarray: Processed frame
    """
    current_height, current_width = frame.shape[:2]

    # If target size matches current, return
    if current_width == target_width and current_height == target_height:
        return frame

    # Compute target aspect ratio
    target_aspect_ratio = target_width / target_height

    # Center crop to match aspect ratio
    cropped_frame = crop_frame_to_aspect_ratio(frame, target_aspect_ratio)

    # Resize to target dimensions
    final_frame = resize_frame(cropped_frame, target_width, target_height)

    return final_frame

#!/usr/bin/env python3
"""
Single WebP image generation module.
"""

import os
import cv2
import numpy as np
from typing import Optional
from PIL import Image


from config.settings import Config
from utils.image_utils import parse_resolution, resize_frame_with_aspect_ratio
from module.video_processors import DecordVideoProcessor

from loguru import logger

def generate_single_webp(video_path: str, result, output_path: str,
                        resolution: Optional[str] = None) -> bool:
    """
    Generate a single WebP image from video processing results.

    Args:
        video_path: Video file path
        result: FaceProcessor result
        output_path: Output WebP file path
        resolution: Optional resolution like "640x480"

    Returns:
        bool: Whether generation succeeded
    """
    # Compute primary_frame_idx from results
    if result.valid_frames:
        # Get the first frame index with the highest score
        highest_score = max(result.valid_frames.keys())
        primary_frame_idx = result.valid_frames[highest_score][0]
    else:
        logger.warning(f"{video_path}没有有效人脸帧，从第0帧开始生成单帧图片")
        primary_frame_idx = 0

    # Use DecordVideoProcessor to read the specified frame
    try:
        processor = DecordVideoProcessor()
        frame = processor.get_frame_by_index(video_path, primary_frame_idx)
    except Exception as e:
        logger.error(f"无法读取视频帧: {video_path}, 帧索引: {primary_frame_idx}, 错误: {e}")
        return False

    # Save frame as WebP
    return save_frame_as_webp(frame, output_path, resolution)


def save_frame_as_webp(frame: np.ndarray, output_path: str, 
                      resolution: Optional[str] = None) -> bool:
    """
    Save a frame as WebP.

    Args:
        frame: Input frame
        output_path: Output path
        resolution: Optional resolution like "640x480"

    Returns:
        bool: Whether save succeeded
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Parse resolution and resize frame
    if resolution:
        target_width, target_height = parse_resolution(resolution)
        frame = resize_frame_with_aspect_ratio(frame, target_width, target_height)

    # Convert to RGB (required by PIL)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image
    pil_image = Image.fromarray(frame_rgb)

    # Save as WebP
    pil_image.save(
        output_path,
        'WEBP',
        quality=Config.WEBP_QUALITY,
        lossless=Config.WEBP_LOSSLESS
    )

    return True



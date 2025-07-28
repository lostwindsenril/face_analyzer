#!/usr/bin/env python3
"""
WebP animation generation module.
"""

import os
import cv2
import numpy as np
from typing import List, Optional
from PIL import Image

from config.settings import Config
from utils.image_utils import parse_resolution, resize_frame_with_aspect_ratio
from module.video_processors import DecordVideoProcessor

from loguru import logger


def get_video_fps(video_path: str) -> float:
    """
    Get the video's actual FPS.

    Args:
        video_path: Video file path

    Returns:
        float: Video FPS, or 0 if unavailable
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0.0

        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        # Validate FPS range
        if fps <= 0 or fps > 120:
            return 0.0

        return fps

    except Exception as e:
        logger.error(f"视频编解码错误: 无法获取视频FPS {video_path}: {e}")
        return 0.0


def generate_webp_animation(video_path: str, result, output_path: str,
                           duration_frames: Optional[int] = None,
                           resolution: Optional[str] = None) -> bool:
    """
    Generate a WebP animation from video processing results.

    Args:
        video_path: Video file path
        result: FaceProcessor result
        output_path: Output WebP animation path
        duration_frames: Frame count; if None, compute from FPS
        resolution: Optional resolution like "640x480"

    Returns:
        bool: Whether generation succeeded
    """
    # Compute start_frame_idx from results
    if not result.valid_frames:
        logger.warning(f"{video_path}没有有效人脸帧，从第0帧开始生成动画")
        start_frame_idx = 0
        # return False
    else:
        # Get the first frame index with the highest score
        highest_score = max(result.valid_frames.keys())
        start_frame_idx = result.valid_frames[highest_score][0]

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Get video FPS dynamically
    video_fps = get_video_fps(video_path)
    if video_fps <= 0:
        return False

    # Compute animation frame count (based on actual FPS)
    if duration_frames is None:
        duration_frames = int(Config.WEBP_ANIMATION_DURATION_SEC * video_fps)

    # Collect animation frames
    frames = collect_animation_frames(video_path, start_frame_idx, duration_frames, resolution)

    if not frames:
        return False

    # Convert to PIL Images
    pil_images = []
    for frame in frames:
        # Convert to RGB (required by PIL)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        pil_images.append(pil_image)

    # Compute per-frame duration (ms) using actual FPS
    frame_duration = int(1000 / video_fps)

    # Save as WebP animation
    pil_images[0].save(
        output_path,
        'WEBP',
        save_all=True,
        append_images=pil_images[1:],
        duration=frame_duration,
        loop=Config.WEBP_ANIMATION_LOOP,
        quality=Config.WEBP_QUALITY,
        lossless=Config.WEBP_LOSSLESS
    )

    return True


def collect_animation_frames(video_path: str, start_frame_idx: int,
                           duration_frames: int, resolution: Optional[str] = None) -> List[np.ndarray]:
    """
    Collect animation frames (continuous, preserve original playback speed).

    Args:
        video_path: Video file path
        start_frame_idx: Start frame index
        duration_frames: Frames to collect (based on actual FPS)
        resolution: Optional resolution

    Returns:
        List[np.ndarray]: Collected frames
    """
    frames = []

    # Create DecordVideoProcessor instance
    try:
        processor = DecordVideoProcessor()
    except Exception as e:
        logger.error(f"无法创建DecordVideoProcessor: {e}")
        return frames

    # Use DecordVideoProcessor to get total frames (for bounds checking)
    try:
        video_info = processor.get_video_info(video_path)
        total_frames = video_info['total_frames']
    except Exception as e:
        logger.error(f"无法获取视频信息: {video_path}, 错误: {e}")
        return frames

    # Compute actual frames available
    max_available_frames = total_frames - start_frame_idx
    actual_frames = min(duration_frames, max_available_frames)

    if actual_frames <= 0:
        logger.warning(f"无有效帧可收集: start_frame_idx={start_frame_idx}, total_frames={total_frames}")
        return frames

    # Collect frames sequentially with get_frame_by_index
    for i in range(actual_frames):
        current_frame_idx = start_frame_idx + i

        try:
            # Read frame with DecordVideoProcessor.get_frame_by_index
            frame = processor.get_frame_by_index(video_path, current_frame_idx)

            # Resize frame
            if resolution:
                target_width, target_height = parse_resolution(resolution)
                frame = resize_frame_with_aspect_ratio(frame, target_width, target_height)
            elif Config.WEBP_DEFAULT_RESOLUTION:
                target_width, target_height = parse_resolution(Config.WEBP_DEFAULT_RESOLUTION)
                frame = resize_frame_with_aspect_ratio(frame, target_width, target_height)

            frames.append(frame)

        except (IndexError, IOError, ValueError) as e:
            logger.warning(f"读取帧 {current_frame_idx} 失败: {e}")
            # Stop on errors and return collected frames
            break
        except Exception as e:
            logger.error(f"读取帧 {current_frame_idx} 时发生未知错误: {e}")
            break

    logger.info(f"成功收集 {len(frames)} 帧，从帧 {start_frame_idx} 开始")
    return frames


def calculate_animation_frames(video_fps: float) -> int:
    """
    Calculate animation frame count from video FPS (configured duration).

    Args:
        video_fps: Video FPS

    Returns:
        int: Animation frame count
    """
    # Compute frames from FPS and configured duration; no frame skipping.
    animation_frames = int(Config.WEBP_ANIMATION_DURATION_SEC * video_fps)

    return max(1, animation_frames)



#!/usr/bin/env python3
"""
FaceProcessor helper module.

Contains helper functions refactored from FaceProcessor:
- Debug visualization
- CUDA error handling
"""

import os
import re
import time
import cv2
import torch
from loguru import logger
from config.settings import Config


def init_debug_visualization():
    """
    Initialize debug visualization.

    Creates debug output directory and logs status.
    """
    if Config.DEBUG_FACE_VISUALIZATION:
        # Create debug output directory
        os.makedirs(Config.DEBUG_OUTPUT_DIR, exist_ok=True)
        logger.info(f"调试可视化已启用，输出目录: {Config.DEBUG_OUTPUT_DIR}")


def save_debug_visualization(frame, face_results, frame_number):
    """
    Save debug visualization image.

    Args:
        frame: Original frame
        face_results: Face detection results
        frame_number: Frame index
    """
    if not Config.DEBUG_FACE_VISUALIZATION or not face_results:
        return

    try:
        # Copy image to avoid modifying original
        debug_frame = frame.copy()

        # Draw face detection results
        for face_result in face_results:
            # Get bounding box coords
            bbox = face_result.bbox
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            # Draw bounding box
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2),
                        (0, 255, 0), 2)  # Green box, thickness 2

            # Draw confidence score
            confidence = face_result.confidence
            score_text = f"{confidence:.3f}"
            text_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX,
                                      0.6, 2)[0]  # Scale 0.6, thickness 2

            # Compute text position (above bbox)
            text_x = x1
            text_y = y1 - 10 if y1 - 10 > text_size[1] else y1 + text_size[1] + 10

            # Draw text background
            cv2.rectangle(debug_frame,
                        (text_x, text_y - text_size[1] - 5),
                        (text_x + text_size[0] + 5, text_y + 5),
                        (0, 0, 0), -1)

            # Draw confidence text
            cv2.putText(debug_frame, score_text, (text_x + 2, text_y - 2),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                      (0, 255, 255), 2)  # Yellow text, scale 0.6, thickness 2

            # Draw face landmarks (5 points)
            if hasattr(face_result, 'landmarks') and face_result.landmarks is not None:
                landmarks = face_result.landmarks.reshape(-1, 2)
                for landmark in landmarks:
                    landmark_x, landmark_y = int(landmark[0]), int(landmark[1])
                    cv2.circle(debug_frame, (landmark_x, landmark_y),
                             3, (255, 0, 0), -1)  # Blue keypoint, radius 3

        # Save debug image
        debug_filename = f"face_frame_{frame_number:06d}.jpg"
        debug_filepath = os.path.join(Config.DEBUG_OUTPUT_DIR, debug_filename)

        # Save with high quality
        cv2.imwrite(debug_filepath, debug_frame,
                   [cv2.IMWRITE_JPEG_QUALITY, 95])  # JPEG quality 95

        logger.debug(f"调试可视化已保存: {debug_filepath}")

    except Exception as e:
        # Error handling: do not break main processing flow
        logger.warning(f"调试可视化保存失败 (帧 {frame_number}): {e}")
        # Do not raise to keep main flow running


def is_cuda_oom_error(error: Exception) -> bool:
    """
    Detect CUDA out-of-memory errors.

    Args:
        error (Exception): Exception object

    Returns:
        bool: Whether it is a CUDA OOM error
    """
    error_str = str(error).lower()
    oom_patterns = [
        r'out of memory',
        r'cuda out of memory',
        r'cuda error: out of memory',
        r'runtime error.*out of memory',
        r'allocation failed',
        r'cuda_error_out_of_memory'
    ]
    
    for pattern in oom_patterns:
        if re.search(pattern, error_str):
            return True
    return False


def handle_cuda_oom_retry(frame_index: int, retry_count: int, error: Exception, thread_id: str) -> bool:
    """
    Handle CUDA OOM retry logic.

    Args:
        frame_index (int): Current frame index
        retry_count (int): Current retry count
        error (Exception): Exception object
        thread_id (str): Thread identifier

    Returns:
        bool: Whether to continue retrying
    """
    if retry_count >= Config.CUDA_OOM_RETRY_COUNT:
        logger.error(f"[Consumer {thread_id}] 帧 {frame_index}: CUDA OOM重试次数已用尽 ({Config.CUDA_OOM_RETRY_COUNT}次)，跳过该帧")
        logger.error(f"[Consumer {thread_id}] 最终错误: {str(error)}")
        return False
    
    logger.warning(f"[Consumer {thread_id}] 帧 {frame_index}: CUDA OOM错误，第 {retry_count + 1}/{Config.CUDA_OOM_RETRY_COUNT} 次重试")
    logger.debug(f"[Consumer {thread_id}] CUDA OOM错误详情: {str(error)}")
    
    # Clear CUDA cache
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug(f"[Consumer {thread_id}] CUDA缓存已清理")
    except Exception as cache_error:
        logger.warning(f"[Consumer {thread_id}] CUDA缓存清理失败: {str(cache_error)}")
    
    # Wait before retry
    logger.debug(f"[Consumer {thread_id}] 等待 {Config.CUDA_OOM_RETRY_DELAY} 秒后重试...")
    time.sleep(Config.CUDA_OOM_RETRY_DELAY)
    
    return True

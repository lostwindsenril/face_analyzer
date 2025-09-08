#!/usr/bin/env python3
"""
API helper module.
Handles video analysis and WebP generation, decoupled from HTTP layer.
"""

import cv2
from typing import Optional, Dict, Any

from core.face_processor import FaceProcessor, process_video_with_segmentation
from task.webp_animation import generate_webp_animation
from task.webp_single import generate_single_webp
from task.cache_manager import FaceAnalyzerCacheManager, process_with_cache

from loguru import logger

def process_video_with_cache(cache_manager: FaceAnalyzerCacheManager,
                           input_path: str, operation_name: str = "视频分析"):
    """
    Unified video processing flow with cache check/process/write.

    Args:
        face_processor: Face processor instance
        cache_manager: Cache manager instance
        input_path: Input video path
        operation_name: Operation name for logging

    Returns:
        ProcessResult: Result object, or None on failure

    """
    # === Cache check flow (reuse BatchManager flow) ===
    cached_result = cache_manager.get_cached_result(input_path)
    if cached_result:
        result = process_with_cache(input_path, cached_result)
    else:
        # === Video processing flow (reuse BatchManager flow) ===
        logger.info(f"开始{operation_name}: {input_path}")
        result = process_video_with_segmentation(input_path)
        if result.success:
            # === Cache write flow (reuse BatchManager flow) ===
            cache_manager.cache_result(input_path, result.valid_frames)

    return result

def analyze_and_generate_webp_animation(cache_manager: FaceAnalyzerCacheManager,
                                      input_path: str, output_path: str,
                                      resolution: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyze video and generate WebP animation (webp_generator compatible).

    Args:
        face_processor: Face processor instance
        cache_manager: Cache manager instance
        input_path: Input video path
        output_path: Output WebP path
        resolution: Optional resolution like "640x480"

    Returns:
        dict: Response with success flag and result
    """
    # Use unified processing flow (reuse BatchManager flow)
    result = process_video_with_cache(cache_manager, input_path, "视频处理")

    # Check success status first
    if not result.success:
        error_msg = f"视频处理失败: {input_path}"
        if hasattr(result, 'error') and result.error:
            error_msg += f", 错误: {result.error}"
        logger.error(error_msg)
        return {"success": False, "error": "视频处理失败"}

    # Then check valid_frames
    if not result.valid_frames:
        logger.warning(f"未找到有效帧: {input_path}")

    # Generate WebP animation (frame index and resolution handled internally)
    success = generate_webp_animation(input_path, result, output_path, resolution=resolution)
    if not success:
        logger.error(f"WebP动画生成失败: {output_path}")
        return {"success": False, "error": "WebP动画生成失败"}

    # Return webp_generator compatible format
    return {
        "success": True,
        "output_path": output_path,
    }

def analyze_and_generate_single_webp(cache_manager: FaceAnalyzerCacheManager,
                                    input_path: str, output_path: str,
                                    resolution: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyze video and generate a single-frame WebP (webp_generator compatible).

    Args:
        face_processor: Face processor instance
        cache_manager: Cache manager instance
        input_path: Input video path
        output_path: Output WebP path
        resolution: Optional resolution like "640x480"

    Returns:
        dict: Response with success flag and result
    """
    # Use unified processing flow (reuse BatchManager flow)
    result = process_video_with_cache(cache_manager, input_path, "视频处理(单帧)")

    # Check success status first
    if not result.success:
        error_msg = f"视频处理失败: {input_path}"
        if hasattr(result, 'error') and result.error:
            error_msg += f", 错误: {result.error}"
        logger.error(error_msg)
        return {"success": False, "error": "视频处理失败"}

    # Then check valid_frames
    if not result.valid_frames:
        logger.warning(f"未找到有效帧: {input_path}")

    # Generate single-frame WebP (frame index and resolution handled internally)
    success = generate_single_webp(input_path, result, output_path, resolution=resolution)
    if not success:
        logger.error(f"单帧WebP生成失败: {output_path}")
        return {"success": False, "error": "单帧WebP生成失败"}

    # Return webp_generator compatible format
    return {
        "success": True,
        "output_path": output_path
    }

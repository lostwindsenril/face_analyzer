#!/usr/bin/env python3
"""
Batch manager module.
"""

import os
from tqdm import tqdm

from core.face_processor import process_video_with_segmentation
from utils.file_service import (
    collect_video_files, ensure_directory_exists, get_output_path
)
from utils.logging_service import ResultStatus
from task import (
    FaceAnalyzerCacheManager, process_with_cache,
    generate_single_webp, generate_webp_animation, generate_video_preview
)
from config.settings import Config

from loguru import logger


class BatchManager:
    """Batch manager for coordinating bulk tasks."""

    def __init__(self):
        """
        Initialize the batch manager.
        """

        # Initialize cache manager (per config)
        if Config.CACHE_ENABLED:
            self.cache_manager = FaceAnalyzerCacheManager()
        else:
            self.cache_manager = None

    def process_videos(self, input_base: str, output_base: str):
        """
        Process video files in batch.

        Args:
            input_base (str): Input base path
            output_base (str): Output base path
        """
        # Initialize services
        result_status = ResultStatus(output_base)

        # Collect video files
        video_paths = collect_video_files(input_base)

        # Ensure output directory exists
        ensure_directory_exists(output_base)

        # Load already processed files
        processed_set = result_status.get_processed()

        # Filter processed videos
        pending_paths = [p for p in video_paths if p not in processed_set]

        # Process videos
        for input_path in tqdm(pending_paths, desc="Processing videos with geometry validation", unit="file"):
            # Get output path and base filename
            output_dir, base_filename = get_output_path(input_path, input_base, output_base)
            ensure_directory_exists(output_dir)

            # === Cache check flow (before FaceProcessor) ===
            cached_result = None
            if self.cache_manager:
                cached_result = self.cache_manager.get_cached_result(input_path)

            if cached_result:
                # Use cached result, skip FaceProcessor
                result = process_with_cache(input_path, cached_result)
            else:
                # Process single video with FaceProcessor
                result = process_video_with_segmentation(input_path)

                # === Cache write flow (after FaceProcessor) ===
                if self.cache_manager and result.success:
                    # Cache frame indices grouped by score
                    self.cache_manager.cache_result(input_path, result.valid_frames)

            # Check if processing succeeded
            if not result.success:
                logger.error(f"视频处理失败: {input_path}")
                error_message = ""
                if hasattr(result, 'error') and result.error:
                    error_message = result.error
                    logger.error(f"错误详情: {result.error}")
                result_status.log_error(input_path, error_message)
                continue

            # Generate single WebP
            if Config.WEBP_SINGLE_ENABLED:
                webp_single_path = os.path.join(output_dir, base_filename + Config.WEBP_OUTPUT_SINGLE_SUFFIX + ".webp")
                generate_single_webp(input_path, result, webp_single_path)

            # Generate WebP animation
            if Config.WEBP_ANIMATION_ENABLED:
                webp_anim_path = os.path.join(output_dir, base_filename + Config.WEBP_OUTPUT_ANIMATION_SUFFIX + ".webp")
                generate_webp_animation(input_path, result, webp_anim_path)

            # === Video preview generation (after WebP) ===
            if result.valid_frames and Config.VIDEO_PREVIEW_ENABLED:
                video_preview_path = os.path.join(output_dir, base_filename + Config.VIDEO_PREVIEW_OUTPUT_SUFFIX + "." + Config.VIDEO_PREVIEW_OUTPUT_FORMAT)
                generate_video_preview(input_path, result, video_preview_path)

            # Mark as processed
            result_status.log_processed(input_path)

            # Log results
            result_status.log_success(input_path)

            if result.valid_frames:
                # Use highest-scoring frame indices for display
                highest_score = max(result.valid_frames.keys())
                frame_indices = result.valid_frames[highest_score]
                cache_info = " (cached)" if cached_result else ""
                logger.info(f"找到有效的帧索引：{frame_indices}{cache_info}")
            else:
                logger.warning(f"未找到有效几何形状")
            

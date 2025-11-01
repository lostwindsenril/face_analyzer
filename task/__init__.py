#!/usr/bin/env python3
"""
Cache task module.
"""

from .cache_manager import (
    FaceAnalyzerCacheManager,
    process_with_cache,
    extract_and_save_cached_frame
)
from .webp_single import generate_single_webp, save_frame_as_webp
from .webp_animation import generate_webp_animation, collect_animation_frames
from .video_preview_generator import (
    VideoPreviewGenerator, KeyFrameSelector, VideoSegment, generate_video_preview
)

__all__ = [
    'FaceAnalyzerCacheManager',
    'process_with_cache',
    'extract_and_save_cached_frame',
    'generate_single_webp',
    'save_frame_as_webp',
    'generate_webp_animation',
    'collect_animation_frames',
    'VideoPreviewGenerator',
    'KeyFrameSelector',
    'VideoSegment',
    'generate_video_preview'
]

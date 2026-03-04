#!/usr/bin/env python3
"""
Core processor module.
"""

from module.video_processors import BaseVideoProcessor, CV2VideoProcessor, DecordVideoProcessor, MVExtractorVideoProcessor, create_video_processor
from .face_processor import FaceProcessor
from .batch_manager import BatchManager
from .api_manager import create_app
from .api_helper import analyze_and_generate_webp_animation, analyze_and_generate_single_webp, process_video_with_cache

__all__ = [
    'BaseVideoProcessor',
    'CV2VideoProcessor',
    'DecordVideoProcessor',
    'MVExtractorVideoProcessor',
    'create_video_processor',
    'FaceProcessor',
    'BatchManager',
    'create_app',
    'analyze_and_generate_webp_animation',
    'analyze_and_generate_single_webp',
    'process_video_with_cache'
]

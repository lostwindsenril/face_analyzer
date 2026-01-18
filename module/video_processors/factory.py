#!/usr/bin/env python3
"""
Video processor factory module.
"""

from .base import BaseVideoProcessor
from .cv2_processor import CV2VideoProcessor
from .decord_processor import DecordVideoProcessor
from .mvextractor_processor import MVExtractorVideoProcessor


def create_video_processor(processor_type: str = "decord") -> BaseVideoProcessor:
    """
    Factory function to create a video processor.

    Args:
        processor_type (str): Processor type: "cv2", "decord", or "mvextractor"

    Returns:
        BaseVideoProcessor: Video processor instance

    Raises:
        ValueError: Unsupported processor type
        ImportError: Required library unavailable
    """
    if processor_type.lower() == "cv2":
        return CV2VideoProcessor()
    elif processor_type.lower() == "decord":
        return DecordVideoProcessor()
    elif processor_type.lower() == "mvextractor":
        return MVExtractorVideoProcessor()
    else:
        raise ValueError(f"不支持的处理器类型: {processor_type}. 支持的类型: 'cv2', 'decord', 'mvextractor'")

#!/usr/bin/env python3
"""
Video processor package.

Provides multiple implementations:
- BaseVideoProcessor: Abstract base class
- CV2VideoProcessor: OpenCV-based implementation
- DecordVideoProcessor: Decord-based implementation
- MVExtractorVideoProcessor: MVExtractor-based implementation with motion vectors

Factory function:
- create_video_processor: Create a processor by type
"""

from .base import BaseVideoProcessor
from .cv2_processor import CV2VideoProcessor
from .decord_processor import DecordVideoProcessor
from .mvextractor_processor import MVExtractorVideoProcessor, MVEXTRACTOR_AVAILABLE
from .factory import create_video_processor

# Export public interfaces
__all__ = [
    'BaseVideoProcessor',
    'CV2VideoProcessor',
    'DecordVideoProcessor',
    'MVExtractorVideoProcessor',
    'create_video_processor',
    'MVEXTRACTOR_AVAILABLE'
]

# Version info
__version__ = '1.0.0'

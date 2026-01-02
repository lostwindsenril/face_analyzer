#!/usr/bin/env python3
"""
Video processor package.
"""

from .base import BaseVideoProcessor
from .cv2_processor import CV2VideoProcessor

__all__ = [
    'BaseVideoProcessor',
    'CV2VideoProcessor',
]

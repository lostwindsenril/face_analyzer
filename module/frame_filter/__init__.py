#!/usr/bin/env python3
"""
Frame filter module.
"""

from .base import BaseFrameFilter
from .quality_filter import QualityFrameFilter

__all__ = [
    'BaseFrameFilter',
    'QualityFrameFilter'
]

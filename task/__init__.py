#!/usr/bin/env python3
"""
Task module.
"""

from .webp_single import generate_single_webp, save_frame_as_webp
from .webp_animation import generate_webp_animation, collect_animation_frames

__all__ = [
    'generate_single_webp',
    'save_frame_as_webp',
    'generate_webp_animation',
    'collect_animation_frames',
]

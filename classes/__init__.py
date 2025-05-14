#!/usr/bin/env python3
"""
Data classes module.
"""

from .retinaface_result import RetinaFaceResult
from .process_result import ProcessResult
from .frame_processing_result import FrameProcessingResult, STOP_SIGNAL

__all__ = [
    'RetinaFaceResult',
    'ProcessResult',
    'FrameProcessingResult',
    'STOP_SIGNAL'
]

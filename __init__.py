#!/usr/bin/env python3
"""
Face analyzer package.
"""

from core.batch_manager import BatchManager
from classes.retinaface_result import RetinaFaceResult
from classes.process_result import ProcessResult

__version__ = "0.1.0"
__author__ = "Face Analyzer Team"

__all__ = [
    'BatchManager',
    'RetinaFaceResult',
    'ProcessResult'
]

#!/usr/bin/env python3
"""
Face analyzer package.
"""


from module.face_filter.hopenet_estimator import HopeNetEstimator
from core.batch_manager import BatchManager
from classes.retinaface_result import RetinaFaceResult
from classes.process_result import ProcessResult

__version__ = "1.0.0"
__author__ = "Face Analyzer Team"
__description__ = "基于RetinaFace的人脸几何形状验证与朝向分析工具"

__all__ = [
    'HopeNetEstimator',
    'BatchManager',
    'RetinaFaceResult',
    'ProcessResult'
]

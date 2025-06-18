#!/usr/bin/env python3
"""
Face filter abstract base module.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
import numpy as np

from classes.retinaface_result import RetinaFaceResult


class BaseFaceFilter(ABC):
    """Abstract base class for face filters."""
    
    @abstractmethod
    def __init__(self, **kwargs):
        """
        Initialize the filter.

        Args:
            **kwargs: Filter-specific init args
        """
        pass

    @abstractmethod
    def __call__(self, frame: np.ndarray, face_results: List[RetinaFaceResult]) -> Tuple[List[float], Dict[str, Any]]:
        """
        Main filter call.

        Args:
            frame (np.ndarray): Input frame
            face_results (List[RetinaFaceResult]): RetinaFace detection results

        Returns:
            Tuple[List[float], Dict[str, Any]]:
                - scores: Per-face scores aligned with face_results (1.0 pass, 0.0 fail)
                - filter_info: Filter debug info dict
        """
        pass



#!/usr/bin/env python3
"""
Scoring face filter module.
"""

from typing import List, Tuple, Dict, Any
import numpy as np

from .base import BaseFaceFilter
from classes.retinaface_result import RetinaFaceResult
from config.settings import Config


class ScoreFaceFilter(BaseFaceFilter):
    """Scoring face filter with confidence and size checks."""

    def __init__(self, **kwargs):
        """
        Initialize the base filter.

        Config parameters are imported from settings.py.

        Args:
            **kwargs: Reserved extension args
        """
        # Config parameters are imported from settings
        self.filter_name = "ScoreFaceFilter"

    def __call__(self, frame: np.ndarray, face_results: List[RetinaFaceResult]) -> Tuple[List[float], Dict[str, Any]]:
        """
        Apply base filtering (confidence + size).

        Args:
            frame (np.ndarray): Input frame
            face_results (List[RetinaFaceResult]): RetinaFace detection results

        Returns:
            Tuple[List[float], Dict[str, Any]]:
                - scores: Per-face scores (1.0 pass, 0.0 fail)
                - filter_info: Filter debug info dict
        """
        frame_height, frame_width = frame.shape[:2]

        scores = []
        filter_info = {
            'total_faces': len(face_results),
            'confidence_passed': 0,
            'size_passed': 0,
            'ratio_passed': 0,
            'final_passed': 0
        }

        for face_result in face_results:
            # Confidence filtering
            confidence_min_valid = face_result.confidence >= Config.MIN_CONFIDENCE
            confidence_valid = face_result.confidence >= Config.MAX_CONFIDENCE
            if confidence_min_valid:
                filter_info['confidence_passed'] += 1

            # Size filtering
            size_valid = (face_result.width > Config.MIN_FILTER_FACE_SIZE and
                         face_result.height > Config.MIN_FILTER_FACE_SIZE)
            if size_valid:
                filter_info['size_passed'] += 1

            width_ratio = face_result.width / frame_width
            height_ratio = face_result.height / frame_height
            ratio_valid = (width_ratio > Config.MIN_FILTER_FACE_RATIO and
                          height_ratio > Config.MIN_FILTER_FACE_RATIO)
            if ratio_valid:
                filter_info['ratio_passed'] += 1

            # Compute score for this face
            if size_valid and ratio_valid:
                if confidence_valid:
                    # Passed all filters, score 1.0
                    scores.append(1.0)
                    filter_info['final_passed'] += 1
                elif confidence_min_valid:
                    scores.append(0.5)
                    filter_info['final_passed'] += 1
                else:
                    # Failed filters, score 0.0
                    scores.append(0.0)
            else:
                # Failed filters, score 0.0
                scores.append(0.0)

        return scores, filter_info

#!/usr/bin/env python3
"""
Face geometry validator module.
"""

import traceback
from typing import List, Tuple, Dict, Any
import numpy as np
from shapely.geometry import Point, Polygon
import logging

# Create logger
logger = logging.getLogger(__name__)

from .base import BaseFaceFilter
from classes.retinaface_result import RetinaFaceResult


class GeometryFaceFilter(BaseFaceFilter):
    """Geometry-based face filter."""

    def __init__(self, **kwargs):
        """
        Initialize the geometry filter.

        Args:
            **kwargs: Reserved extension args
        """
        self.filter_name = "GeometryFaceFilter"

    def __call__(self, frame: np.ndarray, face_results: List[RetinaFaceResult]) -> Tuple[List[float], Dict[str, Any]]:
        """
        Apply geometry filtering.

        Args:
            frame (np.ndarray): Input frame
            face_results (List[RetinaFaceResult]): RetinaFace detection results

        Returns:
            Tuple[List[float], Dict[str, Any]]:
                - scores: Per-face scores (1.0 pass, 0.0 fail)
                - filter_info: Filter debug info dict
        """
        scores = []
        filter_info = {
            'total_faces': len(face_results),
            'quad_valid_count': 0,
            'nose_inside_count': 0,
            'final_passed': 0
        }

        for face_result in face_results:
            landmarks = face_result.landmarks
            quad_valid, nose_inside = self.validate_face_geometry(landmarks)

            if quad_valid:
                filter_info['quad_valid_count'] += 1
            if nose_inside:
                filter_info['nose_inside_count'] += 1

            # Compute score for this face
            if quad_valid and nose_inside:
                # Passed geometry validation, score 1.0
                scores.append(1.0)
                filter_info['final_passed'] += 1
            elif quad_valid:
                scores.append(0.5)
                filter_info['final_passed'] += 1
            else:
                # Failed geometry validation, score 0.0
                scores.append(0.0)

        return scores, filter_info


    def validate_face_geometry(self, landmarks):
        """
        Validate face geometry as a quadrilateral and check nose inclusion.

        Args:
            landmarks: RetinaFace landmarks

        Returns:
            tuple: (is_valid_quad, nose_inside)
        """
        if landmarks is None or len(landmarks) < 10:
            return False, False

        # Extract landmark coordinates
        left_eye = (landmarks[0], landmarks[1])
        right_eye = (landmarks[2], landmarks[3])
        nose = (landmarks[4], landmarks[5])
        left_mouth = (landmarks[6], landmarks[7])
        right_mouth = (landmarks[8], landmarks[9])

        # Build quad vertices
        quad_points = [left_eye, right_eye, right_mouth, left_mouth]

        try:
            # Create polygon
            polygon = Polygon(quad_points)

            # Check polygon validity
            quad_valid = polygon.is_valid

            if not quad_valid:
                return False, False

            # Check if nose is inside quad
            nose_point = Point(nose)
            nose_inside = polygon.contains(nose_point)

            return quad_valid, nose_inside

        except Exception as e:
            error_traceback = traceback.format_exc()
            error_message = f"几何验证错误: {str(e)}\n调用堆栈:\n{error_traceback}"
            logger.error(error_message)
            return False, False


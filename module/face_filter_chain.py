#!/usr/bin/env python3
"""
Face filter chain module.
"""

from typing import List, Dict, Any, Tuple
import numpy as np

# Create logger
from loguru import logger

from .face_filter.base import BaseFaceFilter
from .face_filter.score_filter import ScoreFaceFilter
from .face_filter.geometry_validator import GeometryFaceFilter
from classes.retinaface_result import RetinaFaceResult
from config.settings import Config


class FaceFilterChain:
    """Face filter chain to compose multiple filters."""
    
    def __init__(self, filters: List[BaseFaceFilter]):
        """
        Initialize the filter chain.

        Args:
            filters (List[BaseFaceFilter]): Filter list
        """
        self.filters = filters
        self.max_score = len(filters)  # Max score equals filter count
    
    def __call__(self, frame: np.ndarray, face_results: List[RetinaFaceResult]) -> Tuple[float, Dict[str, Any]]:
        """
        Apply all face filters and compute the highest score.

        Args:
            frame (np.ndarray): Input frame
            face_results (List[RetinaFaceResult]): RetinaFace detection results

        Returns:
            Tuple[float, Dict[str, Any]]:
                - highest_score: Highest score among faces
                - all_filter_info: Detailed info from all filters
        """
        # If no face results, return 0.0
        if not face_results:
            return 0.0, {}

        # If no filters, return 0.0 (no scoring criteria)
        if not self.filters:
            return 0.0, {}

        # Maintain cumulative score per face
        face_scores = [0.0] * len(face_results)
        all_filter_info = {}

        # Apply all filters in sequence
        for filter_instance in self.filters:
            scores, filter_info = filter_instance(frame, face_results)

            # Record info from each filter
            filter_key = filter_instance.filter_name
            all_filter_info[filter_key] = filter_info

            # Accumulate filter scores per face
            for i, score in enumerate(scores):
                face_scores[i] += score

        # Compute highest score among faces
        highest_score = max(face_scores) if face_scores else 0.0

        return highest_score, all_filter_info


def create_face_filter_chain() -> FaceFilterChain:
    """
    Initialize face filter chain from config.

    Returns:
        FaceFilterChain: Initialized filter chain
    """

    # Create filter instance map
    filter_map = {
        'BasicFilter': ScoreFaceFilter(),
        'GeometryFilter': GeometryFaceFilter()
    }

    # Build filter list from config
    filters = []
    for filter_name in Config.FILTER_CHAIN_COMPOSITION:
        if filter_name in filter_map:
            filters.append(filter_map[filter_name])
        else:
            logger.warning(f"未知的过滤器类型 '{filter_name}'，已跳过")

    # Always create a filter chain, even if empty
    return FaceFilterChain(filters)

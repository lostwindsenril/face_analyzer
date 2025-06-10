#!/usr/bin/env python3
"""
Frame filter chain module.
"""

from typing import Dict, Any, Tuple, List
import numpy as np

# Create logger
from loguru import logger

from .frame_filter.base import BaseFrameFilter
from .frame_filter.quality_filter import QualityFrameFilter
from config.settings import Config


class FrameFilterChain:
    """Frame filter chain to compose multiple frame filters."""
    
    def __init__(self, filters: List[BaseFrameFilter]):
        """
        Initialize the frame filter chain.

        Args:
            filters (List[BaseFrameFilter]): List of frame filters
        """
        self.filters = filters
    
    def __call__(self, frame: np.ndarray, frame_info: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Apply all frame filters sequentially.

        Args:
            frame (np.ndarray): Input frame
            frame_info (Dict[str, Any]): Frame info dict

        Returns:
            Tuple[bool, Dict[str, Any]]:
                - is_valid: Whether the frame passes all filters
                - filter_info: Detailed info from all filters
        """
        all_filter_info = {}
        overall_valid = True

        for filter_instance in self.filters:
            is_valid, filter_info = filter_instance(frame, frame_info)

            # Record info for each filter
            filter_key = filter_instance.filter_name
            all_filter_info[filter_key] = filter_info

            # If any filter fails, overall failure
            if not is_valid:
                overall_valid = False
                break

        return overall_valid, all_filter_info


def create_frame_filter_chain() -> FrameFilterChain:
    """
    Create a frame filter chain from config.

    Returns:
        FrameFilterChain: Configured frame filter chain
    """
    # Create filter instance map
    filter_map = {
        'QualityFilter': QualityFrameFilter()
    }

    # Build filter list from config
    filters = []
    for filter_name in Config.FRAME_FILTER_COMPOSITION:
        if filter_name in filter_map:
            filters.append(filter_map[filter_name])
        else:
            logger.warning(f"未知的帧过滤器类型 '{filter_name}'，已跳过")

    return FrameFilterChain(filters)

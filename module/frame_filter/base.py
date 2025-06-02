#!/usr/bin/env python3
"""
Frame filter base module.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np

class BaseFrameFilter(ABC):
    """Abstract base class for frame filters."""
    
    @abstractmethod
    def __init__(self, **kwargs):
        """
        Initialize a frame filter.

        Args:
            **kwargs: Filter-specific init args
        """
        pass

    @abstractmethod
    def __call__(self, frame: np.ndarray, frame_info: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Main frame filter call.

        Args:
            frame (np.ndarray): Input frame
            frame_info (Dict[str, Any]): Frame info dict (from video_processor)

        Returns:
            Tuple[bool, Dict[str, Any]]:
                - is_valid: Whether the frame passes the filter
                - filter_info: Filter debug info dict
        """
        pass



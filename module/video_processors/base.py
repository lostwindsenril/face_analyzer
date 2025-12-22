#!/usr/bin/env python3
"""
Abstract base module for video processors.
"""

import os
from abc import ABC, abstractmethod
from typing import Generator, Tuple, Dict, Any


class BaseVideoProcessor(ABC):
    """Abstract base class for video processors."""

    @abstractmethod
    def __init__(self):
        """Initialize the video processor."""
        pass

    @abstractmethod
    def __call__(self, video_path: str) -> Generator[Tuple[Any, Dict[str, Any]], None, None]:
        """
        Abstract method to extract frames from a video.

        Args:
            video_path (str): Video file path

        Yields:
            Tuple[Any, Dict[str, Any]]: (frame, video_info_dict)
                frame: Frame data
                video_info_dict: Video info dict containing at least:
                    - global_idx: Global frame index
                    - fps: Frame rate
                    - total_frames: Total frame count
        """
        pass

    @abstractmethod
    def get_frame_by_index(self, video_path: str, frame_index: int):
        """
        Return the video frame by index.

        Args:
            video_path (str): Video file path
            frame_index (int): Frame index (0-based)

        Returns:
            numpy.ndarray: Frame image data

        Raises:
            FileNotFoundError: Video file not found
            IOError: Unable to open video file
            IndexError: Frame index out of range
            ValueError: Negative frame index
        """
        pass

    @abstractmethod
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """
        Get basic video info.

        Args:
            video_path (str): Video file path

        Returns:
            Dict[str, Any]: Video info dict with:
                - total_frames (int): Total frames
                - fps (float): FPS
                - width (int): Width
                - height (int): Height
                - duration (float): Duration (seconds)

        Raises:
            FileNotFoundError: Video file not found
            IOError: Unable to open video file
        """
        pass

    def validate_video_path(self, video_path: str) -> bool:
        """
        Validate the video file path.

        Args:
            video_path (str): Video file path

        Returns:
            bool: Whether the path is valid
        """
        return os.path.exists(video_path) and os.path.isfile(video_path)

    def release_resources(self):
        """
        Release resources.

        Subclasses should override this to release specific resources.
        The method should be idempotent.
        """
        pass

    def __del__(self):
        """
        Destructor to ensure resources are released.
        """
        try:
            self.release_resources()
        except Exception:
            # Ignore exceptions in destructor to avoid crashes
            pass

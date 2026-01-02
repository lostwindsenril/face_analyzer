#!/usr/bin/env python3
"""
OpenCV-based video processor module.
"""

import cv2
import numpy as np
from typing import Generator, Tuple, Dict, Any
from config.settings import Config
from loguru import logger

from .base import BaseVideoProcessor


class CV2VideoProcessor(BaseVideoProcessor):
    """OpenCV-based video processor."""

    def __init__(self):
        pass

    def release_resources(self):
        """
        Release resources.

        CV2VideoProcessor uses a local VideoCapture in __call__,
        so no global resources to release, but keep API consistency.
        """
        pass

    def __call__(self, video_path: str) -> Generator[Tuple[Any, Dict[str, Any]], None, None]:
        """
        Extract frames from a video.

        Args:
            video_path (str): Video file path

        Yields:
            Tuple[Any, Dict[str, Any]]: (frame, video_info_dict)
        """
        if not self.validate_video_path(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        # Get basic video parameters
        video_info = self.get_video_info(video_path)
        total_frames = video_info['total_frames']
        fps = video_info['fps']

        sample_step = max(1, int(fps * Config.SAMPLE_INTERVAL_SEC))

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"无法打开视频: {video_path}")

        try:
            global_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if global_idx % sample_step == 0:
                    frame_info = {
                        'global_idx': global_idx,
                        'total_frames': total_frames,
                        'fps': fps
                    }
                    yield frame, frame_info

                global_idx += 1

                if global_idx >= total_frames:
                    break

        except Exception as e:
            logger.error(f"视频编解码错误: CV2视频处理失败: {e}")
            raise IOError(f"无法使用CV2处理视频: {video_path}, 错误: {e}")
        finally:
            cap.release()

    def get_frame_by_index(self, video_path: str, frame_index: int) -> np.ndarray:
        """
        Return a video frame by index.

        Args:
            video_path (str): Video file path
            frame_index (int): Frame index (0-based)

        Returns:
            np.ndarray: Frame image as (H, W, C) uint8 array

        Raises:
            FileNotFoundError: Video file not found
            IOError: Unable to open video file
            IndexError: Frame index out of range
            ValueError: Negative frame index
        """
        if not self.validate_video_path(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        if frame_index < 0:
            raise ValueError(f"帧号不能为负数: {frame_index}")

        # Get video info
        video_info = self.get_video_info(video_path)
        total_frames = video_info['total_frames']

        if frame_index >= total_frames:
            raise IndexError(f"帧号 {frame_index} 超出视频总帧数 {total_frames}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"无法打开视频: {video_path}")

        try:
            # Seek directly to the specified frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

            ret, frame = cap.read()
            if not ret:
                raise IOError(f"无法读取第 {frame_index} 帧")

            return frame

        except Exception as e:
            # Re-raise known exceptions, wrap unknown ones
            if isinstance(e, (IndexError, IOError)):
                raise
            else:
                raise IOError(f"读取帧时发生错误: {e}")
        finally:
            cap.release()

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

        Example:
            >>> processor = CV2VideoProcessor()
            >>> info = processor.get_video_info("video.mp4")
            >>> print(f"总帧数: {info['total_frames']}, 帧率: {info['fps']}")
        """
        if not self.validate_video_path(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"无法打开视频: {video_path}")

        try:
            # Get basic video info
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Compute video duration
            duration = total_frames / fps if fps > 0 else 0.0

            return {
                'total_frames': total_frames,
                'fps': fps,
                'width': width,
                'height': height,
                'duration': duration
            }

        except Exception as e:
            # Wrap unknown exceptions as IOError
            if isinstance(e, (FileNotFoundError, IOError)):
                raise
            else:
                raise IOError(f"获取视频信息时发生错误: {e}")
        finally:
            cap.release()

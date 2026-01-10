#!/usr/bin/env python3
"""
Decord-based video processor module.
"""

import cv2
import numpy as np
from typing import Generator, Tuple, Dict, Any
from config.settings import Config
from loguru import logger

from decord import VideoReader, cpu


from .base import BaseVideoProcessor


class DecordVideoProcessor(BaseVideoProcessor):
    """Decord-based video processor."""

    def __init__(self):
        pass

    def release_resources(self):
        """
        Release resources.

        DecordVideoProcessor uses a local VideoReader in __call__,
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

        try:
            # Open video with decord
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=16)

            for global_idx, frame in enumerate(vr):

                if global_idx % sample_step == 0:
                    # Read frame with decord
                    frame = frame.asnumpy()

                    # Convert RGB to BGR (decord returns RGB, OpenCV uses BGR)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    frame_info = {
                        'global_idx': global_idx,
                        'total_frames': total_frames,
                        'fps': fps
                    }

                    yield frame, frame_info

        except Exception as e:
            logger.error(f"视频编解码错误: Decord视频处理失败: {e}")
            raise IOError(f"无法使用Decord打开视频: {video_path}, 错误: {e}")

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

        try:
            # Open video with decord
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=16)

            # Read specified frame (decord supports random access)
            frame = vr[frame_index].asnumpy()

            # Convert RGB to BGR (decord returns RGB, OpenCV uses BGR)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            return frame

        except IndexError:
            # Re-raise IndexError to keep original message
            raise
        except ValueError:
            # Re-raise ValueError to keep original message
            raise
        except Exception as e:
            raise IOError(f"无法使用Decord读取第 {frame_index} 帧: {video_path}, 错误: {e}")

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
            >>> processor = DecordVideoProcessor()
            >>> info = processor.get_video_info("video.mp4")
            >>> print(f"总帧数: {info['total_frames']}, 帧率: {info['fps']}")
        """
        if not self.validate_video_path(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        try:
            # Open video with decord
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=16)

            # Get basic video info
            total_frames = len(vr)
            fps = float(vr.get_avg_fps())
            width = int(vr[0].shape[1])  # Width from first frame
            height = int(vr[0].shape[0])  # Height from first frame

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
            raise IOError(f"无法使用Decord获取视频信息: {video_path}, 错误: {e}")

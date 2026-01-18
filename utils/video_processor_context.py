#!/usr/bin/env python3
"""
Video processor context manager module.

Provides a context manager to ensure resources are released correctly.
"""

from typing import Generator, Any, Dict, Tuple
from loguru import logger


class VideoProcessorContext:
    """
    Video processor context manager.

    Ensures the video processor generator is closed properly,
    even on early exit (break, exceptions, etc.).
    """
    
    def __init__(self, video_processor, video_path: str):
        """
        Initialize the context manager.

        Args:
            video_processor: Video processor instance
            video_path: Video file path
        """
        self.video_processor = video_processor
        self.video_path = video_path
        self.generator = None
    
    def __enter__(self) -> Generator[Tuple[Any, Dict[str, Any]], None, None]:
        """
        Enter context and create the generator.

        Returns:
            Generator: Frame generator
        """
        self.generator = self.video_processor(self.video_path)
        return self.generator
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit context and ensure resources are released.

        Args:
            exc_type: Exception type
            exc_val: Exception value
            exc_tb: Exception traceback
        """
        try:
            # Close generator
            if self.generator is not None:
                self.generator.close()
        except GeneratorExit:
            # Normal generator closure
            pass
        except Exception as e:
            logger.error(f"关闭视频处理器Generator失败: {e}")
        
        try:
            # Release video processor resources
            if hasattr(self.video_processor, 'release_resources'):
                self.video_processor.release_resources()
        except Exception as e:
            logger.error(f"释放视频处理器资源失败: {e}")


def safe_video_processing(video_processor, video_path: str):
    """
    Safe video processing helper to ensure resources are released.

    Args:
        video_processor: Video processor instance
        video_path: Video file path

    Returns:
        VideoProcessorContext: Context manager

    Example:
        with safe_video_processing(processor, video_path) as video_iterator:
            for frame, frame_info in video_iterator:
                # Process frame
                if some_condition:
                    break  # Resources are still released on early exit
    """
    return VideoProcessorContext(video_processor, video_path)

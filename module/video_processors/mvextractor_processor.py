#!/usr/bin/env python3
"""
MVExtractor-based video processor module.
"""

import os
import cv2
import numpy as np
from typing import Generator, Tuple, Dict, Any
from config.settings import Config
from loguru import logger

try:
    from mvextractor.videocap import VideoCap
    MVEXTRACTOR_AVAILABLE = True
except ImportError:
    MVEXTRACTOR_AVAILABLE = False

from .cv2_processor import CV2VideoProcessor


class MVExtractorVideoProcessor(CV2VideoProcessor):
    """
    MVExtractor-based processor with motion vector extraction.

    Inherits CV2VideoProcessor to reuse video info and frame access,
    and adds motion-vector-based smart frame selection.
    """

    def __init__(self):
        if not MVEXTRACTOR_AVAILABLE:
            raise ImportError("MVExtractor library is not available. Please install it with: pip install motion-vector-extractor")

        # Call parent initializer
        super().__init__()

        # Initialize MVExtractor VideoCap instance
        self.cap = VideoCap()

        # Initialize EMA state
        self.motion_ema = None  # Motion exponential moving average
        self.ema_alpha = 0.4    # EMA smoothing factor

        # Initialize threshold configuration
        self.motion_threshold = Config.MOTION_THRESHOLD

        # Initialize debug video state
        self.debug_video_writer = None
        self.debug_video_path = None

        # Initialize debug visualization
        self._init_debug_visualization()

    def _init_debug_visualization(self):
        """Initialize debug visualization."""
        if Config.DEBUG_MVEXTRACTOR_VISUALIZATION:
            # Create debug output directories
            os.makedirs(Config.DEBUG_MVEXTRACTOR_OUTPUT_DIR, exist_ok=True)
            os.makedirs(Config.DEBUG_MVEXTRACTOR_VIDEO_OUTPUT_DIR, exist_ok=True)
            logger.info(f"MVExtractor调试可视化已启用，输出目录: {Config.DEBUG_MVEXTRACTOR_OUTPUT_DIR}")
            logger.info(f"MVExtractor调试视频输出目录: {Config.DEBUG_MVEXTRACTOR_VIDEO_OUTPUT_DIR}")

    def _init_debug_video(self, video_path, fps, width, height):
        """
        Initialize debug video writer.

        Args:
            video_path: Source video path
            fps: Video FPS
            width: Video width
            height: Video height
        """
        if not Config.DEBUG_MVEXTRACTOR_VISUALIZATION:
            return

        try:
            # Build debug video filename (keep base name, change extension)
            video_basename = os.path.splitext(os.path.basename(video_path))[0]
            debug_video_filename = f"{video_basename}.mp4"
            self.debug_video_path = os.path.join(Config.DEBUG_MVEXTRACTOR_VIDEO_OUTPUT_DIR, debug_video_filename)

            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.debug_video_writer = cv2.VideoWriter(
                self.debug_video_path, fourcc, fps, (width, height)
            )

            if not self.debug_video_writer.isOpened():
                raise IOError(f"无法创建调试视频文件: {self.debug_video_path}")

            logger.info(f"MVExtractor调试视频初始化成功: {self.debug_video_path}")

        except Exception as e:
            logger.error(f"初始化MVExtractor调试视频失败: {e}")
            self.debug_video_writer = None
            self.debug_video_path = None

    def _close_debug_video(self):
        """Close debug video writer."""
        if self.debug_video_writer is not None:
            try:
                self.debug_video_writer.release()
                logger.debug(f"MVExtractor调试视频已保存: {self.debug_video_path}")
            except Exception as e:
                logger.error(f"关闭MVExtractor调试视频失败: {e}")
            finally:
                self.debug_video_writer = None
                self.debug_video_path = None

    def _add_debug_frame(self, frame, motion_stats, motion_ema):
        """
        Add a debug frame to the video.

        Args:
            frame: Frame to add
            motion_stats: Motion statistics
            motion_ema: Motion EMA value
        """
        if not Config.DEBUG_MVEXTRACTOR_VISUALIZATION or self.debug_video_writer is None:
            return

        try:
            # Copy frame for debug visualization
            debug_frame = frame.copy()

            # Draw motion stats in the top-left (two lines)
            # Line 1: current motion magnitude
            motion_text = f"Motion: {motion_stats['mean_magnitude']:.3f}"
            cv2.putText(debug_frame, motion_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                       (0, 255, 255), 2)  # Yellow text, scale 0.7, thickness 2

            # Line 2: EMA (vertical spacing 25px)
            ema_text = f"EMA(3s): {motion_ema:.3f}"
            cv2.putText(debug_frame, ema_text, (10, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                       (0, 255, 255), 2)  # Yellow text, scale 0.7, thickness 2

            # Write to debug video
            self.debug_video_writer.write(debug_frame)

        except Exception as e:
            logger.error(f"添加MVExtractor调试帧失败: {e}")

    def _update_motion_ema(self, current_motion_magnitude):
        """
        Update the motion exponential moving average.

        Args:
            current_motion_magnitude (float): Current frame motion magnitude

        Returns:
            float: Updated EMA value
        """
        if self.motion_ema is None:
            # First frame, initialize EMA to current value
            self.motion_ema = current_motion_magnitude
        else:
            # EMA formula: EMA_new = alpha * current + (1-alpha) * EMA_old
            self.motion_ema = self.ema_alpha * current_motion_magnitude + (1 - self.ema_alpha) * self.motion_ema

        return self.motion_ema

    def release_resources(self):
        """
        Release all resources.

        This method is idempotent and safe to call multiple times.
        """
        try:
            # Close debug video writer
            self._close_debug_video()
        except Exception as e:
            logger.error(f"释放调试视频资源失败: {e}")

        try:
            # Release MVExtractor resources
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
        except Exception as e:
            logger.error(f"释放MVExtractor资源失败: {e}")

    def calculate_motion_statistics(self, motion_vectors, width, height):
        """
        Calculate motion vector statistics.

        Args:
            motion_vectors (numpy.ndarray): Motion vector array

        Returns:
            dict: Motion statistics
        """
        if len(motion_vectors) == 0:
            return {
                'mean_magnitude': 0.0,
            }

        # Compute magnitude for each motion vector
        mv = np.asarray(motion_vectors, dtype=float)
        w = mv[:, 1]
        h = mv[:, 2]
        mx = mv[:, 7]
        my = mv[:, 8]
        ms = mv[:, 9]
        dx = mx / ms
        dy = my / ms

        # Normalize by block area: sqrt(w*h)
        block_diag = np.sqrt(w * h)
        block_diag[block_diag == 0] = 1  # Avoid divide-by-zero
        normalized_mags = np.sqrt(dx**2 + dy**2) / block_diag

        # Optional: normalize by frame diagonal
        #if width is not None and height is not None:
            #frame_diag = np.sqrt(width*width + height*height)
            #if frame_diag > 0:
                #normalized_mags = normalized_mags / frame_diag
        k = 1
        med = np.median(normalized_mags)
        std = np.std(normalized_mags)
        score = (med + k * std) / 2.0
        # Clamp to [0,1]
        score = max(0.0, min(1.0, score))

        return {
            'mean_magnitude': score
        }

    def __call__(self, video_path: str) -> Generator[Tuple[Any, Dict[str, Any]], None, None]:
        """
        Extract frames and motion vectors from a video.

        Args:
            video_path (str): Video file path

        Yields:
            Tuple[Any, Dict[str, Any]]: (frame, video_info_dict)
        """
        try:
            # Reset EMA state for each new video
            self.motion_ema = None

            # Open video with mvextractor
            ret = self.cap.open(video_path)
            if not ret:
                raise IOError(f"无法使用MVExtractor打开视频: {video_path}")

            # Get video info
            video_info = self.get_video_info(video_path)
            total_frames = video_info['total_frames']
            fps = video_info['fps']
            width = video_info['width']
            height = video_info['height']

            # Initialize debug video
            self._init_debug_video(video_path, fps, width, height)

            # Smart frame selection parameters
            window_size = int(Config.SAMPLE_INTERVAL_SEC * fps)
            frame_idx = 0

            # Sliding window processing - read-and-compare
            while True:
                # Initialize best frame data for the current window
                best_frame_data = None
                min_motion_ema = float('inf')
                frames_in_window = 0

                # Collect a window of frames while comparing
                for _ in range(window_size):
                    ret, frame, motion_vectors, _, _ = self.cap.read()

                    if not ret:
                        # End of video
                        break

                    try:
                        motion_stats = self.calculate_motion_statistics(motion_vectors, width, height)

                        # Compute EMA
                        current_magnitude = motion_stats['mean_magnitude']
                        motion_ema = self._update_motion_ema(current_magnitude)

                        # Compare immediately with current best frame
                        if motion_ema < min_motion_ema:
                            min_motion_ema = motion_ema
                            best_frame_data = {
                                'frame': frame.copy(),  # Copy to avoid reference issues
                                'global_idx': frame_idx,
                                'motion_stats': motion_stats,
                                'motion_ema': motion_ema,
                            }

                        # Add each frame to debug video
                        self._add_debug_frame(frame, motion_stats, motion_ema)

                        frames_in_window += 1

                    except Exception as e:
                        logger.error(f"视频编解码错误: 处理第 {frame_idx} 帧失败: {e}")

                    frame_idx += 1

                # After window processing, check best frame validity
                if best_frame_data is not None:
                    # Threshold check for the selected frame
                    if best_frame_data['motion_ema'] < self.motion_threshold:
                        frame_info = {
                            'global_idx': best_frame_data['global_idx'],
                            'total_frames': total_frames,
                            'fps': fps,
                            'motion_statistics': best_frame_data['motion_stats'],
                            'motion_ema': best_frame_data['motion_ema'],
                        }
                        yield best_frame_data['frame'], frame_info
                else:
                    # No frames read; end of video
                    break

                # If fewer frames than window size, video ended
                if frames_in_window < window_size:
                    break

        except Exception as e:
            raise IOError(f"无法使用MVExtractor处理视频: {video_path}, 错误: {e}")
        finally:
            try:
                self.cap.release()
            except Exception as e:
                logger.error(f"视频编解码错误: MVExtractor 资源释放失败: {e}")

            # Close debug video
            self._close_debug_video()

#!/usr/bin/env python3
"""
Video preview generation module.
Select keyframes via K-means and build a preview video.
"""

import os
import cv2
import numpy as np
import tempfile
import subprocess

from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

# Create logger
from loguru import logger

from config.settings import Config
from classes.process_result import ProcessResult


@dataclass
class VideoSegment:
    """Video segment data class."""
    start_time: float  # Start time (seconds)
    end_time: float    # End time (seconds)
    key_frame_idx: int # Keyframe index
    key_frame_time: float  # Keyframe time (seconds)


class KeyFrameSelector:
    """K-means based keyframe selector."""
    
    def __init__(self, n_clusters: int = Config.VIDEO_PREVIEW_KMEANS_CLUSTERS):
        """
        Initialize the keyframe selector.

        Args:
            n_clusters: K-means cluster count
        """
        self.n_clusters = n_clusters
    
    def extract_frame_features_with_scores(self, video_path: str, valid_frames: Dict[float, List[int]]) -> Tuple[np.ndarray, List[Tuple[float, float, int]]]:
        """
        Extract frame features based on k-means_example.py uniform_face_selection.
        Use (normalized_time, quality_score) as 2D feature space.

        Args:
            video_path: Video file path
            valid_frames: FaceProcessor results grouped by score

        Returns:
            Feature matrix (n_frames, 2) and frame info list
        """
        # Get total frame count
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"无法打开视频文件: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Build feature tuples: [(t_i, q_i)]
        # t_i = normalized timestamp = frame_idx / total_frames
        # q_i = quality score
        feats: List[Tuple[float, float, int]] = []
        for score, frames in valid_frames.items():
            for frame_idx in frames:
                normalized_time = frame_idx / total_frames if total_frames > 0 else 0
                feats.append((normalized_time, float(score), frame_idx))

        # Build feature matrix X = [[t, q], ...]
        X = np.array([[t, q] for t, q, _ in feats])  # shape = (M, 2)

        return X, feats
    
    def select_key_frames(self, video_path: str, valid_frames: Dict[float, List[int]]) -> List[int]:
        """
        Select keyframes based on k-means_example.py uniform_face_selection.
        Cluster on (normalized_time, quality_score).

        Args:
            video_path: Video file path
            valid_frames: FaceProcessor results grouped by score

        Returns:
            Selected keyframe indices sorted by time
        """
        if not valid_frames:
            return []

        # Count total valid frames
        total_valid_frames = sum(len(indices) for indices in valid_frames.values())

        # If too few frames, return all frames (time-sorted)
        if total_valid_frames <= self.n_clusters:
            all_frames = []
            for score, indices in valid_frames.items():
                all_frames.extend(indices)
            return sorted(list(set(all_frames)))

        # Use improved feature extraction
        X, feats = self.extract_frame_features_with_scores(video_path, valid_frames)

        if len(X) == 0:
            return []

        # Run K-means clustering (aligned with k-means_example.py)
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        cluster_centers = kmeans.cluster_centers_

        # For each cluster, pick the frame closest to the center
        selected_frames = []
        for cluster_id in range(self.n_clusters):
            idx_in_cluster = np.where(cluster_labels == cluster_id)[0]
            if len(idx_in_cluster) == 0:
                continue

            # Compute L2 distance to center
            distances = np.linalg.norm(X[idx_in_cluster] - cluster_centers[cluster_id], axis=1)
            best_idx = idx_in_cluster[np.argmin(distances)]

            # Add original frame_idx to results
            selected_frames.append(feats[best_idx][2])  # feats[i] = (normalized_time, score, frame_idx)

        # Sort by time
        return sorted(selected_frames)


class VideoPreviewGenerator:
    """Video preview generator."""
    
    def __init__(self):
        """Initialize the video preview generator."""
        self.key_frame_selector = KeyFrameSelector()
    
    def get_video_info(self, video_path: str) -> Tuple[float, float]:
        """
        Get basic video info.

        Args:
            video_path: Video file path

        Returns:
            (fps, duration) tuple
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"无法打开视频文件: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        return fps, duration
    
    def frame_idx_to_time(self, frame_idx: int, fps: float) -> float:
        """
        Convert a frame index to time.

        Args:
            frame_idx: Frame index
            fps: Video FPS

        Returns:
            Time in seconds
        """
        return frame_idx / fps if fps > 0 else 0
    
    def calculate_segments(self, key_frame_indices: List[int], fps: float, duration: float) -> List[VideoSegment]:
        """
        Calculate video segments, handling overlaps and boundaries.

        Args:
            key_frame_indices: Keyframe indices
            fps: Video FPS
            duration: Total video duration

        Returns:
            Video segment list
        """
        if not key_frame_indices:
            return []
        
        segments = []
        
        for frame_idx in key_frame_indices:
            key_frame_time = self.frame_idx_to_time(frame_idx, fps)
            
            # Compute segment start/end
            start_time = max(0, key_frame_time - Config.VIDEO_PREVIEW_SEGMENT_PADDING)
            end_time = min(duration, key_frame_time + Config.VIDEO_PREVIEW_SEGMENT_PADDING)

            # Ensure segment length does not exceed configured max
            segment_duration = end_time - start_time
            if segment_duration > Config.VIDEO_PREVIEW_SEGMENT_DURATION:
                # If too long, crop around the keyframe
                half_duration = Config.VIDEO_PREVIEW_SEGMENT_DURATION / 2
                start_time = max(0, key_frame_time - half_duration)
                end_time = min(duration, key_frame_time + half_duration)
            
            segments.append(VideoSegment(
                start_time=start_time,
                end_time=end_time,
                key_frame_idx=frame_idx,
                key_frame_time=key_frame_time
            ))
        
        # Merge overlapping segments
        merged_segments = self._merge_overlapping_segments(segments)
        
        return merged_segments
    
    def _merge_overlapping_segments(self, segments: List[VideoSegment]) -> List[VideoSegment]:
        """
        Merge overlapping video segments.

        Args:
            segments: Original segment list

        Returns:
            Merged segment list
        """
        if not segments:
            return []
        
        # Sort by start time
        segments.sort(key=lambda x: x.start_time)
        
        merged = [segments[0]]
        
        for current in segments[1:]:
            last_merged = merged[-1]
            
            # Check overlap
            if current.start_time <= last_merged.end_time:
                # Merge segments
                merged[-1] = VideoSegment(
                    start_time=last_merged.start_time,
                    end_time=max(last_merged.end_time, current.end_time),
                    key_frame_idx=last_merged.key_frame_idx,  # Keep first keyframe
                    key_frame_time=last_merged.key_frame_time
                )
            else:
                # No overlap; add new segment
                merged.append(current)
        
        return merged

    def extract_video_segments(self, video_path: str, segments: List[VideoSegment], temp_dir: str) -> List[str]:
        """
        Extract video segments with FFmpeg.

        Args:
            video_path: Source video path
            segments: Video segment list
            temp_dir: Temp directory

        Returns:
            Extracted segment paths
        """
        segment_paths = []

        for i, segment in enumerate(segments):
            segment_path = os.path.join(temp_dir, f"segment_{i:03d}.{Config.VIDEO_PREVIEW_OUTPUT_FORMAT}")

            # Build FFmpeg command
            cmd = [
                'ffmpeg', '-y',  # -y overwrite output
                '-i', video_path,
                '-ss', str(segment.start_time),  # Start time
                '-t', str(segment.end_time - segment.start_time),  # Duration
                '-c:v', Config.VIDEO_PREVIEW_VIDEO_CODEC,  # Video codec
                '-c:a', Config.VIDEO_PREVIEW_AUDIO_CODEC,  # Audio codec
                '-crf', str(Config.VIDEO_PREVIEW_CRF),  # Quality
                '-preset', Config.VIDEO_PREVIEW_PRESET,  # Preset
                segment_path
            ]

            try:
                # Execute FFmpeg command
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                segment_paths.append(segment_path)
            except subprocess.CalledProcessError as e:
                print(f"警告: 提取片段 {i} 失败: {e}")
                print(f"FFmpeg错误输出: {e.stderr}")
                continue

        return segment_paths

    def concatenate_segments(self, segment_paths: List[str], output_path: str, temp_dir: str) -> bool:
        """
        Concatenate video segments with FFmpeg.

        Args:
            segment_paths: Segment file paths
            output_path: Output file path
            temp_dir: Temp directory

        Returns:
            Whether concatenation succeeded
        """
        if not segment_paths:
            return False

        # Create FFmpeg concat file
        concat_file = os.path.join(temp_dir, "concat_list.txt")
        with open(concat_file, 'w', encoding='utf-8') as f:
            for segment_path in segment_paths:
                f.write(f"file '{segment_path}'\n")

        # Build FFmpeg concat command
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_file,
            '-c', 'copy',  # Direct copy without re-encode
            output_path
        ]

        try:
            # Execute FFmpeg command
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"拼接视频失败: {e}")
            logger.error(f"FFmpeg错误输出: {e.stderr}")
            return False

    def generate_preview(self, video_path: str, process_result: ProcessResult, output_path: str) -> bool:
        """
        Generate a video preview.

        Args:
            video_path: Source video path
            process_result: FaceProcessor result
            output_path: Output preview video path

        Returns:
            Whether generation succeeded
        """
        if not Config.VIDEO_PREVIEW_ENABLED:
            return False

        if not process_result.valid_frames:
            logger.warning("没有有效帧，无法生成预览")
            return False

        try:
            # Get video info
            fps, duration = self.get_video_info(video_path)

            # Select keyframes
            key_frame_indices = self.key_frame_selector.select_key_frames(video_path, process_result.valid_frames)

            if not key_frame_indices:
                logger.warning("没有选中关键帧，无法生成预览")
                return False

            logger.info(f"选中 {len(key_frame_indices)} 个关键帧: {key_frame_indices}")

            # Compute video segments
            segments = self.calculate_segments(key_frame_indices, fps, duration)

            if not segments:
                logger.warning("没有有效片段，无法生成预览")
                return False

            logger.info(f"计算出 {len(segments)} 个视频片段")

            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract video segments
                segment_paths = self.extract_video_segments(video_path, segments, temp_dir)

                if not segment_paths:
                    logger.error("没有成功提取任何视频片段")
                    return False

                logger.info(f"成功提取 {len(segment_paths)} 个视频片段")

                # Ensure output directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # Concatenate segments
                success = self.concatenate_segments(segment_paths, output_path, temp_dir)

                if success:
                    logger.info(f"预览视频生成成功: {output_path}")
                    return True
                else:
                    logger.error("预览视频拼接失败")
                    return False

        except Exception as e:
            logger.error(f"生成预览视频时发生异常: {e}")
            import traceback
            traceback.print_exc()
            return False


def generate_video_preview(video_path: str, process_result: ProcessResult, output_path: str) -> bool:
    """
    Convenience wrapper to generate a video preview.

    Args:
        video_path: Source video path
        process_result: FaceProcessor result
        output_path: Full output file path

    Returns:
        Whether generation succeeded
    """
    if not Config.VIDEO_PREVIEW_ENABLED:
        return False

    # Create generator and build preview
    generator = VideoPreviewGenerator()
    return generator.generate_preview(video_path, process_result, output_path)

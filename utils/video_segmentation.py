#!/usr/bin/env python3
"""
Video segmentation and parallel processing module.
"""

import os
import tempfile
import subprocess
import multiprocessing
from typing import List

from classes.process_result import ProcessResult
from config.settings import Config
from loguru import logger




def get_video_duration(video_path: str) -> float:
    """
    Get video duration in seconds.

    Args:
        video_path (str): Video file path

    Returns:
        float: Video duration (seconds)
    """
    cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json',
        '-show_format', video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    import json
    data = json.loads(result.stdout)
    duration = float(data['format']['duration'])
    return duration

def process_long_video(face_processor, video_path: str, duration: float) -> ProcessResult:
    """
    Process long video by splitting and processing in parallel.

    Args:
        face_processor: Face processor instance
        video_path (str): Video file path
        duration (float): Video duration (seconds)

    Returns:
        ProcessResult: Merged result
    """
    segment_duration = duration / Config.VIDEO_SPLIT_SEGMENTS
    logger.info(f"将视频分割为{Config.VIDEO_SPLIT_SEGMENTS}个片段，每段 {segment_duration:.1f}s")

    with tempfile.TemporaryDirectory() as temp_dir:
        logger.debug(f"临时目录: {temp_dir}")

        # 1. Split video
        segment_paths = _split_video(video_path, temp_dir, segment_duration)

        # 2. Process in parallel
        results = _process_segments_parallel(face_processor, segment_paths)

        # 3. Merge results
        final_result = _merge_results(results, video_path, segment_duration)

        return final_result

def _split_video(video_path: str, temp_dir: str, segment_duration: float) -> List[str]:
    """
    Split video into segments using FFmpeg.

    Args:
        video_path (str): Source video path
        temp_dir (str): Temp directory
        segment_duration (float): Segment duration (seconds)

    Returns:
        List[str]: Segment file paths
    """
    segment_paths = []

    for i in range(Config.VIDEO_SPLIT_SEGMENTS):
        start_time = i * segment_duration
        output_path = os.path.join(temp_dir, f"segment_{i:02d}.mp4")

        cmd = [
            'ffmpeg', '-y', '-v', 'quiet',
            '-ss', str(start_time),
            '-t', str(segment_duration),
            '-i', video_path,
            '-c', 'copy',  # Stream copy to avoid re-encoding
            '-avoid_negative_ts', 'make_zero',
            output_path
        ]

        subprocess.run(cmd, check=True)
        segment_paths.append(output_path)
        logger.info(f"片段 {i+1}/{Config.VIDEO_SPLIT_SEGMENTS} 分割完成: {os.path.basename(output_path)}")

    logger.info(f"成功分割 {len(segment_paths)}/{Config.VIDEO_SPLIT_SEGMENTS} 个片段")
    return segment_paths

def _process_segments_parallel(face_processor, segment_paths: List[str]) -> List[ProcessResult]:
    """
    Process video segments in parallel.

    Args:
        segment_paths (List[str]): Segment paths

    Returns:
        List[ProcessResult]: Results list
    """
    if not segment_paths:
        return []

    # Use spawn start method to avoid CUDA issues
    multiprocessing.set_start_method('spawn', force=True)

    # Create process pool; max workers from config
    max_workers = Config.VIDEO_SPLIT_SEGMENTS
    logger.info(f"启动 {max_workers} 个并行进程处理 {len(segment_paths)} 个片段")

    # Prepare args: (segment_path, thread_id)
    args_list = [(path, f"seg_{i:02d}") for i, path in enumerate(segment_paths)]

    with multiprocessing.Pool(processes=max_workers) as pool:
        results = pool.starmap(face_processor.process_single_video, args_list)

    successful_results = [r for r in results if r.success]
    failed_count = len(results) - len(successful_results)

    logger.info(f"并行处理完成: {len(successful_results)} 成功, {failed_count} 失败")

    return results

def _merge_results(results: List[ProcessResult], original_video_path: str,
                  segment_duration: float) -> ProcessResult:
    """
    Merge processing results from segments.

    Args:
        results (List[ProcessResult]): Segment results
        original_video_path (str): Original video path
        segment_duration (float): Segment duration (seconds)

    Returns:
        ProcessResult: Final merged result
    """
    if not results:
        return ProcessResult(
            success=False,
            video_path=original_video_path,
            valid_frames={}
        )

    # Count successful results
    successful_results = [r for r in results if r.success]

    if not successful_results:
        return ProcessResult(
            success=False,
            video_path=original_video_path,
            error="没有可以合并的片段"
        )

    logger.info(f"合并 {len(successful_results)}/{len(results)} 个成功的片段结果")

    # Merge valid_frames and adjust frame indices
    merged_valid_frames = {}

    # Get original FPS to compute frame offset
    fps = _get_video_fps(original_video_path)

    for i, result in enumerate(successful_results):

        # Compute frame offset for this segment
        segment_start_time = i * segment_duration
        frame_offset = int(segment_start_time * fps)

        logger.debug(f"片段 {i}: 时间偏移 {segment_start_time:.1f}s, 帧偏移 {frame_offset}")

        # Adjust frame indices and merge
        for score, frame_indices in result.valid_frames.items():
            if score not in merged_valid_frames:
                merged_valid_frames[score] = []

            # Adjust frame indices
            adjusted_indices = [idx + frame_offset for idx in frame_indices]
            merged_valid_frames[score].extend(adjusted_indices)

    # Sort and deduplicate per score group
    for score in merged_valid_frames:
        merged_valid_frames[score] = sorted(list(set(merged_valid_frames[score])))

    # Summarize results
    total_frames = sum(len(indices) for indices in merged_valid_frames.values())
    logger.info(f"合并结果: 总共 {total_frames} 个有效帧")
    for score, indices in merged_valid_frames.items():
        logger.debug(f"  得分 {score}: {len(indices)} 个帧")

    return ProcessResult(
        success=True,
        video_path=original_video_path,
        valid_frames=merged_valid_frames
    )

def _get_video_fps(video_path: str) -> float:
    """
    Get video FPS.

    Args:
        video_path (str): Video file path

    Returns:
        float: Video FPS
    """
    cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json',
        '-show_streams', '-select_streams', 'v:0', video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    import json
    data = json.loads(result.stdout)

    # Get video stream FPS
    for stream in data.get('streams', []):
        if stream.get('codec_type') == 'video':
            fps_str = stream.get('r_frame_rate', '25/1')
            # Handle fractional FPS like "25/1"
            if '/' in fps_str:
                num, den = fps_str.split('/')
                fps = float(num) / float(den)
            else:
                fps = float(fps_str)
            return fps

    return 25.0

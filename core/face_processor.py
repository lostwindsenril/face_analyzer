#!/usr/bin/env python3
"""
Face processor module.
"""

import sys
import os
import threading
import time
import queue
import numpy as np


# Add project root to Python path for model imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.retinaface.detection import RetinaFaceDetection
from module.video_processors import create_video_processor
from classes.process_result import ProcessResult
from classes.frame_processing_result import FrameProcessingResult, STOP_SIGNAL
from classes.retinaface_result import RetinaFaceResult
from module.face_filter_chain import create_face_filter_chain
from module.frame_filter_chain import create_frame_filter_chain

from utils.image_utils import crop_frame_to_aspect_ratio
from config.settings import Config
from utils.video_segmentation import process_long_video, get_video_duration
from utils.video_processor_context import safe_video_processing
from core.face_processor_helper import (
    init_debug_visualization, save_debug_visualization,
    is_cuda_oom_error, handle_cuda_oom_retry
)

# Create logger
from loguru import logger


# Global detector removed; initialized within FaceProcessor


def process_video_with_segmentation(video_path: str) -> ProcessResult:
    """
    Process a video; split long videos for parallel processing.

    Args:
        face_processor: Face processor instance
        video_path (str): Video file path

    Returns:
        ProcessResult: Processing result
    """
    face_processor = FaceProcessor()
    # 1. Detect video duration
    duration = get_video_duration(video_path)

    # 2. Determine whether to split
    if duration <= 60 or Config.VIDEO_SPLIT_SEGMENTS <= 1:
        return face_processor.process_single_video(video_path, "single")

    # 3. Split long video for parallel processing
    return process_long_video(face_processor, video_path, duration)

class FaceProcessor:
    """Face processor handling a single video end-to-end."""

    def __init__(self):
        """
        Initialize face processor (lazy init).
        """
        self.detector = RetinaFaceDetection(Config.RETINAFACE_MODEL_PATH, device=Config.DEVICE)

        self.video_processor = create_video_processor(Config.VIDEO_PROCESSOR_TYPE)

        self.frame_filter_chain = create_frame_filter_chain()

        self.face_filter_chain = create_face_filter_chain()

        # Initialize debug visualization
        init_debug_visualization()


    def release_resources(self):
        """Release all resources."""
        try:
            if hasattr(self, 'video_processor') and self.video_processor:
                self.video_processor.release_resources()
        except Exception as e:
            logger.error(f"释放video_processor资源失败: {e}")

    def __del__(self):
        """Destructor to ensure resources are released."""
        try:
            self.release_resources()
        except Exception:
            # Ignore exceptions in destructor
            pass

    def detect_faces(self, frame):
        """
        Detect faces and convert to RetinaFaceResult instances.

        Args:
            frame: Input frame

        Returns:
            List[RetinaFaceResult]: List of RetinaFaceResult
        """
        input_data = {'img': frame}
        detection_result = self.detector(input_data)

        # Convert detections to RetinaFaceResult instances
        face_results = []

        if detection_result is not None and len(detection_result) == 2:
            dets, landms = detection_result

            if dets is not None and len(dets) > 0:
                for i in range(len(dets)):
                    det = dets[i]
                    if len(det) < 5:
                        continue

                    x1, y1, x2, y2, confidence = det[:5]

                    # Get corresponding landmarks
                    landmarks = None
                    if landms is not None and i < len(landms):
                        landmarks = landms[i]

                    if landmarks is not None:
                        # Create RetinaFaceResult instance
                        face_result = RetinaFaceResult(
                            face_id=i + 1,
                            confidence=float(confidence),
                            bbox=[float(x1), float(y1), float(x2), float(y2)],
                            landmarks=landmarks
                        )
                        face_results.append(face_result)

        return face_results

    def process_single_video(self, video_path: str, thread_id: str = "main") -> ProcessResult:
        """
        Full processing flow for a single video.

        Args:
            video_path (str): Video file path
            thread_id (str): Thread ID for logging

        Returns:
            ProcessResult: Processing result
        """
        # Create FrameProcessingResult instance
        processing_result = FrameProcessingResult()

        try:
            # Call the new find_valid_frame method
            self.find_valid_frame(video_path, processing_result, thread_id=thread_id)
            
            # Debug
            if not processing_result.valid_frames:
                MOTION_THRESHOLD_BAK, Config.MOTION_THRESHOLD = Config.MOTION_THRESHOLD, 0.6
                self.video_processor = create_video_processor(Config.VIDEO_PROCESSOR_TYPE)
                Config.MOTION_THRESHOLD = MOTION_THRESHOLD_BAK
                self.find_valid_frame(video_path, processing_result, thread_id=thread_id)

                if not processing_result.valid_frames:
                    MOTION_THRESHOLD_BAK, Config.MOTION_THRESHOLD = Config.MOTION_THRESHOLD, 1.0
                    self.video_processor = create_video_processor(Config.VIDEO_PROCESSOR_TYPE)
                    Config.MOTION_THRESHOLD = MOTION_THRESHOLD_BAK
                    self.find_valid_frame(video_path, processing_result, thread_id=thread_id)

        except Exception as e:
            logger.error(f"视频处理异常: {video_path}, 错误: {str(e)}")
            return ProcessResult(
                success=False,
                error=str(e)
            )

        # Build and return ProcessResult
        return ProcessResult(
            success=True,
            video_path=video_path,
            valid_frames=dict(processing_result.valid_frames)
        )

    def find_valid_frame(self, video_path: str, processing_result: FrameProcessingResult, thread_id: str = "main") -> None:
        """
        Find valid geometric faces in a video (single or full search).

        Args:
            video_path (str): Video file path
            processing_result (FrameProcessingResult): Result object to write into
            thread_id (str): Thread identifier

        Returns:
            None: Results are written into processing_result

        Note:
            Search mode controlled by Config.SEARCH_ALL:
            - False: Exit after first valid frame (default)
            - True: Search all valid frames in the video
        """
        producer_id = f"{thread_id}_producer"
        consumer_id = f"{thread_id}_consumer"

        # Create producer and consumer threads
        producer_thread = threading.Thread(
            target=self._frame_producer,
            args=(video_path, processing_result, producer_id)
        )
        consumer_thread = threading.Thread(
            target=self._frame_consumer,
            args=(processing_result, consumer_id)
        )

        # Start threads
        producer_thread.start()
        consumer_thread.start()

        # Wait for threads to finish
        producer_thread.join()
        consumer_thread.join()

    def _frame_producer(self, video_path: str, processing_result: FrameProcessingResult, thread_id: str):
        """
        Producer thread: read and preprocess video frames.

        Args:
            video_path (str): Video file path
            processing_result (FrameProcessingResult): Result object
            thread_id (str): Thread identifier
        """
        # Use context manager to ensure resources are released
        with safe_video_processing(self.video_processor, video_path) as video_iterator:
            for frame, frame_info in video_iterator:
                # Check stop signal
                if processing_result.stop_event.is_set():
                    break

                # Frame filter preprocessing
                is_valid, _ = self.frame_filter_chain(frame, frame_info)

                if not is_valid:
                    continue

                # Queue write
                frame_data = {
                    'frame': frame,
                    'frame_info': frame_info,
                    'is_stop_signal': False
                }

                # Non-blocking queue write with retry
                max_retries = 10
                retry_count = 0
                write_success = False

                while retry_count < max_retries:
                    try:
                        # Non-blocking write
                        processing_result.frame_queue.put_nowait(frame_data)
                        write_success = True
                        break  # Success, exit retry loop
                    except queue.Full:
                        # Check stop signal
                        if processing_result.stop_event.is_set():
                            return  # Exit producer immediately

                        retry_count += 1
                        time.sleep(1.0)  # Wait 1s before retry

                if not write_success:
                    # Handle retry failure
                    logger.warning(f"[Producer Debug] {thread_id} - 队列写入失败，跳过帧 {frame_info['global_idx']}，重试次数: {max_retries}")
                    continue  # Skip current frame

            # Send stop signal - non-blocking write with retry
            max_retries = 10
            retry_count = 0
            stop_signal_sent = False

            while retry_count < max_retries:
                try:
                    # Non-blocking stop signal write
                    processing_result.frame_queue.put_nowait(STOP_SIGNAL)
                    stop_signal_sent = True
                    break  # Success, exit retry loop
                except queue.Full:
                    retry_count += 1
                    time.sleep(1.0)  # Wait 1s before retry

            if not stop_signal_sent:
                # Stop signal failed
                logger.error(f"[Producer Debug] {thread_id} - 停止信号发送失败，重试次数: {max_retries}")

    def _frame_consumer(self, processing_result: FrameProcessingResult, thread_id: str):
        """
        Consumer thread: face detection/filtering with smart error handling.

        Args:
            processing_result (FrameProcessingResult): Result object
            thread_id (str): Thread identifier

        Note:
            Search mode controlled by Config.SEARCH_ALL
        """
        while True:
            frame_data = processing_result.frame_queue.get()

            # Check stop signal
            if frame_data['is_stop_signal']:
                break

            # Process a single frame with error handling
            success = self._process_single_frame_with_retry(
                frame_data, processing_result, thread_id
            )

            # If success triggered early exit, break
            if success and processing_result.stop_event.is_set():
                break

    def _process_single_frame_with_retry(self, frame_data: dict, processing_result: FrameProcessingResult,
                                       thread_id: str) -> bool:
        """
        Single-frame processing with retry.

        Args:
            frame_data (dict): Frame data
            processing_result (FrameProcessingResult): Result object
            thread_id (str): Thread identifier

        Returns:
            bool: Whether processing succeeded

        Note:
            Search mode controlled by Config.SEARCH_ALL
        """
        frame_index = frame_data['frame_info']['global_idx']
        retry_count = 0

        while retry_count <= Config.CUDA_OOM_RETRY_COUNT:
            try:
                # Crop image to square (1:1 aspect ratio)
                cropped_frame = crop_frame_to_aspect_ratio(frame_data['frame'], 1.0)

                # Face detection (conversion handled internally)
                face_results = self.detect_faces(cropped_frame)

                # Debug visualization: save detection results
                save_debug_visualization(cropped_frame, face_results, frame_index)

                if face_results:
                    # Filter with face filter chain
                    highest_score, _ = self.face_filter_chain(cropped_frame, face_results)

                    # Thread-safe add valid frames (grouped by score)
                    if highest_score >= Config.FACEFILTER_MIN_THRESHOLD:
                        with processing_result.lock:
                            processing_result.valid_frames[highest_score].append(frame_index)

                    # Early exit (non-full search only)
                    if highest_score >= Config.FACEFILTER_THRESHOLD and not Config.SEARCH_ALL:
                        processing_result.stop_event.set()

                # On success, log if retried
                if retry_count > 0:
                    logger.info(f"[Consumer {thread_id}] 帧 {frame_index}: 重试成功，共重试 {retry_count} 次")

                return True

            except Exception as e:
                # Check for CUDA OOM
                if is_cuda_oom_error(e):
                    # CUDA OOM, attempt retry
                    if handle_cuda_oom_retry(frame_index, retry_count, e, thread_id):
                        retry_count += 1
                        continue
                    else:
                        # Retries exhausted; skip this frame
                        return False
                else:
                    # Other errors; skip this frame
                    logger.error(f"[Consumer {thread_id}] 帧 {frame_index}: 处理失败，跳过该帧")
                    logger.error(f"[Consumer {thread_id}] 错误详情: {str(e)}")
                    return False

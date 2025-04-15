#!/usr/bin/env python3
"""
Global configuration for the face analyzer.
"""

import torch


class Config:
    """
    Face analyzer configuration class.
    Centralized management of all config variables.
    """

    # ============================================================================
    # Face detection configuration
    # ============================================================================

    # ============================================================================
    # BasicFilter configuration
    # ============================================================================
    MIN_CONFIDENCE = 0.97  # Minimum confidence threshold
    MAX_CONFIDENCE = 0.99
    MIN_FILTER_FACE_SIZE = 30.0  # BasicFilter minimum face size (pixels)
    MIN_FILTER_FACE_RATIO = 0.1  # BasicFilter minimum face-to-frame ratio

    # ============================================================================
    # HopeNet configuration
    # ============================================================================
    ENABLE_HOPENET = True  # Disabled by default; enable via CLI args
    HOPENET_WEIGHTS_PATH = '/home/shaohan/video_summerization/test_script/face_analyzer/models/hopenet/hopenet_robust_alpha1.pkl'
    DEEP_HEAD_POSE_PATH = '/home/shaohan/video_summerization/test_script/face_analyzer/models/hopenet'



    # ============================================================================
    # Debug visualization configuration
    # ============================================================================
    DEBUG_FACE_VISUALIZATION = False  # Face detection debug visualization toggle
    DEBUG_OUTPUT_DIR = "debug_output"  # Debug output directory
    DEBUG_MVEXTRACTOR_VISUALIZATION = False  # MVExtractor debug visualization toggle
    DEBUG_MVEXTRACTOR_OUTPUT_DIR = "debug_mvextractor_output"  # MVExtractor debug output dir
    DEBUG_MVEXTRACTOR_VIDEO_OUTPUT_DIR = "debug_mvextractor_videos"  # MVExtractor debug video dir


    # ============================================================================
    # Video processing configuration
    # ============================================================================
    MIN_BRIGHTNESS = 10      # Minimum brightness threshold
    MIN_CONTRAST = 5         # Minimum contrast threshold
    MOTION_THRESHOLD = 0.3
    # Video processor configuration
    VIDEO_PROCESSOR_TYPE = "mvextractor"  # Processor type: "decord", "cv2", "mvextractor"
                                    # "decord": use Decord, faster, more formats
                                    # "cv2": use OpenCV, better compatibility, no extra deps
                                    # "mvextractor": use MVExtractor with motion vectors

    # ============================================================================
    # Device configuration
    # ============================================================================
    @classmethod
    def _get_device(cls):
        """Get device configuration dynamically."""
        try:
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            print(f"❌ 设备配置错误: {str(e)}\n调用堆栈:\n{error_traceback}")
            return 'cpu'

    DEVICE = _get_device.__func__(None)  # Call classmethod to get device

    # ============================================================================
    # Model path configuration
    # ============================================================================
    RETINAFACE_MODEL_PATH = 'models/retinaface/pytorch_model.pt'



    # ============================================================================
    # Filter chain configuration
    # ============================================================================
    FILTER_CHAIN_COMPOSITION = ['BasicFilter', 'GeometryFilter']  # Filters in order

    # ============================================================================
    # Frame filter configuration
    # ============================================================================
    FRAME_FILTER_COMPOSITION = ['QualityFilter']  # Frame filter chain order

    # ============================================================================
    # Video sampling configuration
    # ============================================================================
    SAMPLE_INTERVAL_SEC = 3.0  # Sampling interval (seconds)

    # ============================================================================
    # Video processing configuration
    # ============================================================================
    SEARCH_ALL = False  # Search all frames (True: full search, False: early exit)

    # ============================================================================
    # Cache configuration
    # ============================================================================
    CACHE_ENABLED = True  # Enable caching
    CACHE_FILE = "face_analysis_cache.json"  # Cache file path

    # ============================================================================
    # WebP feature configuration
    # ============================================================================
    WEBP_ENABLED = True  # Global WebP toggle
    WEBP_SINGLE_ENABLED = True  # Single-frame WebP toggle
    WEBP_ANIMATION_ENABLED = True  # Animated WebP toggle

    # WebP generation parameters
    WEBP_QUALITY = 80  # WebP quality (0-100)
    WEBP_LOSSLESS = False  # Use lossless compression
    WEBP_ANIMATION_DURATION_SEC = 3.0  # Animation duration (seconds)
    WEBP_ANIMATION_LOOP = 0  # Loop count (0 = infinite)
    WEBP_DEFAULT_RESOLUTION = "640x480"  # Default resolution

    # WebP output naming
    WEBP_OUTPUT_SINGLE_SUFFIX = "_frame"  # Single-frame suffix
    WEBP_OUTPUT_ANIMATION_SUFFIX = "_anim"  # Animation suffix

    # ============================================================================
    # Face filter scoring configuration
    # ============================================================================
    FACEFILTER_MIN_THRESHOLD = 0.5
    FACEFILTER_THRESHOLD = 1.5  # Score threshold for face filtering

    # ============================================================================
    # Video splitting and parallel processing
    # ============================================================================
    VIDEO_SPLIT_SEGMENTS = 1  # Segments; also number of parallel processes

    # ============================================================================
    # Video preview generation configuration
    # ============================================================================
    VIDEO_PREVIEW_ENABLED = True  # Global video preview toggle
    VIDEO_PREVIEW_SEGMENT_DURATION = 1.0  # Segment duration per keyframe (seconds)
    VIDEO_PREVIEW_SEGMENT_PADDING = 0.5  # Padding before/after keyframe (seconds)
    VIDEO_PREVIEW_KMEANS_CLUSTERS = 30  # K-means clusters (keyframe count)
    VIDEO_PREVIEW_OUTPUT_SUFFIX = "_preview"  # Preview output suffix
    VIDEO_PREVIEW_OUTPUT_FORMAT = "mp4"  # Preview output format
    VIDEO_PREVIEW_VIDEO_CODEC = "libx264"  # Video codec
    VIDEO_PREVIEW_AUDIO_CODEC = "aac"  # Audio codec
    VIDEO_PREVIEW_CRF = 23  # CRF quality (18-28, lower is better)
    VIDEO_PREVIEW_PRESET = "medium"  # Preset (ultrafast, fast, medium, slow, veryslow)

    # ============================================================================
    # HTTP API service configuration
    # ============================================================================
    HTTP_ENABLED = True
    HTTP_HOST = '0.0.0.0'
    HTTP_PORT = 5000
    HTTP_DEBUG = False

    # API configuration
    API_VERSION = "1.0.0"
    API_SERVICE_NAME = "Face Analyzer API"

    # Request limits
    HTTP_MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    HTTP_REQUEST_TIMEOUT = 3600  # 5 minutes

    # Concurrency control
    CONCURRENCY_LIMIT = 8  # Global max concurrent requests
    BATCH_PROCESSING_MAX_WORKERS = 8  # Max concurrent batch workers

    # CUDA error handling
    CUDA_OOM_RETRY_COUNT = 10  # CUDA OOM retry count
    CUDA_OOM_RETRY_DELAY = 10  # CUDA OOM retry interval (seconds)

    # Logging configuration
    LOG_OUTPUT_MODE = "both"  # Options: "screen", "file", "both", "none"
    LOG_LEVEL = "INFO"  # Options: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"

    # Result status log file configuration
    RESULT_SUCCESS_LOG = "success_log.txt"
    RESULT_ERROR_LOG = "error_log.txt"
    RESULT_PROCESSED_LOG = "processed_results.log"

    SUPPORTED_VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']

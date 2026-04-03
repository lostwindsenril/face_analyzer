# Configuration Guide

All configuration is centralized in `config/settings.py` inside the `Config`
class. This document describes every tunable parameter.

## Face Detection

| Parameter            | Default | Description |
|----------------------|---------|-------------|
| `MIN_CONFIDENCE`     | `0.97`  | Minimum detection confidence score to accept a face. |
| `MAX_CONFIDENCE`     | `0.99`  | Upper confidence bound (used for score normalization). |
| `MIN_FILTER_FACE_SIZE` | `30.0` | Minimum face bounding box size in pixels. |
| `MIN_FILTER_FACE_RATIO` | `0.1` | Minimum ratio of face area to frame area. |

## HopeNet Head Pose Estimation

| Parameter              | Default           | Description |
|------------------------|-------------------|-------------|
| `ENABLE_HOPENET`       | `True`            | Enable head pose filtering. |
| `HOPENET_WEIGHTS_PATH` | *(see settings)*  | Path to HopeNet `.pkl` weights. |
| `DEEP_HEAD_POSE_PATH`  | *(see settings)*  | Path to the HopeNet model directory. |

## Video Processing

| Parameter                | Default        | Description |
|--------------------------|----------------|-------------|
| `VIDEO_PROCESSOR_TYPE`   | `"mvextractor"`| Which video processor to use: `cv2`, `decord`, or `mvextractor`. |
| `SAMPLE_INTERVAL_SEC`    | `3.0`          | Seconds between sampled frames. |
| `SEARCH_ALL`             | `False`        | If `False`, stop after finding the first valid frame. |
| `MOTION_THRESHOLD`       | `0.3`          | Motion vector magnitude threshold (MVExtractor only). |

## Frame Quality Filters

| Parameter        | Default | Description |
|------------------|---------|-------------|
| `MIN_BRIGHTNESS` | `10`    | Minimum average brightness to accept a frame. |
| `MIN_CONTRAST`   | `5`     | Minimum standard deviation of pixel intensities. |

## WebP Output

| Parameter                     | Default | Description |
|-------------------------------|---------|-------------|
| `WEBP_QUALITY`                | `80`    | WebP compression quality (0–100). |
| `WEBP_ANIMATION_DURATION_SEC` | `3.0`   | Duration in seconds for animated WebP. |

## Video Preview

| Parameter                         | Default | Description |
|-----------------------------------|---------|-------------|
| `VIDEO_PREVIEW_KMEANS_CLUSTERS`   | `30`    | Number of K-means clusters for keyframe selection. |
| `VIDEO_PREVIEW_SEGMENT_DURATION`  | `1.0`   | Duration of each segment (seconds) in the preview. |

## API & Concurrency

| Parameter          | Default | Description |
|--------------------|---------|-------------|
| `HTTP_PORT`        | `5000`  | Port for the Flask server. |
| `CONCURRENCY_LIMIT`| `8`    | Maximum concurrent requests handled by the API. |

## CUDA / GPU

| Parameter              | Default  | Description |
|------------------------|----------|-------------|
| `DEVICE`               | `"cuda"` | Torch device. Auto-detected; falls back to `cpu`. |
| `CUDA_OOM_RETRY_COUNT` | `10`    | Number of retries on CUDA out-of-memory errors. |
| `CUDA_OOM_RETRY_DELAY` | `10`    | Seconds to wait between OOM retries. |

## Debug Visualization

| Parameter                            | Default                       | Description |
|--------------------------------------|-------------------------------|-------------|
| `DEBUG_FACE_VISUALIZATION`           | `False`                       | Draw detection overlays and save debug images. |
| `DEBUG_OUTPUT_DIR`                   | `"debug_output"`              | Directory for face visualization output. |
| `DEBUG_MVEXTRACTOR_VISUALIZATION`    | `False`                       | Visualize motion vectors. |
| `DEBUG_MVEXTRACTOR_OUTPUT_DIR`       | `"debug_mvextractor_output"`  | Directory for MVExtractor debug frames. |
| `DEBUG_MVEXTRACTOR_VIDEO_OUTPUT_DIR` | `"debug_mvextractor_videos"`  | Directory for MVExtractor debug videos. |

## Cache

| Parameter    | Default                       | Description |
|--------------|-------------------------------|-------------|
| Cache path   | `"face_analysis_cache.json"`  | TinyDB JSON file for persistent cache. |
| Flush interval | `8` seconds                 | Background thread flush period. |

## Environment Variables

An optional `.env` file can be placed in the project root. Docker Compose
configurations reference it for volume mounts, port mappings, and model
paths. See `.env.example` for available variables.

# Architecture Overview

This document describes the high-level architecture of Face Analyzer, the design
decisions behind each component, and how data flows through the system.

## System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Entry Points                            │
│   main.py (CLI)  ──►  local batch  /  HTTP API server           │
└──────────┬────────────────────────────────┬─────────────────────┘
           │                                │
           ▼                                ▼
  ┌─────────────────┐            ┌─────────────────────┐
  │  BatchManager    │            │  APIManager (Flask)  │
  │  (core/)         │            │  (core/)             │
  └───────┬─────────┘            └───────┬─────────────┘
          │                              │
          ▼                              ▼
  ┌──────────────────────────────────────────────┐
  │              APIHelper (core/)                │
  │  Unified business logic for both modes        │
  │  Cache lookup ─► Process ─► Generate output   │
  └──────────────────────┬───────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
  ┌─────────────┐ ┌───────────┐ ┌───────────────┐
  │  Cache       │ │  Face      │ │  Task          │
  │  Manager     │ │  Processor │ │  Generators    │
  │  (task/)     │ │  (core/)   │ │  (task/)       │
  └─────────────┘ └─────┬─────┘ └───────────────┘
                        │
            ┌───────────┼───────────┐
            ▼           ▼           ▼
    ┌────────────┐ ┌─────────┐ ┌──────────┐
    │ Video      │ │ Frame   │ │ Face     │
    │ Processors │ │ Filter  │ │ Filter   │
    │ (module/)  │ │ Chain   │ │ Chain    │
    └────────────┘ └─────────┘ └──────────┘
```

## Component Descriptions

### Entry Points (`main.py`)

The CLI entry point supports two modes:

- **`local`** — Batch-process a directory of video files. The
  `BatchManager` walks the input directory, processes each video through
  `FaceProcessor`, and writes output artifacts.
- **`server`** — Start a Flask HTTP server. `APIManager` exposes two
  endpoints (`/analyze`, `/analyze_frame`) that accept JSON payloads and
  delegate to `APIHelper`.

### Core Processing Layer (`core/`)

| Module                  | Responsibility |
|-------------------------|----------------|
| `face_processor.py`     | End-to-end processing for a single video: read frames, detect faces, filter, and collect results. Uses a producer–consumer threading model. |
| `batch_manager.py`      | Iterates over a directory of videos and calls `FaceProcessor` for each, collecting results and logging summaries. |
| `api_manager.py`        | Flask route definitions. Thin HTTP layer — all logic is delegated to `APIHelper`. |
| `api_helper.py`         | Shared business logic used by both local and API modes. Handles cache lookup, invokes `FaceProcessor`, and triggers output generation. |
| `face_processor_helper.py` | Debug visualization and error reporting helpers for `FaceProcessor`. |

### Filter Chains (`module/`)

Face Analyzer uses the **Chain of Responsibility** pattern for filtering.

#### Face Filters (`module/face_filter/`)

Each filter extends `BaseFaceFilter` and implements a `filter()` method that
returns a boolean.

| Filter                | Purpose |
|-----------------------|---------|
| `ScoreFaceFilter`     | Rejects faces below a confidence threshold or smaller than a minimum pixel size / ratio. |
| `GeometryFaceFilter`  | Validates that the five facial landmarks form a valid quadrilateral and that the nose is contained within it. Uses Shapely for geometric tests. |
| `HopenetEstimator`    | (Optional) Estimates head pose via HopeNet and rejects extreme yaw/pitch/roll. |

`FaceFilterChain` composes these filters and applies them in sequence.

#### Frame Filters (`module/frame_filter/`)

| Filter           | Purpose |
|------------------|---------|
| `QualityFilter`  | Checks brightness, contrast, and sharpness thresholds before frames enter the detection pipeline. |

`FrameFilterChain` composes frame-level filters.

### Video Processors (`module/video_processors/`)

An abstract `BaseVideoProcessor` defines the interface. Three concrete
implementations are provided:

| Processor                  | Backend     | Notes |
|----------------------------|-------------|-------|
| `CV2VideoProcessor`        | OpenCV      | Simple, synchronous frame reading. |
| `DecordVideoProcessor`     | Decord      | Efficient GPU-accelerated random access. |
| `MVExtractorVideoProcessor`| MVExtractor | Uses motion vectors to select high-motion frames, with EMA smoothing and sliding windows. |

`VideoProcessorFactory` chooses the appropriate processor based on
`Config.VIDEO_PROCESSOR_TYPE`.

### Task Layer (`task/`)

| Module                      | Purpose |
|-----------------------------|---------|
| `cache_manager.py`          | MD5-keyed result cache backed by TinyDB. Memory cache with async periodic flush to disk. |
| `webp_single.py`            | Picks the highest-scoring frame and exports a single WebP image. |
| `webp_animation.py`         | Collects consecutive frames around the best score and produces an animated WebP. |
| `video_preview_generator.py`| Selects keyframes via K-means clustering, then concatenates video segments with FFmpeg. |

### Utility Layer (`utils/`)

| Module                       | Purpose |
|------------------------------|---------|
| `image_utils.py`             | Crop, resize, and aspect-ratio-preserving image operations. |
| `file_service.py`            | Recursive file discovery and path helpers. |
| `logging_service.py`         | Structured logging with Loguru (console + file). |
| `concurrency_control.py`     | Semaphore-based request limiter exposed as a decorator. |
| `video_processor_context.py` | Context manager for safe video reader lifecycle. |
| `video_segmentation.py`      | Splits long videos into segments for parallel processing. |
| `visualization.py`           | Draws bounding boxes, landmarks, and score annotations for debugging. |

### Models (`models/`)

#### RetinaFace (`models/retinaface/`)

- `models/net.py`, `models/retinaface.py` — PyTorch model definitions.
- `detection.py` — TensorRT FP16 inference engine with pre-allocated GPU
  buffers and CUDA stream management.
- `compile_tensorrt.py` — Utility to compile a PyTorch `.pt` checkpoint to
  a TensorRT `.trt` engine.
- `pytorch_model.pt` — Pre-trained weights (Git LFS).
- `retinaface_fp16.trt` — Pre-compiled TensorRT engine (Git LFS).

#### HopeNet (`models/hopenet/`)

- `hopenet.py` — ResNet-based head pose regression model.
- `datasets.py` — Data loading utilities (300W-LP and AFLW2000).
- `utils.py` — Softmax-based angle decoding and visualization helpers.

## Threading Model

`FaceProcessor` uses a **producer–consumer** pattern:

```
Producer thread                      Consumer thread
──────────────────                   ──────────────────
VideoProcessor.read_frames()  ──►   Queue (max 100)  ──►  RetinaFace detect
FrameFilterChain.filter()                                  FaceFilterChain.filter()
                                                           Store results (thread-locked)
```

A `threading.Event` allows early termination when `SEARCH_ALL=False` and a
valid face has been found. A `threading.Lock` protects the shared result
dictionary.

## GPU Memory Management

- TensorRT pre-allocates input/output buffers at engine load time.
- CUDA streams are used for asynchronous host-to-device and device-to-host
  memory transfers.
- On CUDA OOM, the processor clears the CUDA cache, waits a configurable
  delay, and retries (up to `CUDA_OOM_RETRY_COUNT` times).

## Caching Strategy

The cache key is an MD5 hash of `(file_path, file_size, mtime)`. This
detects file replacement or modification. The cache manager maintains an
in-memory dictionary that is periodically flushed to a TinyDB JSON file
every 8 seconds by a background thread. An `atexit` handler ensures a final
flush on shutdown.

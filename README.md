# Face Analyzer

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Face Analyzer is a GPU-accelerated video analysis tool for batch face detection
and keyframe extraction. It scans video files, detects and scores faces using
deep learning models, and produces WebP summaries or preview videos to support
downstream video structuring and analysis.

## Purpose

- Scan video files in bulk and detect faces at sampled intervals.
- Filter and score faces to keep high-quality, meaningful frames.
- Produce artifacts such as single-frame WebP, animated WebP, and preview videos.
- Support local batch processing and an HTTP API mode.

## Key Features

- **RetinaFace detection** with TensorRT FP16 inference for real-time performance.
- **Multi-stage filtering** — composable face and frame filter chains (confidence,
  geometry, brightness, contrast, head pose).
- **Three video backends** — OpenCV, Decord, and MVExtractor (motion-vector aware).
- **Flexible output** — single-frame WebP, animated WebP, and K-means keyframe
  preview videos.
- **Caching** — TinyDB-backed result cache with async periodic flush.
- **Concurrency** — producer–consumer threading with semaphore-based API rate
  limiting.
- **Docker-ready** — CUDA 12.4 runtime image with Docker Compose configs.

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/architecture.md) | System design, component descriptions, threading model. |
| [API Reference](docs/api-reference.md) | HTTP endpoints, request/response schemas. |
| [Configuration](docs/configuration.md) | All tunable parameters explained. |
| [Deployment](docs/deployment.md) | Docker and bare-metal deployment guide. |
| [Contributing](CONTRIBUTING.md) | Development setup, coding standards, PR process. |
| [Roadmap](ROADMAP.md) | Planned features and long-term vision. |
| [Changelog](CHANGELOG.md) | Version history and notable changes. |

## Project Structure

```
face_analyzer/
  classes/        Data classes for results and inter-thread communication
  config/         Global configuration
  core/           Main processing logic and API glue
  module/         Face filters, frame filters, and video processors
  task/           WebP generation, caching, preview generation
  utils/          Helper utilities (io, logging, visualization, segmentation)
  models/         RetinaFace model assets and TensorRT tooling
  main.py         CLI entry point (local and server modes)
  ffmpeg_wrapper.sh  FFmpeg wrapper with optional analysis server integration
```

## Implementation Principles

1. Separation of concerns: core processing is decoupled from HTTP.
2. Modular filtering: face filters and frame filters are composable chains.
3. Performance-aware: GPU acceleration, sampling, caching, and optional
   TensorRT inference are used where available.
4. Resilience: retry logic and robust resource cleanup are implemented to avoid
   stalls or leaks in long-running jobs.

## How It Works

### 1. Video ingestion and sampling

- A `BaseVideoProcessor` implementation reads frames and metadata.
- Sampling interval is controlled by `Config.SAMPLE_INTERVAL_SEC`.
- Supported processors:
  - OpenCV (`CV2VideoProcessor`)
  - Decord (`DecordVideoProcessor`)
  - MVExtractor (`MVExtractorVideoProcessor`, motion-vector aware)

### 2. Frame filtering

- `FrameFilterChain` applies frame-level filters (e.g., brightness/contrast).
- Only frames passing frame filters proceed to face detection.

### 3. Face detection and scoring

- RetinaFace is used for face detection.
- Results are converted to `RetinaFaceResult` objects.
- `FaceFilterChain` applies filters (score and geometry).
- Frames are grouped by score and stored as valid candidates.

### 4. Artifact generation

Depending on configuration:

- Single-frame WebP: pick the highest-scoring frame and export.
- Animated WebP: collect consecutive frames around the highest score.
- Video preview: select keyframes via K-means on time/quality and
  concatenate segments.

### 5. Caching

- Results are cached to avoid reprocessing unchanged videos.
- Cached entries are keyed by file path, size, and modification time.

## Configuration

The main configuration lives in `config/settings.py`. Key areas:

- Face detection and filters
- Sampling and search behavior
- WebP and preview generation
- API settings and concurrency
- CUDA OOM retry policy
- Cache control

There is also an example environment file at `.env` (optional in Docker setups).

## Running

### Local batch mode

Process a directory or a single file:

```bash
python main.py local /path/to/input /path/to/output
```

### HTTP API mode

```bash
python main.py server --host 0.0.0.0 --port 5000
```

Endpoints:

- `POST /analyze` for video analysis and WebP animation generation
- `POST /analyze_frame` for video analysis and single-frame WebP

Expected JSON payload:

```json
{
  "input_path": "/path/to/video.mp4",
  "output_path": "/path/to/output.webp",
  "resolution": "640x480"
}
```

## Docker

The project includes:

- `Dockerfile` for a GPU-enabled runtime
- `docker-compose.yml` and `docker-compose.simple.yml` for service setup

Adjust volumes and ports as needed for your environment.

## Dependencies

Core dependencies include:

- Python 3.x
- OpenCV
- PyTorch
- Decord or MVExtractor (optional, configurable)
- FFmpeg (for preview generation and segmenting)

Optional GPU acceleration:

- CUDA-compatible GPU and drivers
- TensorRT engine (see `models/retinaface/compile_tensorrt.py`)

## Output Artifacts

Depending on configuration, outputs may include:

- Cached results: `face_analysis_cache.json`
- Single-frame WebP: `*_frame.webp`
- Animated WebP: `*_anim.webp`
- Preview video: `*_preview.mp4` (or configured format)

## Contribution Guide

See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

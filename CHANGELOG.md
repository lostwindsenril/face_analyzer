# Changelog

All notable changes to Face Analyzer are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- Comprehensive project documentation (`docs/` directory).
- `CONTRIBUTING.md` with development workflow guidelines.
- `ROADMAP.md` outlining future plans.
- This `CHANGELOG.md` file.

---

## [0.6.0] — 2026-03-15

### Added
- Docker support with CUDA 12.4.1 runtime image.
- `docker-compose.yml` and `docker-compose.simple.yml` for service
  orchestration.
- Git LFS tracking for model weights (`.pt`, `.pkl`, `.trt`).

---

## [0.5.0] — 2026-02-05

### Added
- MVExtractor video processor with motion-vector-aware frame selection.
- EMA smoothing and sliding window for motion detection.
- FFmpeg wrapper script (`ffmpeg_wrapper.sh`) for segment processing.
- Debug visualization for motion vectors.

---

## [0.4.0] — 2026-01-05

### Changed
- Refactored monolithic video processor into an abstract base class with
  three backends: CV2, Decord, and MVExtractor.
- Added `VideoProcessorFactory` for runtime processor selection.
- Added `VideoProcessorContext` for safe resource management.

---

## [0.3.0] — 2025-11-25

### Added
- TensorRT FP16 inference for RetinaFace, with pre-allocated GPU buffers
  and CUDA stream management.
- `compile_tensorrt.py` utility for converting PyTorch weights to TensorRT
  engines.
- CUDA OOM auto-retry mechanism.

### Changed
- Detection module rewritten to use TensorRT instead of pure PyTorch
  inference.

---

## [0.2.0] — 2025-09-18

### Added
- Flask HTTP API with `/analyze` and `/analyze_frame` endpoints.
- `APIHelper` for decoupled business logic.
- Semaphore-based concurrency control decorator.
- CLI entry point (`main.py`) supporting `local` and `server` modes.
- HopeNet head pose estimation model integration.
- `FaceProcessorHelper` for debug visualization.
- Video preview generator with K-means keyframe selection.
- Video segmentation utilities.
- Visualization tools for bounding boxes and landmarks.

---

## [0.1.0] — 2025-07-07

### Added
- Initial project structure and configuration system.
- RetinaFace face detection model (PyTorch).
- Core `FaceProcessor` with producer–consumer threading.
- Face filter chain: `ScoreFaceFilter`, `GeometryFaceFilter`.
- Frame filter chain: `QualityFilter` (brightness, contrast).
- Single-frame and animated WebP output generation.
- TinyDB-backed result cache with async flush.
- Batch processing manager.
- File discovery and logging utilities.
- Image crop/resize helpers.

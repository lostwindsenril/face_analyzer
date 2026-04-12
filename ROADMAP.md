# Roadmap

This document outlines the planned evolution of Face Analyzer. Items are
grouped by theme and roughly ordered by priority. Timelines are estimates
and may shift as the project evolves.

---

## Near-Term (Next 3 Months)

### Automated Testing Infrastructure
- [ ] Set up `pytest` with a small corpus of test videos.
- [ ] Unit tests for filter chains, cache manager, and image utilities.
- [ ] Integration tests for the full local-mode pipeline.
- [ ] CI pipeline (GitHub Actions) running lint and tests on every push.

### Face Recognition / Clustering
- [ ] Integrate a face embedding model (e.g., ArcFace or AdaFace) to group
      detected faces by identity.
- [ ] Per-identity best-frame selection for multi-person videos.
- [ ] Optional face-identity-based deduplication across a video library.

### Improved API
- [ ] Add request validation with Pydantic models.
- [ ] Return structured progress events via Server-Sent Events (SSE) for
      long-running jobs.
- [ ] OpenAPI / Swagger documentation auto-generated from route definitions.

---

## Mid-Term (3–6 Months)

### Multi-GPU and Distributed Processing
- [ ] Support multiple GPUs with round-robin or least-loaded scheduling.
- [ ] Optional Celery / Redis task queue for horizontal scaling.
- [ ] Batch-level parallelism: process multiple videos concurrently across
      GPUs.

### Model Upgrades
- [ ] Evaluate YOLOv8-Face or SCRFD as drop-in RetinaFace replacements for
      improved speed and accuracy.
- [ ] Add configurable model selection (swap detection backbone without
      code changes).
- [ ] Explore ONNX Runtime as a cross-platform alternative to TensorRT.

### Enhanced Output Formats
- [ ] AVIF single-frame and animated output as a WebP alternative.
- [ ] Thumbnail contact sheets (grid of top-N faces per video).
- [ ] JSON metadata sidecar files with detection coordinates and scores.

### Configuration Overhaul
- [ ] Migrate from class-level constants to a YAML/TOML config file with
      schema validation.
- [ ] Support per-job configuration overrides via the API.
- [ ] Environment variable overrides for all settings (12-factor friendly).

---

## Long-Term (6–12 Months)

### Web Dashboard
- [ ] Lightweight web UI for browsing processed results, viewing previews,
      and triggering re-analysis.
- [ ] Job queue visualization and real-time progress tracking.
- [ ] Annotation interface: manually accept/reject detected faces to build
      a feedback loop.

### Streaming / Real-Time Mode
- [ ] Accept RTSP or HLS streams as input for near-real-time face
      detection.
- [ ] Sliding-window processing with configurable buffer size.
- [ ] WebSocket push for live detection results.

### Plugin Architecture
- [ ] Define a plugin interface for custom filters, processors, and output
      generators.
- [ ] Support loading plugins from external Python packages.
- [ ] Community plugin registry.

### Edge Deployment
- [ ] Optimize for NVIDIA Jetson (TensorRT INT8, reduced memory footprint).
- [ ] ARM-compatible Docker images.
- [ ] Benchmark and document performance on Jetson Orin / AGX.

### Data Management
- [ ] SQLite or PostgreSQL backend as an alternative to TinyDB for large
      libraries.
- [ ] Result versioning: track how detection results change when models or
      settings are updated.
- [ ] S3 / MinIO object storage integration for output artifacts.

---

## Ongoing

- Dependency updates and security patches.
- Performance profiling and optimization.
- Documentation improvements.
- Community feedback triage and issue resolution.

---

*This roadmap is a living document. Priorities may change based on user
feedback, funding, and contributor availability. Suggestions are welcome —
open an issue or start a discussion on GitHub.*

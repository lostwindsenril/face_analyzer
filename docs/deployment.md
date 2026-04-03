# Deployment Guide

This document covers building, configuring, and running Face Analyzer in
production environments.

## Prerequisites

- **NVIDIA GPU** with CUDA 12.x compatible drivers.
- **Docker** ≥ 24.0 and **Docker Compose** ≥ 2.20.
- **NVIDIA Container Toolkit** (`nvidia-docker2`) for GPU passthrough.
- At least **8 GB GPU memory** recommended for TensorRT FP16 inference.

## Docker Deployment (Recommended)

### 1. Build the image

```bash
docker compose build face-analyzer
```

The Dockerfile is based on `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` and
installs all system and Python dependencies, including FFmpeg, OpenCV, PyTorch,
Decord, and TensorRT.

### 2. Configure volumes

Edit `docker-compose.yml` or `docker-compose.simple.yml` and adjust volume
mounts to point to your video library and output directories:

```yaml
volumes:
  - /path/to/videos:/data/input:ro
  - /path/to/output:/data/output
  - /path/to/models:/app/models
```

### 3. Start the service

```bash
# Full stack (with dependent services if any)
docker compose up -d

# Simplified standalone
docker compose -f docker-compose.simple.yml up -d
```

### 4. Verify

```bash
curl http://localhost:5000/
```

### 5. View logs

```bash
docker compose logs -f face-analyzer
```

## Bare-Metal Deployment

### System dependencies

```bash
# Ubuntu / Debian
sudo apt-get update && sudo apt-get install -y \
  python3 python3-pip ffmpeg libgl1 libglib2.0-0

# Optional: Decord
pip install decord

# Optional: MVExtractor (see its upstream repo for build instructions)
```

### Python dependencies

```bash
pip install torch torchvision opencv-python-headless flask loguru \
  tinydb pillow scikit-learn shapely pycuda
```

### TensorRT setup (optional but recommended)

1. Install TensorRT from NVIDIA's repositories matching your CUDA version.
2. Compile the TensorRT engine:

```bash
python models/retinaface/compile_tensorrt.py \
  --weights models/retinaface/pytorch_model.pt \
  --output models/retinaface/retinaface_fp16.trt \
  --fp16
```

### Run

```bash
# Local batch mode
python main.py local /data/input /data/output

# API server mode
python main.py server --host 0.0.0.0 --port 5000
```

## Performance Tuning

| Parameter | Recommendation |
|-----------|----------------|
| `VIDEO_PROCESSOR_TYPE` | Use `mvextractor` for smart frame selection; fall back to `decord` if MVExtractor is not available. |
| `SAMPLE_INTERVAL_SEC` | Lower values increase accuracy but raise GPU load. `3.0` is a good default. |
| `SEARCH_ALL` | Set to `False` for throughput; `True` for quality. |
| `CONCURRENCY_LIMIT` | Match to available GPU memory. 4–8 is typical for 16 GB GPUs. |
| `CUDA_OOM_RETRY_COUNT` | Keep at 10; reduce if you prefer fast-fail. |

## Monitoring

- Application logs are written to stdout (JSON via Loguru) and to
  `success_log.txt` / `error_log.txt` on disk.
- Processing results are logged to `processed_results.log`.
- Cache state is in `face_analysis_cache.json`.

## Security Notes

- The API binds to `0.0.0.0` by default. In production, place it behind a
  reverse proxy (Nginx, Traefik) with TLS termination.
- Input paths are accepted as-is from API requests. Ensure the service runs
  with minimal filesystem permissions and validate input paths in your proxy
  or orchestrator layer.
- The Docker image runs as root by default. For hardened deployments, add a
  non-root user in the Dockerfile.

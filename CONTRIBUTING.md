# Contributing to Face Analyzer

Thank you for considering a contribution to Face Analyzer! This document
explains how to set up a development environment, propose changes, and
maintain the quality bar expected by the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Branching Strategy](#branching-strategy)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Adding New Filters](#adding-new-filters)
- [Adding a New Video Processor](#adding-a-new-video-processor)
- [Documentation](#documentation)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

All participants are expected to treat each other with respect. Harassment,
discrimination, and abusive behavior will not be tolerated. Keep
discussions technical and constructive.

## Getting Started

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/face_analyzer.git
   cd face_analyzer
   ```
3. **Create a branch** for your work:
   ```bash
   git checkout -b feature/my-feature
   ```

## Development Setup

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA 12.x drivers (for GPU features)
- FFmpeg installed and on `$PATH`

### Install dependencies

```bash
pip install torch torchvision opencv-python-headless flask loguru \
  tinydb pillow scikit-learn shapely pycuda decord
```

### Compile TensorRT engine (optional)

```bash
python models/retinaface/compile_tensorrt.py \
  --weights models/retinaface/pytorch_model.pt \
  --output models/retinaface/retinaface_fp16.trt --fp16
```

### Verify installation

```bash
# Batch mode on a sample video
python main.py local /path/to/sample_video.mp4 /tmp/output

# Server mode
python main.py server --port 5000
curl -X POST http://localhost:5000/analyze_frame \
  -H "Content-Type: application/json" \
  -d '{"input_path":"/path/to/video.mp4","output_path":"/tmp/out.webp"}'
```

## Branching Strategy

| Branch pattern       | Purpose |
|----------------------|---------|
| `main`               | Stable release branch. |
| `feature/<name>`     | New features. |
| `fix/<name>`         | Bug fixes. |
| `refactor/<name>`    | Code restructuring without behavior change. |
| `docs/<name>`        | Documentation-only changes. |

## Commit Guidelines

This project follows **Conventional Commits**:

```
<type>(<scope>): <short summary>

<optional body>
```

Common types: `feat`, `fix`, `refactor`, `docs`, `perf`, `test`, `chore`.

Examples:
```
feat(filter): add blur detection to QualityFilter
fix(cache): prevent race condition during async flush
docs(readme): add TensorRT compilation instructions
perf(detection): switch to FP16 TensorRT inference
```

Keep each commit focused on a single logical change.

## Pull Request Process

1. Ensure your branch is up to date with `main`.
2. Verify that the application runs correctly in both local and server
   modes.
3. Write a clear PR description explaining **what** changed and **why**.
4. Link related issues if applicable.
5. Request a review from a maintainer.
6. Address review feedback in follow-up commits or by amending.

PRs should be **small and focused**. Large changes should be split into a
series of PRs when possible.

## Coding Standards

- **Language**: Python 3.10+.
- **Style**: Follow PEP 8. Use 4-space indentation.
- **Naming**: `snake_case` for functions and variables, `PascalCase` for
  classes.
- **Imports**: Group into standard library, third-party, and local imports,
  separated by blank lines.
- **Type hints**: Encouraged for public function signatures.
- **Docstrings**: Required for public classes and functions. Use the Google
  style.
- **Comments**: Write in English. Explain *why*, not *what*.
- **Configuration**: All tunable values belong in `config/settings.py`.
  Do not scatter magic numbers through the code.
- **Error handling**: Catch specific exceptions. Use logging, not print
  statements.
- **Security**: Never log or expose file-system paths from API responses
  beyond what the client already knows.

## Testing

There is currently no automated test suite. Contributions that add tests
are highly welcome. In the meantime:

1. **Manual smoke test** — Run local batch mode on a small set of videos
   and verify output artifacts.
2. **API test** — Start the server and exercise both endpoints with valid
   and invalid payloads.
3. **Edge cases** — Test with corrupt videos, extremely short clips, and
   videos with no faces.

When adding tests, place them in a `tests/` directory mirroring the source
tree structure.

## Adding New Filters

### Face Filter

1. Create a new file in `module/face_filter/`.
2. Subclass `BaseFaceFilter` and implement the `filter()` method.
3. Register the filter in `FaceFilterChain.__init__()`.
4. Add any required configuration to `config/settings.py`.
5. Update `docs/configuration.md`.

### Frame Filter

Same process using `BaseFrameFilter` in `module/frame_filter/`.

## Adding a New Video Processor

1. Create a new file in `module/video_processors/`.
2. Subclass `BaseVideoProcessor`.
3. Add the new type to `VideoProcessorFactory` and `Config.VIDEO_PROCESSOR_TYPE`.
4. Document the new option in `docs/configuration.md`.

## Documentation

- Keep `README.md` up to date with any user-facing changes.
- Update files in `docs/` when architecture or configuration changes.
- Use English for all documentation and code comments.

## Reporting Issues

Open a GitHub issue with:

- A clear title summarizing the problem.
- Steps to reproduce (input file format, configuration, command used).
- Expected vs. actual behavior.
- Environment details (OS, GPU, CUDA version, Python version).
- Relevant log snippets.

---

Thank you for helping improve Face Analyzer!

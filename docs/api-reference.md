# API Reference

Face Analyzer exposes a lightweight HTTP API when started in server mode. The
server is built with Flask and listens on the port specified by
`Config.HTTP_PORT` (default `5000`).

## Starting the Server

```bash
python main.py server --host 0.0.0.0 --port 5000
```

## Endpoints

### `POST /analyze`

Analyze a video and produce an **animated WebP** summary.

**Request body** (JSON):

| Field         | Type   | Required | Description |
|---------------|--------|----------|-------------|
| `input_path`  | string | yes      | Absolute path to the video file. |
| `output_path` | string | yes      | Absolute path for the output WebP file. |
| `resolution`  | string | no       | Target resolution as `WxH` (e.g., `640x480`). If omitted, the original resolution is used. |

**Example request:**

```bash
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "input_path": "/data/videos/sample.mp4",
    "output_path": "/data/output/sample_anim.webp",
    "resolution": "640x480"
  }'
```

**Success response** (`200`):

```json
{
  "status": "success",
  "output_path": "/data/output/sample_anim.webp",
  "frame_count": 15,
  "best_score": 0.9845
}
```

**Error response** (`400` / `500`):

```json
{
  "status": "error",
  "message": "Input file not found"
}
```

---

### `POST /analyze_frame`

Analyze a video and produce a **single-frame WebP** (the highest-scoring
face frame).

**Request body** (JSON):

Same schema as `/analyze`.

**Example request:**

```bash
curl -X POST http://localhost:5000/analyze_frame \
  -H "Content-Type: application/json" \
  -d '{
    "input_path": "/data/videos/sample.mp4",
    "output_path": "/data/output/sample_frame.webp"
  }'
```

**Success response** (`200`):

```json
{
  "status": "success",
  "output_path": "/data/output/sample_frame.webp",
  "best_score": 0.9912
}
```

---

## Concurrency Limits

The server enforces a concurrency limit via a semaphore. When the limit is
reached, new requests block until a slot becomes available. The limit is set
by `Config.CONCURRENCY_LIMIT` (default `8`).

## Error Handling

- `400 Bad Request` — Missing or invalid fields in the JSON body.
- `500 Internal Server Error` — Processing failure (CUDA error,
  corrupted video, etc.). The response body includes a `message` field
  with details.

## Health Check

A simple liveness check can be performed with:

```bash
curl http://localhost:5000/
```

Returns `200` with a plain text body when the server is running.

# Server inference service

This directory contains the server-side validation path and a FastAPI single-
image inference service. The benchmark and service use ONNX Runtime and
the existing `deploy_loongson_256/onnx_inference.py` preprocessing and
postprocessing helpers.

The service is intended to run on the team member's AMD64/x86_64 computer;
the Loongson client does not need `onnxruntime`.

## Files to deploy

Keep this structure when copying the deployment package:

```text
AI-Fall-Detection-System/
├── yolo11n-pose-256.onnx
├── deploy_server/
│   ├── server_inference.py
│   └── requirements.txt
└── deploy_loongson_256/
    └── onnx_inference.py
```

The final server IP is a deployment setting. Pass it to the Loongson client
with `--server-url` rather than writing it into source code.

## Install

```bash
python3 -m pip install -r requirements.txt
```

`onnxruntime` is the default CPU package. If a supported server GPU setup is
needed later, handle it as a separate deployment change. The service in this
directory only permits `CPUExecutionProvider` and never selects a GPU.

## Run

From `deploy_server/`:

```bash
python3 benchmark_server_inference.py --warmup 20 --iterations 100
```

The default model is the repository-root `yolo11n-pose-256.onnx`. A different
model path can be supplied with `--model`. The default provider is
`CPUExecutionProvider`; available providers are printed before the benchmark.

To manually request another provider when it is installed:

```bash
python3 benchmark_server_inference.py --provider CUDAExecutionProvider
```

The output includes model loading/session validation, one pipeline smoke test,
average/min/max/P50/P95 inference latency, theoretical FPS, CPU/OS/Python
information, and ONNX Runtime provider information. No benchmark result is
stored automatically, so existing Loongson benchmark artifacts are untouched.

## FastAPI service

Start from the repository root or `deploy_server/`:

```bash
python3 server_inference.py --model ../yolo11n-pose-256.onnx --host 0.0.0.0 --port 8000
```

`--model`, `--host`, and `--port` are configurable. The service uses the
`CPUExecutionProvider` only. Pillow is not required because OpenCV decodes
JPEG and PNG request bodies.

Call the single-image endpoint with multipart form data:

```bash
curl -X POST http://127.0.0.1:8000/infer \
  -F "file=@sample.jpg;type=image/jpeg"
```

The JSON response contains `boxes`, `keypoints`, `scores`, and
`inference_time_ms`. The FallDetector state machine is not used or modified;
the Loongson client can keep that state locally after receiving these results.

Run the local smoke tests without a real HTTP client:

```bash
python3 -m unittest discover -s . -p 'test_*.py'
```

After starting the service, check readiness before connecting the Loongson:

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok"}
```

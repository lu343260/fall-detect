# Loongson 2K0300 Deployment

This directory is the runtime-only 256x256 deployment package. It contains no training,
export, or PyTorch model dependency.

## Install

```bash
python3 -m pip install -r requirements.txt
```

The original `yolo11n-pose-256.onnx` is kept unchanged. An INT8 model can be
placed beside it and selected at runtime:

```bash
python3 pose_video.py --camera 0
python3 pose_video.py --camera 0 --model yolo11n-pose-256-int8.onnx
```

Useful options:

```bash
python3 pose_video.py --camera 0 --no-rotate
python3 pose_video.py --camera 0 --no-display
python3 pose_video.py --camera 0 --frame-skip 2 --no-display
```

`--frame-skip N` keeps reading every camera frame but runs YOLO only once every
N frames. Skipped frames reuse the latest pose result; the existing
`FallDetector` update condition is unchanged. Runtime statistics are appended to
`performance_log.csv`.

Real camera/video frame-skip benchmark (runs `1`, `2`, `5`, and `10`):

```bash
python3 benchmark_frame_skip.py --source 0 --duration 30 \
  --output benchmark/results/frame_skip_benchmark.csv
```

Use a video file instead of `--source 0` for repeatable tests. The CSV contains
model name, input size, frame skip, total frames, inference count, average
inference time, camera FPS, AI FPS, and the final fall-detection result.

Pure OpenCV DNN inference-boundary benchmark (no camera, preprocessing,
postprocessing, or drawing):

```bash
python3 benchmark_opencv_forward.py --model yolo11n-pose-256.onnx \
  --input-size 256 --warmup 20 --iterations 100
```

The benchmark reports separate `setInput()` and `forward()` timings, including
average/min/max/P50/P95, plus the theoretical FPS for their combined time.
The default model is the FP32 model actually used by `pose_video.py`; pass
`--model yolo11n-pose-256-int8.onnx` to test an optional INT8 model separately.

Compare both models and write a compact CSV:

```bash
python3 benchmark_opencv_int8.py \
  --model yolo11n-pose-256.onnx \
  --int8-model yolo11n-pose-256-int8.onnx \
  --input-size 256 --warmup 20 --iterations 100 \
  --output benchmark/results/opencv_int8_benchmark.csv
```

This first executes the INT8 model once and prints whether OpenCV DNN/CPU can
run it. The CSV contains `model_name`, `precision`, `avg_forward_ms`, and
`fps`; `status` and `error_reason` are also included so unsupported or missing
INT8 models are recorded without stopping the FP32 result.

`fall_detector.py` is kept unchanged from the 320 deployment package. The deployment
inference output remains compatible with `keypoints.xy` shaped `(N, 17, 2)` and
`keypoints.conf` shaped `(N, 17)`.

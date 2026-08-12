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
python3 pose_video.py --camera 0 --process-every 3
```

Pure OpenCV DNN forward benchmark (no camera, preprocessing, postprocessing,
or drawing):

```bash
python3 benchmark_opencv_forward.py --warmup 20 --iterations 100
```

The benchmark reports average `net.forward()` time, forward FPS, and the INT8
speedup ratio relative to FP32.

`fall_detector.py` is kept unchanged from the 320 deployment package. The deployment
inference output remains compatible with `keypoints.xy` shaped `(N, 17, 2)` and
`keypoints.conf` shaped `(N, 17)`.

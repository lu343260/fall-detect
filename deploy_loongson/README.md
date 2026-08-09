# Loongson 2K0300 Deployment

This directory is the runtime-only deployment package. It contains no training,
export, or PyTorch model dependency.

## Install

```bash
python3 -m pip install -r requirements.txt
```

Copy `yolo11n-pose.onnx` into this directory, then connect a camera and run:

```bash
python3 pose_video.py --camera 0
```

Useful options:

```bash
python3 pose_video.py --camera 0 --no-rotate
python3 pose_video.py --camera 0 --no-display
python3 pose_video.py --camera 0 --process-every 3
```

`fall_detector.py` is kept unchanged from the main project. The deployment
inference output remains compatible with `keypoints.xy` shaped `(N, 17, 2)` and
`keypoints.conf` shaped `(N, 17)`.

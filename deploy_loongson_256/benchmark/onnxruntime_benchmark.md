# OpenCV DNN / ONNX Runtime benchmark

在 `deploy_loongson_256` 目录执行：

```bash
python3 benchmark_onnxruntime.py
```

benchmark 固定使用：

- 模型：`yolo11n-pose-256.onnx`
- 输入：`1x3x256x256` FP32
- 设备：CPU
- 后端：OpenCV DNN、ONNX Runtime `CPUExecutionProvider`

输出文件默认为 `benchmark/results/opencv_onnxruntime_benchmark.csv`，字段为：

```text
backend,avg_forward_ms,fps
```

计时不包含模型加载、输入创建和 warmup。OpenCV 的 `setInput()` 也不计入
`avg_forward_ms`，与现有 OpenCV forward benchmark 保持一致。

LoongArch Linux 上需要预先安装或提供可用的 LoongArch-compatible
`onnxruntime` 构建；它是 benchmark 的可选依赖，不加入部署运行时
`requirements.txt`。

先确认当前 `python3` 环境：

```bash
python3 -m pip show onnxruntime
python3 -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
```

如果目标系统没有 LoongArch64 wheel，需要使用适配目标机的 ONNX Runtime
源码构建 Python wheel，再用同一个 `python3` 安装。官方文档提供了 Linux
源码构建和 Python wheel 安装流程；不要直接把 x86_64 或 aarch64 wheel
复制到 LoongArch 系统。

可调整采样次数，但不会改变模型、输入尺寸或执行设备：

```bash
python3 benchmark_onnxruntime.py --warmup 20 --iterations 100 \
  --output benchmark/results/opencv_onnxruntime_benchmark.csv
```

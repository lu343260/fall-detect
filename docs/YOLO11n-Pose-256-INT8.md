# YOLO11n-Pose-256 INT8 测试流程

## 1. 量化

在安装了 `opencv-python`、`numpy`、`onnx` 和 `onnxruntime` 的环境中，使用代表性图片目录校准：

```bash
python quantize_yolo11_pose_int8.py \
  --model yolo11n-pose-256.onnx \
  --output yolo11n-pose-256-int8.onnx \
  --calibration path/to/calibration-images \
  --max-images 100
```

脚本使用 ONNX Runtime 静态 QDQ INT8，激活为 `QUInt8`、权重为 `QInt8`。为兼容 opset 12 和旧版 OpenCV DNN，默认使用 per-tensor 权重量化。它会拒绝覆盖输入模型或已有输出文件。

如果目标 OpenCV DNN 对 QDQ 不兼容，可试用 `--quant-format qoperator`，但仍必须在目标板端完成前向验证。

没有真实校准图时可用 `--synthetic-calibration 10` 做流程冒烟测试，但该结果不能作为精度结论。

## 2. FP32/INT8 对比

准备一张代表性测试图片后运行：

```bash
python compare_yolo11_pose_int8.py \
  --fp32 yolo11n-pose-256.onnx \
  --int8 yolo11n-pose-256-int8.onnx \
  --source path/to/test.jpg \
  --warmup 10 --iterations 100 \
  --report int8_comparison_report.json
```

报告包含：模型文件大小、CPU FP32/INT8 平均推理耗时、推理 FPS，以及 top-1 检测中以下关键点的平均欧氏坐标误差和置信度差：

- nose
- left/right shoulder
- left/right hip
- left/right knee

这里的坐标误差是模型 256×256 输入坐标上的 MAE；它用于模型输出一致性检查，不等同于 mAP。最终精度应使用代表性的跌倒/非跌倒标注集，统计检测率、误检率、漏检率和关键点误差。

## 3. LoongArch 部署切换

将生成的 INT8 模型复制到 `deploy_loongson_256`，与原始模型并列保存，然后选择模型启动：

```bash
python3 pose_video.py --camera 0 --model yolo11n-pose-256.onnx
python3 pose_video.py --camera 0 --model yolo11n-pose-256-int8.onnx
```

跌倒判断代码和关键点接口不变，仅替换 ONNX 推理模型。请在目标 LoongArch 设备上分别记录部署日志中的 `avg_inference_ms` 和 `inference_fps`；PC 上的数字不能直接代表目标板性能。

## 4. 报告填写

`int8_comparison_report.json` 是机器可读结果。建议最终报告至少记录：

| 指标 | FP32 | INT8 | 变化 |
|---|---:|---:|---:|
| 模型大小 (MiB) | `models[0].size_mib` | `models[1].size_mib` | `(INT8-FP32)/FP32` |
| 平均推理耗时 (ms) | `inference_mean_ms` | `inference_mean_ms` | `(INT8-FP32)/FP32` |
| 推理 FPS | `inference_fps` | `inference_fps` | `(INT8-FP32)/FP32` |

关键点一致性直接查看 `keypoint_consistency_top_detection`。若 INT8 在关键点或置信度上出现明显偏差，应增加真实校准图片数量并覆盖不同光照、姿态、距离和遮挡情况后重新量化。

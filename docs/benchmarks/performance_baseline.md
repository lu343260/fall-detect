# AI-Fall-Detection-System 统一性能基线

> 版本：Baseline-2026-08-25  
> 数据来源：`docs/development/当前项目状态报告.md`、`docs/benchmarks/龙芯2K0300_YOLO11n_Pose推理性能测试报告.md`、根目录 `performance_log.csv`、根目录 `int8_comparison_report.json` 和 `deploy_loongson_256/benchmark/results/opencv_int8_benchmark.csv`。

## 1. 使用说明

本文档把当前已有 benchmark 按统一字段整理，作为后续性能优化的对照基线。数据没有重新测试，原始测试的 warmup、iteration、计时边界和运行环境可能不同，因此必须区分：

- **正式基线**：测试条件和结果文件较清楚，可用于当前阶段比较；
- **参考数据**：来自报告中的历史实测，能说明趋势，但不宜直接和其他行做精确排名；
- **待复测**：缺少目标文件、目标板结果或统一测试条件，暂不能作为正式性能结论。

本文只记录推理性能和模型文件对比，不把关键点输出一致性当成检测准确率。

## 2. 统一字段与计时口径

每次后续测试建议至少记录：

| 字段 | 说明 |
|---|---|
| date / commit | 测试日期和代码版本 |
| platform | 硬件、操作系统、CPU 架构 |
| runtime | Python、OpenCV、ONNX Runtime 版本 |
| model | 模型文件和精度 |
| input_size | 模型输入尺寸 |
| backend | OpenCV DNN 或 ONNX Runtime provider |
| threads | 推理线程配置 |
| frame_skip | 摄像头主循环跳帧参数；不适用于纯 forward benchmark 时填 N/A |
| warmup / iterations | 预热次数和正式采样次数 |
| timing_scope | `forward`、`setInput + forward` 或端到端 |
| avg_ms / p50_ms / p95_ms | 平均、中位数和 P95 延迟 |
| fps | 由同一计时口径换算的理论或实际 FPS |
| source_file | 原始 CSV、JSON 或报告位置 |

换算关系：

```text
FPS = 1000 / avg_inference_time_ms
latency_reduction = (baseline_ms - candidate_ms) / baseline_ms
```

不同 `timing_scope` 的数值不能直接横向比较。单次 `forward` 的 FPS 也不等于摄像头端到端 FPS。

## 3. 当前推荐主基线

当前龙芯部署的主基线建议固定为：

| 项目 | 主基线配置 |
|---|---|
| 平台 | 龙芯 2K0300 / LoongArch64 |
| 模型 | `yolo11n-pose-256.onnx` |
| 精度 | FP32 |
| 输入 | `1×3×256×256` |
| 后端 | OpenCV DNN CPU |
| 线程 | 以目标板正式 benchmark 记录为准；已有实验中 1～2 线程较优 |
| 跳帧 | 纯推理 benchmark 为 N/A；摄像头测试单独记录 `FRAME_SKIP` |
| 目标 | 作为模型、后端、算子和系统架构优化的对照 |

该基线不是“实时可用”标准，而是当前已能稳定运行、且已有最多对照数据的部署配置。

## 4. 龙芯 2K0300 输入尺寸基线

| 平台 | 模型/输入 | 后端 | 延迟 | FPS | 数据等级 | 结论 |
|---|---|---|---:|---:|---|---|
| LoongArch64 | YOLO11n-Pose / 640×640 FP32 | OpenCV DNN CPU | 约 24.680 s/frame | 约 0.041 | 参考数据 | 不适合视频检测 |
| LoongArch64 | YOLO11n-Pose / 640×640 FP32 | ONNX Runtime CPU | 约 21.526 s/frame | 约 0.046 | 参考数据 | 比 OpenCV DNN 快，但仍不可用 |
| LoongArch64 | YOLO11n-Pose / 320×320 FP32 | OpenCV DNN CPU | 约 6.2 s/frame | 约 0.16 | 参考数据 | 输入尺寸降低有效，但仍很慢 |
| LoongArch64 | `yolo11n-pose-256.onnx` / 256×256 FP32 | OpenCV DNN CPU | 约 4.10～4.2 s/frame | 约 0.23～0.24 | 参考数据 | 当前 OpenCV 部署主基线 |
| LoongArch64 | `yolo11n-pose-256.onnx` / 256×256 FP32 | ONNX Runtime CPUExecutionProvider | 约 3451.525 ms | 约 0.290 | 待正式复测 | 快速测试显示约 17.751% 延迟降低 |

说明：报告中的 ONNX Runtime 256×256 结果记录了 warmup 20、iterations 100 的目标配置，但当前工作区未找到对应的 `benchmark/results/backend_compare.csv`，因此暂列“待正式复测”。

## 5. 龙芯 OpenCV DNN 线程基线

固定模型为 `yolo11n-pose-256.onnx`，输入为 256×256，后端为 OpenCV DNN CPU：

| 线程数 | 平均推理耗时 | FPS | 相对线程 1 |
|---:|---:|---:|---:|
| 1 | 3926.840 ms | 0.255 | 基线 |
| 2 | 3858.437 ms | 0.259 | 延迟降低约 1.7% |
| 4 | 4146.693 ms | 0.241 | 变慢 |
| 8 | 4705.376 ms | 0.213 | 变慢 |

当前结论：2 线程是已有样本中的较优配置，但收益很小；4/8 线程出现退化。该结论只适用于当前龙芯 2K0300、OpenCV DNN CPU、256×256 模型和测试环境，更换硬件或后端必须重测。

## 6. 龙芯 INT8 基线

### 6.1 模型大小与独立输出对比

根目录 `int8_comparison_report.json` 记录了 60 张评估图片上的 FP32/INT8 对比：

| 指标 | FP32 | INT8 | 变化 |
|---|---:|---:|---:|
| 模型大小 | 11.089 MiB | 3.102 MiB | 减少约 72.0% |
| 独立环境平均推理耗时 | 5.398 ms | 6.539 ms | INT8 慢约 21.1% |
| 独立环境推理 FPS | 185.25 | 152.94 | INT8 慢约 17.4% |
| 配对图片 | 60 | 60 | — |

关键点坐标误差范围约为 1.28～2.63 px。该结果是有限图片上的顶部检测输出一致性，不是 mAP、召回率或跌倒识别准确率。

### 6.2 龙芯目标板结果

已有性能报告记录：

| 平台 | 模型/输入 | 后端 | 延迟 | FPS | 结论 |
|---|---|---|---:|---:|---|
| LoongArch64 | FP32 / 256×256 | OpenCV DNN CPU | 约 4.10 s/frame | 约 0.24 | 基准 |
| LoongArch64 | INT8 / 256×256 | OpenCV DNN CPU | 约 4.27 s/frame | 约 0.23 | 没有速度收益，略慢 |

当前 INT8 的确定收益是模型体积和存储/传输成本下降，不是推理加速。目标板 INT8 结果仍建议用同一脚本、同一输入和正式采样次数重新确认。

## 7. 跳帧与摄像头运行基线

已有跳帧调用计数如下：

| `FRAME_SKIP` | 总帧数 | YOLO 推理次数 | 观察 |
|---:|---:|---:|---|
| 1 | 15 | 15 | 每帧推理 |
| 2 | 30 | 15 | 约每 2 帧推理一次 |
| 5 | 75 | 15 | 约每 5 帧推理一次 |
| 10 | 140 | 14 | 约每 10 帧推理一次 |

该实验确认跳帧逻辑减少了模型调用次数，但没有降低约 4 秒的单次推理耗时，也没有证明告警延迟、漏检率或端到端响应得到改善。

Windows/AMD64 历史运行日志（根目录 `performance_log.csv`）：

| 平台 | `FRAME_SKIP` | Camera FPS | AI FPS | 平均推理耗时 |
|---|---:|---:|---:|---:|
| Windows / AMD64 | 1 | 21.18～21.87 | 21.06～21.93 | 31.20～34.02 ms |

该数据用于开发机功能和统计链路参考，不用于龙芯部署性能结论。

## 8. 已有文件状态与证据等级

| 数据/文件 | 当前状态 | 基线用途 |
|---|---|---|
| `docs/benchmarks/龙芯2K0300_YOLO11n_Pose推理性能测试报告.md` | 有完整历史结果和分析 | 输入尺寸、线程、INT8 和后端趋势参考 |
| `performance_log.csv` | Windows/AMD64 摄像头运行记录 | 通用日志格式和开发机参考 |
| `int8_comparison_report.json` | 60 张图片的独立 FP32/INT8 对比 | 模型大小和输出一致性参考 |
| `deploy_loongson_256/benchmark/results/opencv_int8_benchmark.csv` | FP32 成功，INT8 因目标路径模型不存在而失败 | 不能作为完整 INT8 对比基线 |
| `benchmark/results/backend_compare.csv` | 当前未发现 | ONNX Runtime 正式基线待生成 |

`opencv_int8_benchmark.csv` 中的有效记录是：FP32 平均 `9.222 ms`、108.442 FPS；INT8 状态为 `FAILED`，原因是部署目录中找不到 INT8 模型。这组 9.222 ms 数据的计时范围和目标环境应在正式报告中补充确认，不能与龙芯约 4 秒/帧的端到端数据直接比较。

## 9. 后续统一 benchmark 规范

后续优化前，建议在龙芯目标板固定：

```text
model: yolo11n-pose-256.onnx
input: 1x3x256x256 FP32
backends: OpenCV DNN CPU / ONNX Runtime CPUExecutionProvider
warmup: 20
iterations: 100
runs: 3 or more
```

每组同时输出：

- `avg`、`min`、`max`、P50、P95 延迟；
- `setInput`、`forward` 和端到端耗时；
- FPS；
- CPU、内存和线程配置；
- 模型文件 SHA-256；
- 原始 CSV 和运行日志。

只有在同一口径下重复测量后，才能把某个后端、线程数、INT8 模型或更小输入尺寸升级为正式优化结论。

## 10. 当前基线结论

1. 当前正式主基线应暂定为龙芯 2K0300、256×256、FP32、OpenCV DNN CPU，约 4.1～4.2 秒/帧。
2. ONNX Runtime 256×256 约 3.45 秒/帧的结果值得优先复测，潜在收益高于继续调 OpenCV 线程数。
3. INT8 当前主要是模型体积优化，不能宣称带来推理加速。
4. 跳帧是调用频率控制，不是单帧延迟优化。
5. 任何性能优化都必须和检测效果测试一起进行；当前性能基线不代表系统已经达到实时或生产可用。

## 11. FD-004 执行状态（2026-08-25）

本次执行对 FD-004 的 benchmark 入口完成了静态可执行性检查，但没有形成新的性能数据：

- 内置 Python 3.12.13 可完成 4 个 benchmark/支持检查脚本的 `py_compile`；
- 当前 Windows/AMD64 环境没有可用的 `python` 命令，内置 Python 未安装 `cv2`，因此 OpenCV INT8 支持检查在导入阶段终止；
- 当前没有龙芯 2K0300 目标板，也没有可用的 LoongArch ONNX Runtime，因此没有把本机或历史数据升级为正式基线；
- 本次未覆盖既有 `deploy_loongson_256/benchmark/results/opencv_int8_benchmark.csv`，避免改变用户已有失败证据。

因此，FD-004 状态为“阻塞”，当前仍只能保留以下结论：INT8 已确认模型体积优化和有限图片输出一致性；是否在龙芯上可加载、是否加速，以及 ORT 是否优于 OpenCV DNN，均待目标板同条件复测。

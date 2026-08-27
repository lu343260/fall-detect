# FD-004：完成龙芯 INT8 与推理后端同条件验证

- **状态**：`BLOCKED`
- **优先级**：P1
- **负责人**：待分配
- **创建日期**：2026-08-25
- **依赖任务**：FD-001
- **执行环境**：Loongson 2K0300 / LoongArch64
- **阻塞原因**：当前无法连接 Loongson 2K0300 开发板；缺少目标板 LoongArch ONNX Runtime 环境。
- **下一步动作**：连接龙芯开发板后继续执行正式 benchmark。
- **最后更新时间**：2026-08-25

## 任务背景

INT8 模型体积已减少约 72%，但现有证据显示在 OpenCV DNN CPU 路径上约 4.27 秒/帧，未产生速度收益；部署目录 benchmark 还存在 INT8 模型缺失导致的失败记录。需要把“存储优化”和“推理加速”分开验证。

## 当前问题

- 目标板 INT8 结果尚未按统一脚本、输入和采样次数正式复测。
- ONNX Runtime 256×256 的快速优势尚未形成正式 CSV。
- 当前没有算子级或计时分段证据解释 INT8 变慢原因。

## 目标

在同一目标板和统一 benchmark 口径下，确认 FP32/INT8 及 OpenCV DNN/ONNX Runtime 的可用性、延迟、模型大小和输出差异，并给出是否继续投入 INT8/ORT 的决策。

## 修改范围

- 允许修改：`deploy_loongson_256/` benchmark 入口、模型路径配置和相关 benchmark 文档。
- 允许生成：后端/精度对比 CSV、模型校验信息、输出差异和失败日志。
- 不在范围：未经 profiling 直接编写 LSX/LASX 内核或修改状态机。

## 验收标准

- [ ] FP32/INT8 模型路径明确且可加载；失败时记录可诊断原因，不以失败样本计算速度结论。
- [ ] 在相同 256×256、warmup、iterations、线程和计时口径下完成后端/精度对比。
- [ ] 报告模型大小、平均/P50/P95 延迟、FPS、输出差异和资源配置。
- [ ] 明确 INT8 是否只带来存储收益、是否带来速度收益，以及 ORT 结果是否足以升级为正式基线。

## 需要更新的文档

- `docs/benchmarks/performance_baseline.md`：更新 INT8/后端正式数据和证据等级。
- `docs/benchmarks/YOLO11n-Pose-256-INT8.md`：补充目标板验证边界。
- `docs/development/当前项目状态报告.md`：更新优化路线决策。
- 本任务文档：记录原始输出、结论和遗留风险。

## commit要求

- 建议提交信息：`perf(int8): validate LoongArch backend parity`
- 一个任务一个独立 commit；模型大文件和原始实验输出按项目约定管理，不混入无关改动。
- commit 前必须记录模型 SHA-256、后端版本、线程、warmup、iterations 和结果文件路径。

## 执行记录

- **执行日期**：2026-08-25
- **执行 commit**：当前工作区未创建新 commit；执行前保留既有用户变更。
- **验证环境**：Windows/AMD64 工作区；未连接龙芯 2K0300 目标板。
- **模型文件**：FP32 为 11,627,437 bytes，SHA-256 `F4C31783D3E40034501247932CE87680975B92B12A8CCA3137E36EDFCBDDE8C2`；INT8 为 3,252,974 bytes，SHA-256 `C8F483C98667622EF1870BD594A8D3227BD6F29B565526CD401EA44FFC577AD4`。
- **静态验证**：内置 Python 3.12.13 执行 4 个 benchmark/支持检查脚本的 `py_compile`，通过。
- **运行验证**：未完成。系统 `python` 命令不可用；内置 Python 缺少 `cv2`，运行 `test_opencv_int8_support.py` 在导入阶段以 `ModuleNotFoundError: No module named 'cv2'` 退出。
- **目标板验证**：未完成。当前会话没有 LoongArch64 目标板、目标板 Python/OpenCV/ORT 运行时或 `CPUExecutionProvider` 环境。
- **既有证据**：`int8_comparison_report.json` 记录 60 张图片的独立输出对比；FP32/INT8 模型大小约减少 72%，但 INT8 在该独立环境平均耗时约慢 21.1%。这不是龙芯目标板性能结论。
- **原始输出路径**：既有 `deploy_loongson_256/benchmark/results/opencv_int8_benchmark.csv` 保留为历史失败记录；本次未覆盖它，也未生成新的性能 CSV。

### 阻塞原因

本次执行完成了“读取任务 → 检查入口 → 静态验证 → 运行前置检查 → 记录失败原因”的诊断闭环，但未完成 FD-004 的目标板性能闭环。解阻塞需要在龙芯目标板准备 `python3`、OpenCV DNN、可加载的 FP32/INT8 模型，以及兼容的 ONNX Runtime；随后按 FD-001 的统一参数执行：`warmup=20`、`iterations=100`、相同 256×256 输入和明确线程配置。

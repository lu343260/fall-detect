# FD-001：统一龙芯性能基线与可复现 benchmark

- **状态**：`RUNNING`
- **优先级**：P0
- **负责人**：待分配
- **创建日期**：2026-08-25
- **依赖任务**：无
- **执行环境**：Windows/AMD64 文档整理环境；Loongson 2K0300 / LoongArch64 正式复测环境
- **阻塞原因**：已有 FP32、线程、`FRAME_SKIP` 和 INT8 记录，但正式统一 CSV、`setInput`/`forward`/后处理/端到端分段计时及完整后端复测尚未闭环。
- **下一步动作**：先固化现有 256×256 FP32 主基线；获得目标板后按统一参数补齐 backend benchmark 和原始 CSV。
- **最后更新时间**：2026-08-25

## 任务背景

当前龙芯 2K0300 已有 256×256 OpenCV DNN、ONNX Runtime 快速测试和 INT8 历史数据，但测试批次、计时边界和原始输出不完全统一。下一阶段所有性能优化都需要建立在同一口径上。

## 当前问题

- `benchmark/results/backend_compare.csv` 尚未形成正式证据。
- ORT 约 3.45 秒/帧的优势仍属于待正式复测结果。
- 现有数据混合了 `forward`、端到端和不同 warmup/iterations，不能直接排名。

## 当前进度

已完成或已具备：

- `docs/benchmarks/performance_baseline.md` 已整理统一性能基线文档；
- 龙芯 256×256 FP32 OpenCV DNN 主基线；
- OpenCV DNN 线程数实验；
- `FRAME_SKIP` 调用频率实验；
- INT8 模型大小、输出对比和历史推理记录。

尚未完成：

- 正式生成 `benchmark/results/backend_compare.csv`；
- 在相同 runs、warmup、iterations、线程和计时边界下完成 OpenCV DNN/ONNX Runtime/INT8 全量对比；
- 对 `setInput`、`forward`、后处理和端到端耗时进行统一分段记录；
- 将上述结果升级为完整、可审计的目标板正式基线。

## 目标

在龙芯目标板上生成可重复的 OpenCV DNN FP32、ONNX Runtime FP32 和 OpenCV DNN INT8 对比基线，并明确 `setInput`、`forward`、后处理和端到端的计时口径。

## 修改范围

- 允许修改：`deploy_loongson_256/` 中现有 benchmark 入口及其必要文档。
- 允许生成：正式 CSV、运行日志和必要的结果摘要；大型模型与原始实验数据按项目约定保留。
- 不在范围：修改 FallDetector 规则、引入新依赖、宣称实时可用或直接优化模型结构。

## 验收标准

- [ ] 固定模型、256×256 输入、平台、后端、线程、warmup=20、iterations=100 和 runs≥3，并记录模型 SHA-256。
- [ ] 同一命令或明确的同等命令可生成 FP32/INT8 后端对比 CSV，包含 avg、min、max、P50、P95、FPS 和失败原因。
- [ ] 分别记录 `setInput`、`forward`、后处理和端到端耗时，避免混淆计时范围。
- [ ] 结果注明 LoongArch64、运行时版本、线程配置和原始日志路径；结论可区分正式基线与参考数据。

## 需要更新的文档

- `docs/benchmarks/performance_baseline.md`：补充正式结果、计时口径和证据等级。
- `docs/development/当前项目状态报告.md`：更新后端结论和下一步建议。
- 本任务文档：填写验证命令、实际结果和遗留风险。

## commit要求

- 建议提交信息：`perf(benchmark): establish LoongArch unified baseline`
- 一个任务一个独立 commit；不得混入模型、视频、校准数据或无关代码。
- commit 前必须附测试平台、模型、输入、后端、线程、warmup、iterations 和原始 CSV 路径。

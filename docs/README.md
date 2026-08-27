# 文档索引

这里记录 AI-Fall-Detection-System 的开发过程、性能实验和测试证据。先看本页，再按任务进入对应目录。

## 目录

```text
docs/
├── development/
│   └── 项目开发日志.md
├── benchmarks/
│   ├── 龙芯2K0300_YOLO11n_Pose推理性能测试报告.md
│   └── YOLO11n-Pose-256-INT8.md
└── testing/
    └── test_record.md
```

## 推荐阅读顺序

1. `development/项目开发日志.md`：了解状态机、ONNX、龙芯部署和当前未完成事项。
2. `benchmarks/龙芯2K0300_YOLO11n_Pose推理性能测试报告.md`：查看平台、后端、输入尺寸、线程和 INT8 实测结论。
3. `benchmarks/YOLO11n-Pose-256-INT8.md`：按步骤复现量化和 FP32/INT8 对比。
4. `testing/test_record.md`：补充短期人工测试记录；正式性能数据应进入 benchmark 报告或机器可读 CSV/JSON。

## 维护规则

- 新文档按“开发过程 / 性能基准 / 功能测试”归类，不再直接堆在 `docs/` 根目录。
- 实验记录至少包含日期、代码版本、平台、模型、输入尺寸、后端、参数和输出文件。
- 区分“已测量事实”“推测原因”和“后续计划”；不要用有限样本的关键点一致性代替准确率结论。
- 文档中的命令从项目根目录或指定部署目录执行时，要明确工作目录。

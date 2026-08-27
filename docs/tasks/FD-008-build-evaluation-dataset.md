# FD-008：建立第一版场景评估数据集

## 任务元数据

| 字段 | 内容 |
|---|---|
| 任务 ID | FD-008 |
| 状态 | `RUNNING` |
| 优先级 | P0 |
| 执行环境 | Windows/AMD64 数据整理环境 |
| 依赖任务 | FD-002 |
| 阻塞原因 | 真实视频和事件标签尚未准备；本阶段只建立采集规范，不采集视频 |
| 下一步动作 | 按采集规范准备样本、填写 manifest/events 并运行数据检查工具 |
| 最后更新时间 | 2026-08-27 |

## 背景

FD-002 已完成场景评估框架建设，但 `datasets/fall_eval_v1/` 尚无可用于回归的真实媒体和完整标注。FD-007 Phase 1 Validation 因此无法执行。本任务只建立第一版数据集结构、字段模板和完整性检查工具，为后续固定数据回归提供输入。

## 目标

建立以下数据集目录：

```text
datasets/fall_eval_v1/
├── videos/
├── labels/
│   └── events.csv
├── reports/
├── manifest.csv
└── README.md
```

第一版数据集至少覆盖以下场景：

`normal_standing`、`sitting`、`bending`、`front_fall`、`side_fall`、`slow_fall`、`recovery`、`occlusion`。

## 数据格式

### manifest.csv

字段必须为：

```text
sample_id,video_path,scene_type,fps,duration,person_id
```

其中 `sample_id` 必须唯一，`video_path` 相对于 `datasets/fall_eval_v1/`，`scene_type` 使用上述规范名称。

### labels/events.csv

字段必须为：

```text
sample_id,start_time,end_time,expected_alarm
```

每个 manifest 样本都必须有事件标签；无报警场景使用 `expected_alarm=false`，报警事件使用 `true` 并填写秒级时间范围。

## 数据检查工具

新增根目录脚本 `check_evaluation_dataset.py`，仅使用 Python 标准库检查：

- `videos/`、`labels/`、`reports/` 是否存在；
- manifest 和 events 的字段是否正确；
- `sample_id` 是否非空且唯一；
- 视频文件是否存在且位于数据集目录内；
- 场景类型是否覆盖八类必需场景；
- FPS、时长和事件时间是否有效；
- manifest 样本是否都有完整事件标签。

运行方式：

```powershell
python check_evaluation_dataset.py --dataset datasets/fall_eval_v1
```

当前空模板运行检查工具应报告数据集尚未完成；补齐样本和标签后，输出 `DATASET_OK` 才可进入 FD-007 回归验证。

## 范围边界

- 不采集、提交或伪造真实视频；
- 不修改 YOLO 模型；
- 不修改 `FallDetector`；
- 不修改现有推理代码或部署流程；
- 不在本任务中生成模型评估指标。

## 验收标准

- [x] 创建 `videos/`、`labels/`、`reports/` 目录占位；
- [x] 创建 `manifest.csv` 和 `events.csv` 字段模板；
- [x] 明确八类必需场景；
- [x] 提供检查文件、标签完整性和 `sample_id` 唯一性的脚本；
- [x] 建立视频采集、场景覆盖、标签、命名和质量检查规范；
- [x] 建立数据质量验收、`INVALID`/`INCOMPLETE` 状态和输出格式规范；
- [ ] 补齐真实样本并使检查工具输出 `DATASET_OK`；
- [ ] 使用数据集执行 FD-007 Phase 1 baseline/Phase 1 对比验证。

## 后续执行记录

已完成采集规范和数据质量验收设计。本任务仍处于 `RUNNING`，待真实样本和标注补齐后再完成数据集验收。

## 建议提交信息

```text
test(dataset): scaffold fall evaluation dataset
```

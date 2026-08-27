# 跌倒检测场景回归测试集规范 v1

> 适用任务：FD-002  
> 版本：`scenario-eval-v1`  
> 建立日期：2026-08-27

## 1. 目标与边界

本规范用于建立可追溯、可复现的跌倒检测场景回归测试集，支持后续比较模型、输入尺寸、推理后端和参数变化。

本阶段只定义场景、数据、标签、运行记录和指标口径，不修改 YOLO 模型，不修改 `FallDetector` 核心逻辑。当前规范完成不代表项目已经具备准确率、召回率或误报率结论；这些指标必须在真实样本完成标注并运行评估后计算。

## 2. 场景分类

每个样本至少有一个主场景标签，可附加视角、遮挡、光照、距离和人数标签。

| 类别 | 场景标签 | 说明 | 期望检测结果 |
|---|---|---|---|
| 正常活动 | `standing` | 稳定站立、走动或轻微活动 | 不报警 |
| 正常活动 | `sitting` | 坐下、坐着或从站立转为坐下 | 不报警 |
| 正常活动 | `bending` | 正面/侧身弯腰、拾取物品 | 不报警；重点观察误报 |
| 跌倒正例 | `fall_front` | 正面方向跌倒 | 报警 |
| 跌倒正例 | `fall_side` | 侧面方向跌倒 | 报警 |
| 跌倒正例 | `fall_slow` | 缓慢滑倒、失去平衡后倒地 | 报警 |
| 恢复 | `recovery` | 跌倒后重新站起或恢复正常姿态 | 报警解除并记录恢复时间 |
| 观测困难 | `occlusion` | 人体部分被家具、物体或他人遮挡 | 记录关键点缺失，不默认判定算法错误 |
| 观测困难 | `no_person` | 画面无人 | 不报警 |

建议附加标签：`front_view`、`side_view`、`back_view`、`low_light`、`small_person`、`multi_person`、`partial_occlusion`、`full_occlusion`。

## 3. 数据目录规范

实验数据默认不提交到版本库，建议放在项目外部或按项目约定归档。目录结构如下：

```text
datasets/fall_eval_v1/
├── README.md
├── manifest.csv                 # 样本元数据
├── media/
│   ├── standing/
│   ├── sitting/
│   ├── bending/
│   ├── fall_front/
│   ├── fall_side/
│   ├── fall_slow/
│   ├── recovery/
│   ├── occlusion/
│   └── no_person/
├── labels/
│   └── events.csv                # 真实事件标签
├── predictions/<run_id>.csv      # 每次模型/参数运行结果
└── reports/<run_id>/             # 指标摘要、日志和配置快照
```

命名规则：`<scene>_<camera>_<take>.<ext>`。`manifest.csv` 必须记录 `sample_id`、相对媒体路径、主场景、视角、遮挡、光照、人数、来源、帧率、分辨率和标注版本。媒体路径应使用相对路径，不写入真实设备凭据或敏感信息。

## 4. 标签与测试记录格式

### 4.1 事件标签 `events.csv`

推荐字段：

```text
sample_id,scene,person_id,event_start_s,event_end_s,fall_start_s,fall_end_s,expected_alarm,expected_recovery,occlusion,notes,annotator,label_version
```

- `event_start_s` / `event_end_s`：样本中该行为或观测事件的时间范围；
- `fall_start_s` / `fall_end_s`：真实跌倒开始和结束时间，无跌倒场景留空；
- `expected_alarm`：该事件是否应触发报警；
- `expected_recovery`：是否要求记录恢复；
- `person_id`：多人场景必须稳定标识目标人员。

### 4.2 单次运行记录

每次对比必须生成唯一 `run_id`，并记录：

| 字段 | 内容 |
|---|---|
| `run_id` | 日期、代码版本和参数组合的唯一标识 |
| `label_version` | 使用的标签版本 |
| `model` / `model_sha256` | 模型文件及哈希 |
| `input_size` | 例如 `256x256` |
| `backend` | OpenCV DNN 或 ONNX Runtime provider |
| `platform` | Windows/AMD64 或 LoongArch64 |
| `threshold_config` | 阈值和 `FallDetector` 配置快照，不在本任务中修改 |
| `source_path` | 输入媒体或清单路径 |
| `output_path` | 原始预测结果和报告路径 |

### 4.3 预测结果记录

推荐每个样本/人员一行，字段如下：

```text
run_id,sample_id,person_id,scene,predicted_alarm,alarm_time_s,recovery_time_s,first_alarm_latency_s,recovery_latency_s,missing_keypoints,missing_keypoint_rate,notes
```

`predicted_alarm` 只能填写 `ALARM`、`NO_ALARM` 或 `INVALID`；输入损坏、模型加载失败等情况填写 `INVALID`，不能计入 TP/FP/FN。

## 5. 评估指标定义

默认以事件级统计为主，帧级统计作为辅助。每次报告必须说明统计级别和报警匹配窗口 `alarm_match_window_s`。

### 5.1 TP、FP、FN

- **TP（True Positive）**：真实标签为跌倒，且在允许匹配窗口内产生有效报警；同一真实事件只计一次。
- **FP（False Positive）**：真实标签为不应报警，或没有对应真实跌倒事件，却产生有效报警。
- **FN（False Negative）**：真实标签为跌倒，但在允许匹配窗口内没有有效报警。
- `INVALID` 样本单独统计，不计入 TP、FP、FN；报告必须列出无效原因。

可选报告：`precision = TP / (TP + FP)`、`recall = TP / (TP + FN)`、`F1 = 2 * precision * recall / (precision + recall)`。分母为 0 时填写 `N/A`，不得填 0 伪造结果。

### 5.2 首次报警时间

- `first_alarm_time_s`：该样本第一次进入有效报警状态的时间戳；没有报警填写 `N/A`。
- `first_alarm_latency_s = first_alarm_time_s - fall_start_s`：真实跌倒开始到首次报警的延迟。
- 非跌倒场景产生报警时记录报警时间，并计为 FP；不计算跌倒报警延迟。
- 报告至少给出有效样本数、平均值、中位数、P95 和最大值。

### 5.3 恢复时间

- `recovery_time_s`：报警后首次满足“恢复为正常”判定的时间戳；必须同时记录所需连续确认帧数或时长。
- `recovery_latency_s = recovery_time_s - first_alarm_time_s`：首次报警到报警解除的恢复延迟。
- 没有报警或没有观察到恢复时填写 `N/A`，并在报告中区分“未恢复”和“样本结束”。

### 5.4 关键点缺失

- 每个评估帧记录 `missing_keypoints`，列出低于本次运行置信度阈值的关键点名称或索引；阈值必须写入运行配置。
- `missing_keypoint_rate = 缺失关键点数量 / 应检查关键点数量`。
- 报告按场景统计关键点缺失率，并分别列出因关键点缺失导致的 `INVALID`、误报和漏报数量。
- 关键点缺失是观测质量指标，不直接等同于检测错误。

## 6. 推荐评估流程

1. 固定 `label_version`，核对 `manifest.csv` 和 `events.csv`。
2. 固定模型、输入尺寸、后端、阈值配置和运行平台，生成 `run_id`。
3. 对全部场景运行推理，保存原始预测、报警时间、恢复时间和关键点缺失记录。
4. 按事件级规则匹配报警，计算 TP、FP、FN、报警延迟和恢复延迟。
5. 按场景、视角、遮挡和人数输出分组结果，保留无效样本和失败原因。
6. 只有在相同数据、标签版本和统计口径下，才比较不同模型或参数的变化。

## 7. 自动评估工具

项目根目录的 `evaluate_scenario_set.py` 是独立评估入口，复用现有 `ONNXPoseDetector` 和 `FallDetector`，不改变摄像头部署流程。

```powershell
python evaluate_scenario_set.py `
  --dataset datasets/fall_eval_v1 `
  --model yolo11n-pose-256.onnx
```

输出默认写入 `datasets/fall_eval_v1/reports/<run_id>/`：

- `video_predictions.csv`：每个视频/样本的预测、TP/FP/FN 分类、首次报警时间、恢复时间和关键点缺失率；
- `frame_predictions.csv`：逐帧状态、报警标记和关键点缺失记录；
- `summary.json`：整体及按场景汇总、运行参数、模型 SHA-256 和延迟统计。

可用 `--alarm-match-window` 调整事件级报警匹配窗口，使用 `--max-frames` 做小规模冒烟运行。实际结果必须记录模型、平台、输入尺寸、后端、阈值配置和输出路径。

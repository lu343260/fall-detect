# FD-007 Phase 1：评估器部署一致性验证方案

> 目标：确认 `evaluate_scenario_set.py` 的 Phase 1 修改，相比修改前 baseline 更接近正式 `pose_video.py` 摄像头处理流程。  
> 范围：仅验证图像旋转、`FRAME_SKIP`、状态机更新频率和无效帧状态保持；不包含 FD-007 Phase 2/3。

## 1. 对照对象

使用同一份 `datasets/fall_eval_v1/`、同一模型、同一标签、同一平台和同一参数，分别运行：

1. **修改前 baseline**：Phase 1 修改前保存的评估器版本，使用固定文件副本或可复现 Git ref；
2. **修改后版本**：当前 `evaluate_scenario_set.py`。

baseline 必须记录文件 SHA-256 或 Git ref，不能用运行后临时修改的文件代替。两次运行的模型和 `events.csv` 必须完全相同。

推荐固定配置：

```text
dataset: datasets/fall_eval_v1
model: 同一个 ONNX 模型
frame_skip: 1、2、5 分别验证
confidence_threshold: 0.5
alarm_match_window_s: 5
time basis: 当前版本沿用 media time；本方案不提前引入 realtime 模式
```

## 2. 修改前后行为差异

| 行为 | 修改前 baseline | 修改后 Phase 1 | 应与正式部署一致的判断 |
|---|---|---|---|
| 图像旋转 | 读取视频帧后直接送入模型 | 每个读取帧先执行 `cv2.rotate(frame, cv2.ROTATE_180)` | 模型输入帧方向一致 |
| `FRAME_SKIP` | 每帧调用 `ONNXPoseDetector` | 增加 `--frame-skip`；仅在 `capture_frame_number % frame_skip == 0` 时推理 | 推理帧集合与正式入口一致 |
| 状态机更新频率 | 每个有效关键点帧都调用 `FallDetector.detect()` | 仅在推理帧且帧号为 3 的倍数时调用 | 与正式入口的 `frame_count % 3 == 0` 一致 |
| 无效帧处理 | `result=None` 后按 `fall=False` 处理，可能触发恢复 | 无人体、关键点无效、跳过推理或异常时保留 `last_result`，不更新状态机、不触发恢复 | 不因观测无效改变有效状态 |

正式入口的参考规则是：先读取并旋转每一帧；满足 `FRAME_SKIP` 才推理；只有必要关键点有效且帧号为 3 的倍数时才更新状态机；其他情况保留 `last_result`。

## 3. 回归验证方法

### 3.1 同一样本双版本运行

对每个选定视频，确保两次运行使用：

- 相同 `sample_id` 和媒体文件；
- 相同模型文件和模型 SHA-256；
- 相同 `manifest.csv`、`events.csv` 和 `label_version`；
- 相同 `--frame-skip`、关键点置信度阈值和报警匹配窗口；
- 相同 Python、ONNX Runtime、OpenCV、平台和工作目录。

分别保存到不同目录，例如：

```text
datasets/fall_eval_v1/reports/fd007_phase1_before/<run_id>/
datasets/fall_eval_v1/reports/fd007_phase1_after/<run_id>/
```

修改前版本的运行命令以其实际参数为准；修改后版本使用：

```powershell
python evaluate_scenario_set.py `
  --dataset datasets/fall_eval_v1 `
  --model yolo11n-pose-256.onnx `
  --frame-skip 1 `
  --output datasets/fall_eval_v1/reports/fd007_phase1_after/<run_id>
```

至少使用一个包含跌倒和恢复过程的视频、一个包含遮挡/关键点缺失的视频，以及一个正常活动视频。正式验收应使用完整固定回归集。

### 3.2 结构行为检查

除指标报告外，检查 `frame_predictions.csv`：

- 对 `frame_skip=N`，`inference_run=1` 的帧必须恰好满足 `frame_number % N == 0`；
- `state_update=1` 的帧必须同时满足推理帧和 `frame_number % 3 == 0`；
- 旋转检查应使用测试替身或输入方向可识别的样本，确认模型接收的是旋转后的帧；
- 无人体/关键点无效帧的 `state_update` 必须为 0；
- 无效帧前后状态应保持不变，不能新增 `recovery_time_s`。

## 4. 对比指标

每个版本和每个 `run_id` 均记录整体及按场景结果：

| 指标 | 对比方式 |
|---|---|
| TP | 真实跌倒且在报警匹配窗口内报警的事件数 |
| FP | 非跌倒事件产生报警的事件数 |
| FN | 真实跌倒但未在匹配窗口内报警的事件数 |
| 首次报警时间 | 记录 `first_alarm_time_s` 和 `first_alarm_latency_s`，比较有效报警事件的均值、中位数、P95 |
| 恢复时间 | 记录 `recovery_time_s` 和 `recovery_latency_s`；重点检查无效帧后是否出现假恢复 |

指标比较必须同时保留：样本数、有效样本数、`INVALID` 数、场景标签版本、模型哈希和原始输出路径。不能只比较总 TP/FP/FN 而忽略样本覆盖变化。

## 5. 有效性判断标准

认为 Phase 1 修改有效，需要同时满足：

1. 结构检查全部通过：旋转、`FRAME_SKIP`、状态机更新帧集合与正式入口规则一致；
2. 无效帧不会更新 `FallDetector`，不会把上一有效跌倒状态改成 `fall=False`，不会产生假恢复时间；
3. 同一数据和配置下，修改后没有因评估器自身处理错误增加 `INVALID` 样本；
4. TP、FP、FN、首次报警时间和恢复时间均可从逐帧/逐视频输出追溯；
5. 若 TP/FP/FN 或报警延迟发生变化，能由“状态机更新频率与部署一致化”解释，并在报告中单独说明，不能直接表述为算法精度提升；
6. 修改后评估结果与正式部署产生的同条件参考轨迹在帧处理和状态转换上保持一致。

以下情况不能判定为有效：只看到 FPS 或运行时间变化；只比较最终报警数量；只运行修改后版本；使用不同媒体、模型、阈值或标签版本；把无效帧产生的假恢复当作真实恢复。

## 6. 验证记录模板

- **baseline ref/SHA-256**：
- **after ref/SHA-256**：
- **数据集路径与 label_version**：
- **模型与 SHA-256**：
- **平台/运行时**：
- **frame_skip**：
- **样本数量**：
- **结构检查结果**：
- **before 报告路径**：
- **after 报告路径**：
- **TP/FP/FN 对比**：
- **首次报警时间对比**：
- **恢复时间对比**：
- **结论**：有效 / 部分有效 / 无效
- **遗留问题**：


# FD-007 Phase 1 Validation Report

- **日期**：2026-08-27
- **结论**：`NEED_REVIEW`
- **范围**：仅验证 Phase 1：图像旋转、`FRAME_SKIP`、状态机更新频率和无效帧状态保持。
- **未执行**：FD-007 Phase 2、Phase 3。

## 1. 固定配置

| 项目 | 固定值 |
|---|---|
| 数据集 | `datasets/fall_eval_v1/` |
| manifest | `datasets/fall_eval_v1/manifest.csv` |
| 标签 | `datasets/fall_eval_v1/labels/events.csv` |
| 模型 | `yolo11n-pose-256.onnx` |
| 模型 SHA-256 | `F4C31783D3E40034501247932CE87680975B92B12A8CCA3137E36EDFCBDDE8C2` |
| `confidence_threshold` | `0.5` |
| `alarm_match_window_s` | `5` |
| Phase 1 `frame_skip` | `1` |
| Python | `C:\Users\pc\AppData\Local\Programs\Python\Python311\python.exe` |

## 2. baseline 运行

baseline 使用 Phase 1 修改前的 `HEAD:evaluate_scenario_set.py`，其 Git ref 为当前 `HEAD`。由于 baseline 不支持新增的 `--frame-skip` 参数，因此使用其原有默认行为运行：

```powershell
git show HEAD:evaluate_scenario_set.py | `
  C:\Users\pc\AppData\Local\Programs\Python\Python311\python.exe - `
  --dataset datasets/fall_eval_v1 `
  --model yolo11n-pose-256.onnx `
  --alarm-match-window 5 `
  --confidence-threshold 0.5
```

结果：

```text
manifest not found: E:\AI_Project\AI-Fall-Detection-System\datasets\fall_eval_v1\manifest.csv
exit code: 1
```

因此 baseline 未进入视频读取和推理阶段，TP、FP、FN、首次报警时间、恢复时间、`inference_run` 次数和 `state_update` 次数均为 `N/A`。

## 3. Phase 1 运行

```powershell
C:\Users\pc\AppData\Local\Programs\Python\Python311\python.exe `
  evaluate_scenario_set.py `
  --dataset datasets/fall_eval_v1 `
  --model yolo11n-pose-256.onnx `
  --frame-skip 1 `
  --alarm-match-window 5 `
  --confidence-threshold 0.5
```

结果：

```text
manifest not found: E:\AI_Project\AI-Fall-Detection-System\datasets\fall_eval_v1\manifest.csv
exit code: 1
```

因此 Phase 1 版本同样未进入视频读取和推理阶段，TP、FP、FN、首次报警时间、恢复时间、`inference_run` 次数和 `state_update` 次数均为 `N/A`。

## 4. 对比分析

| 项目 | baseline | Phase 1 | 当前判断 |
|---|---|---|---|
| 数据集读取 | 失败，manifest 缺失 | 失败，manifest 缺失 | 无法进行数据层对比 |
| 视频预测结果 | 未生成 | 未生成 | 无法验证 |
| TP / FP / FN | `N/A` | `N/A` | 无法验证 |
| 首次报警时间 | `N/A` | `N/A` | 无法验证 |
| 恢复时间 | `N/A` | `N/A` | 无法验证 |
| `inference_run` 次数 | `N/A` | `N/A` | 无法验证 |
| `state_update` 次数 | `N/A` | `N/A` | 无法验证 |

当前只能确认两次运行使用了同一模型参数，且都在相同的 manifest 缺失前置条件处停止；不能据此证明 Phase 1 在真实视频上的部署一致性。

## 5. 结构验证状态

Phase 1 的离线单元测试已在前序执行中验证旋转、`FRAME_SKIP`、状态机调用频率和无效帧状态保持，3 项测试通过。但本次要求的 baseline 与 Phase 1 同数据双版本回归尚未完成。

## 6. 判定与后续动作

**FD-007 Phase 1：`NEED_REVIEW`**

判定原因：

1. `datasets/fall_eval_v1/manifest.csv` 当前不存在；
2. 无法加载真实视频并生成两套预测结果；
3. 无法比较 TP、FP、FN、首次报警时间、恢复时间、`inference_run` 和 `state_update`；
4. 不能把“两个版本同时因前置文件缺失退出”当作一致性通过。

恢复验证所需动作：

1. 准备固定版本的 `manifest.csv`、`labels/events.csv` 和至少一个跌倒/恢复、一个遮挡、一个正常活动视频；
2. 固定同一模型、标签版本、阈值、`frame_skip` 和运行环境重新运行 baseline 与 Phase 1；
3. 保存两个版本的 `video_predictions.csv`、`frame_predictions.csv` 和 `summary.json`；
4. 按本报告第 4 节补齐指标与结构检查结果后重新判定。


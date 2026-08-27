# fall_eval_v1 数据采集规范

## 1. 目的与范围

本规范用于建立第一版跌倒检测场景回归数据集，为 FD-002 场景评估和 FD-007 Phase 1 Validation 提供可重复输入。

本阶段只定义采集、命名、标注和质量检查要求，不采集真实视频，不修改 YOLO 模型、`FallDetector`、推理代码或部署流程。

## 2. 视频采集规范

每个视频样本必须记录以下元数据，并填写到 `manifest.csv`：

| 项目 | 要求 |
|---|---|
| 分辨率 | 优先使用与评估模型输入链路匹配的原始分辨率；不得在采集后隐式裁剪或拉伸。实际宽高应在采集记录中保留。 |
| FPS | 记录视频实际帧率，使用正数；不要用估计值替代媒体元数据。 |
| 时长 | 记录实际视频时长，单位为秒，保留至少两位小数。 |
| 摄像头角度 | 记录 `front`、`side`、`diagonal` 或更具体的说明；正面跌倒和侧面跌倒必须能从记录中区分。 |
| 光照条件 | 记录 `daylight`、`indoor`、`low_light`、`backlight` 等条件，必要时补充灯光变化或阴影说明。 |
| 画面稳定性 | 摄像头固定安装，避免采集过程中移动、变焦或频繁自动曝光变化。 |
| 内容完整性 | 视频从场景开始前保留短暂上下文，直到动作结束或恢复阶段结束；不得剪掉关键动作。 |

推荐每个样本包含 3 秒以上的稳定上下文；具体时长以能完整覆盖场景事件为准。采集人员应确认视频可正常解码，且不包含无关个人隐私或敏感信息。

## 3. 场景覆盖要求

第一版数据集至少覆盖以下场景，并使用规定的 `scene_type`：

| scene_type | 场景要求 | 默认报警期望 |
|---|---|---|
| `normal_standing` | 人员正常站立、走动前静止或轻微活动，不发生跌倒 | `false` |
| `sitting` | 人员主动坐下并保持坐姿，动作不应被标为跌倒 | `false` |
| `bending` | 人员弯腰、拾取物品或短暂前倾后恢复 | `false` |
| `front_fall` | 人员向摄像头方向或画面正面方向跌倒 | `true` |
| `side_fall` | 人员侧向跌倒，需记录可见侧面和摄像头角度 | `true` |
| `slow_fall` | 人员以明显较慢过程倒地或滑落 | 按实际预期填写 |
| `recovery` | 跌倒后从地面起身或恢复站立，记录恢复过程 | 按事件定义填写 |
| `occlusion` | 人体或关键点被家具、障碍物或其他物体部分遮挡 | 按实际预期填写 |
| 无人 | 画面中没有目标人员，用于验证误报 | `false` |

由于 `manifest.csv` 的 `scene_type` 字段使用固定值，`无人`场景建议使用 `normal_standing` 之外的扩展值前先同步任务索引和检查工具；第一版当前检查工具的必需集合暂不包含 `no_person`，不得擅自改动工具，应在数据集版本变更时一并更新规范。

每类至少准备多个样本，并尽量覆盖不同人员、摄像头角度、光照和背景。正例、负例和恢复样本应保持可区分，不能仅凭文件名推断标签。

## 4. 标签填写规范

数据集根目录为 `datasets/fall_eval_v1/`。

### 4.1 manifest.csv

表头必须严格为：

```text
sample_id,video_path,scene_type,fps,duration,person_id
```

填写要求：

- `sample_id`：全局唯一、稳定且不得重复；推荐格式为 `<scene_type>_<camera>_<序号>`。
- `video_path`：相对于 `fall_eval_v1/`，例如 `videos/front_fall/front_fall_cam01_001.mp4`。
- `scene_type`：使用本规范的固定场景名称。
- `fps`：视频实际 FPS，必须大于 0。
- `duration`：视频实际时长，单位秒，必须大于或等于 0。
- `person_id`：匿名且稳定的人员编号，例如 `person_001`；不得填写真实姓名。

### 4.2 labels/events.csv

表头必须严格为：

```text
sample_id,start_time,end_time,expected_alarm
```

填写要求：

- `sample_id` 必须与 manifest 中的 ID 完全一致；每个样本至少一条记录。
- `start_time`、`end_time` 使用相对于视频起点的秒数，且 `end_time >= start_time`。
- `expected_alarm` 只能填写 `true` 或 `false`，统一使用小写。
- `true` 表示该时间区间内应出现报警；`false` 表示该样本或时间区间不应报警。
- 恢复场景应把恢复相关事件单独记录，避免用“视频结束”替代实际恢复时间。

标注人员应同时保留采集批次、标注日期和标注者记录；这些附加信息可放在版本控制之外的采集登记表，不得改变上述 CSV 表头。

## 5. 样本命名规范

视频文件放在 `datasets/fall_eval_v1/videos/<scene_type>/` 下，推荐命名：

```text
<scene_type>_<camera_id>_<sample_number>.<extension>
```

示例：

```text
videos/front_fall/front_fall_cam01_001.mp4
videos/normal_standing/normal_standing_cam02_001.mp4
videos/occlusion/occlusion_cam01_001.mp4
```

规则：

- 仅使用小写字母、数字、下划线和常见视频扩展名；
- 文件名中的场景、摄像头编号和序号必须与 `sample_id` 可追溯对应；
- 同一视频不得以不同文件名重复登记；
- 不使用真实姓名、身份证号、住址或其他敏感信息；
- 文件移动或重新编码后，必须同步更新 manifest，并重新执行检查。

## 6. 数据质量检查规则

提交或用于 FD-007 验证前，必须执行：

```powershell
python check_evaluation_dataset.py --dataset datasets/fall_eval_v1
```

检查工具和人工复核共同确认：

1. `videos/`、`labels/`、`reports/` 目录存在；
2. `manifest.csv`、`events.csv` 表头严格匹配规范；
3. 所有 `sample_id` 非空且唯一，events 中的 ID 均存在于 manifest；
4. manifest 中每个视频文件存在、可访问且路径未越出数据集目录；
5. 八类必需场景均有样本；无人场景需额外登记并确认是否纳入当前版本标签集合；
6. FPS、时长、事件起止时间为有效非负数，起止时间顺序正确；
7. 每个样本都有事件标签，报警期望与场景实际内容经过人工复核；
8. 视频可完整解码，关键动作没有被截断，画面无明显损坏；
9. 采集元数据中的分辨率、FPS、时长、摄像头角度和光照条件可追溯；
10. 通过检查后才允许生成 TP、FP、FN、报警时间等评估报告。

当前目录只有空模板，因此检查结果为 `DATASET_INVALID` 是正常的；这不代表检查工具失败，而是表示真实样本和标签尚未补齐。

## 7. 版本与后续流程

数据集每次新增、删除或重标注都应更新数据集版本和变更记录。完成首轮样本与标签后，先运行检查工具，再使用固定模型、参数和标签版本执行 FD-007 Phase 1 baseline/Phase 1 对比验证。

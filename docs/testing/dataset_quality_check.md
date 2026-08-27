# fall_eval_v1 数据质量验收设计

## 1. 目的

本文件定义 `datasets/fall_eval_v1/` 进入正式评估前的最低质量门槛、无效样本规则和检查输出。它与根目录 `check_evaluation_dataset.py` 配套使用，不执行模型推理，也不修改任何检测逻辑。

## 2. 正式评估准入条件

整个数据集以及其中每个样本在进入正式评估前，必须满足以下条件：

- manifest 中引用的视频文件存在；
- 视频能够被评估环境完整读取和解码，不能只检查文件名存在；
- FPS 存在、为数值且大于 0；
- 时长存在、为数值且不小于 0，并与视频实际时长基本一致；
- `manifest.csv` 表头和每行必填字段完整；
- `events.csv` 表头和每行必填字段完整，且每个 manifest 样本至少有对应标签；
- `sample_id` 非空、全局唯一，events 中的 ID 必须能在 manifest 中找到；
- 场景信息完整，必需场景均有样本；
- 事件起止时间有效，且不超过视频实际时长；
- 数据集质量检查输出 `DATASET_OK`。

当前检查工具已覆盖目录、CSV 字段、文件存在性、FPS/时长数值、标签关联、ID 唯一性和八类必需场景。视频“可完整读取/解码”、时长与媒体元数据一致性、无人场景覆盖和样本数量仍需结合媒体工具及人工复核确认。

## 3. 样本状态与 INVALID 规则

### 3.1 INVALID

以下任一情况会使样本不能参与 TP、FP、FN 或报警时间统计，应标记为 `INVALID` 并记录原因：

- 标签缺失、`sample_id` 无法关联，或事件时间非法；
- 视频文件不存在、路径越界、无法打开或解码中途失败；
- FPS、时长等关键元数据缺失或无效；
- 场景信息缺失、无法确认场景类别，或视频内容与场景标签不符；
- 关键动作被截断，无法判断事件是否发生；
- 视频严重损坏、画面不可辨识或存在影响判断的异常；
- 同一个 `sample_id` 重复登记；
- 样本数量低于该数据集版本规定的最低数量，导致该场景无法形成有效覆盖。

`INVALID` 是样本级结果，不得用默认负例替代，也不得计入 TP、FP、FN。应在质量报告中保留 `sample_id`、原因、发现时间和处理结论。

### 3.2 INCOMPLETE

以下情况表示数据集结构可读取，但尚未达到正式评估条件，输出 `DATASET_INCOMPLETE`：

- 目录和 CSV 模板存在，但尚未填入样本或事件标签；
- 必需场景未全部覆盖；
- 样本数量不足，无法满足版本要求；
- 文件和字段基本存在，但仍有待人工复核的采集元数据或视频可读性检查；
- 存在 `INVALID` 样本，但补齐或替换后仍可继续构建该数据集版本。

`INCOMPLETE` 表示“尚不能正式评估”，不是算法误报或漏报结论。

## 4. 输出格式

数据集检查结果使用单一首行状态码：

```text
DATASET_OK
DATASET_INVALID
DATASET_INCOMPLETE
```

### DATASET_OK

表示目录、字段、样本、标签、场景覆盖和质量复核均通过，可以进入正式评估。输出应同时包含样本数量、数据集路径、数据集版本或清单摘要。

### DATASET_INVALID

表示发现明确的数据错误，例如文件不存在、视频损坏、字段错误、标签缺失或 `sample_id` 重复。输出必须列出每个错误对应的文件、行号或 `sample_id`，修复前不得进入正式评估。

### DATASET_INCOMPLETE

表示数据结构尚可读取，但样本或质量证据不足，例如空模板、场景缺失或样本数量不足。输出必须列出缺失场景、缺失样本或待复核项目。

## 5. 与 check_evaluation_dataset.py 的对应关系

运行命令：

```powershell
python check_evaluation_dataset.py --dataset datasets/fall_eval_v1
```

当前脚本直接输出 `DATASET_OK` 或 `DATASET_INVALID`：

- `DATASET_INVALID`：脚本发现目录、字段、文件、数值、标签关联、ID 或必需场景问题；
- 空模板也会输出 `DATASET_INVALID`，其中“无样本/无标签/缺少场景”在验收流程中应解释为 `DATASET_INCOMPLETE`，而不是有效数据错误；
- `DATASET_OK`：仅表示脚本检查项通过，仍需完成视频完整解码、媒体元数据一致性、无人场景和最低样本数等人工/媒体工具复核。

由于本阶段要求不修改代码，`DATASET_INCOMPLETE` 作为验收语义由质量报告记录，不改变现有检查脚本的返回行为。若后续需要脚本原生区分三种状态，应另建任务并增加测试覆盖。

## 6. 验收记录格式

每次验收建议在 `datasets/fall_eval_v1/reports/` 保存一份报告，至少包含：

```text
check_time
dataset_path
dataset_version
manifest_path
events_path
sample_count
scene_coverage
invalid_samples
incomplete_items
tool_status
manual_review_status
final_status
```

`final_status` 只有在脚本状态和人工/媒体复核均通过时才能填写 `DATASET_OK`。任何未解决的 INVALID 或关键缺项都必须阻止正式评估。

## 7. 当前数据集状态

当前 `fall_eval_v1` 仍为空模板，没有真实视频和事件标签。运行现有检查工具得到 `DATASET_INVALID` 是预期结果；按本设计的验收语义，数据集整体应记录为 `DATASET_INCOMPLETE`，直至样本、标签和媒体质量复核完成。

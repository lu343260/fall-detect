# AI-Fall-Detection-System 协作说明

## 项目定位

这是一个基于 YOLO11n-Pose 人体关键点和规则状态机的跌倒检测项目。当前主线已经从 PyTorch/Ultralytics 推理扩展到 ONNX，并包含面向龙芯 2K0300（LoongArch64）的运行时部署、固定输入尺寸、跳帧和性能记录实验。

当前处理链路是：

```text
摄像头/视频 → 预处理与姿态推理 → 关键点筛选
→ 髋部运动、身体角度、宽高比等特征 → FallDetector 状态机
→ 画面/结果输出与性能日志
```

项目目前更接近“算法与部署验证阶段”，不是已经达到实时生产部署的成品。龙芯 2K0300 上 256×256 FP32 的已记录单帧耗时约 4.1～4.2 秒，INT8 约 4.27 秒；跳帧只减少推理调用次数，不会降低单次推理延迟。

## 目录约定

- 根目录 Python 脚本：模型导出、ONNX 验证/基准、量化和通用推理入口。
- `deploy_loongson/`：较早的龙芯运行时包，使用普通输入尺寸。
- `deploy_loongson_256/`：当前重点的 256×256 龙芯运行时包和 benchmark 工具。
- `datasets/`：量化校准/评估图片等实验数据；不要把新数据默认提交到版本库。
- `docs/`：按 `development`、`benchmarks`、`testing` 分类的项目知识库，入口见 `docs/README.md`。
- 根目录 `performance_log.csv`、`int8_comparison_report.json`：已有实验输出；新的实验输出应说明平台、模型、输入尺寸、后端和参数。

## 运行与验证

项目没有统一的锁定依赖文件和测试框架。运行前应根据目标路径安装依赖；龙芯部署包的依赖说明分别在两个 `deploy_loongson*/requirements.txt` 中。

常用入口：

```powershell
# 通用 ONNX 摄像头/视频推理
python pose_video.py

# 导出固定尺寸 ONNX（默认尺寸由脚本定义）
python export_onnx.py

# ONNX 与 PyTorch 结果验证
python onnx_test.py

# 多尺寸推理基准；必须显式传入 --model SIZE=PATH
python onnx_size_benchmark.py --model 256=yolo11n-pose-256.onnx

# 量化与 FP32/INT8 对比，完整流程见 docs/benchmarks/YOLO11n-Pose-256-INT8.md
python quantize_yolo11_pose_int8.py --model yolo11n-pose-256.onnx --output yolo11n-pose-256-int8.onnx --calibration <image-dir>
python compare_yolo11_pose_int8.py --source <image-or-dir>
```

龙芯 256×256 运行时应在 `deploy_loongson_256/` 目录执行：

```bash
python3 pose_video.py --camera 0 --no-display
python3 benchmark_runner.py --runs 3 --warmup 20 --iterations 100
python3 benchmark_opencv_int8.py --model yolo11n-pose-256.onnx --int8-model yolo11n-pose-256-int8.onnx
```

验证结果必须注明是 Windows/AMD64 还是龙芯 LoongArch64，以及使用 OpenCV DNN 还是 ONNX Runtime。PC 上的推理数字不能直接当作龙芯部署结论。

## 修改原则

1. 先阅读调用链和相关日志，再修改；优先复用 `FallDetector`、结果兼容层和既有推理入口。
2. 不要把模型文件、视频、校准数据、`__pycache__` 或 benchmark 产物混入无关改动；遵循 `.gitignore`，并保留用户已有未提交文件。
3. 不要把“关键点输出一致性”写成“检测准确率”。当前 INT8 报告只代表有限图片上的输出对比，不等于 mAP、召回率或误报率。
4. 修改预处理、输入尺寸、后处理或状态机时，必须同时检查关键点坐标语义和跌倒状态转换；输入尺寸应在模型、预处理和后处理之间保持一致。
5. 修改部署代码后，至少运行语法检查或目标平台可执行的冒烟测试；性能结论需要在相同模型、输入、warmup、iteration 和线程配置下复测。
6. 默认不升级依赖、不修改 CI/格式化配置、不删除历史日志、不自动提交或推送 Git。

## 当前协作目标

- 继续以可重复 benchmark 为基础，比较输入尺寸、OpenCV DNN/ONNX Runtime、线程数、INT8 和更轻量模型。
- 建立统一的跌倒、坐下、弯腰、侧身、遮挡、无人和恢复场景回归集，补齐准确率、召回率、误报率和告警延迟证据。
- 在不积压旧帧的前提下评估“采集线程 → 最新帧 → 推理线程 → 状态机”的架构；线程解耦不能替代单帧推理优化。
- 所有新结论写入对应 `docs/` 分类，并记录日期、commit、平台、模型、输入、后端、线程、跳帧和原始输出路径。

## 敏感信息与实验数据

不要在代码、日志或文档中写入密钥、令牌、密码或真实设备凭据。实验视频、图片和模型属于较大文件，新增前先确认是否应保留在本地、归档目录或版本库外部。

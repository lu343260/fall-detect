# AI-Fall-Detection-System 小车联调交接说明

更新时间：2026-09-11

## 1. 本次交接目标

在真实小车上验证以下完整链路：

```text
小车摄像头
    → Loongson 客户端压缩为 JPEG
队长电脑 :8000/infer
    → YOLO11n-Pose 远程推理
Loongson FallDetector
    → 跌倒状态与跟踪特征
UDP JSON :9001
    → 运动进程
```

当前 Git 分支：

```text
codex/loongson-remote-client
```

当前虚拟机已经完成单张图片接口验证：ONNX Runtime 1.29.0、
`CPUExecutionProvider`、`POST /infer` 返回 200、测试图检测到 3 人且每人
17 个关键点。记录的服务端单张人体图推理时间约为 119 ms，黑图约为
146 ms。这些结果只证明接口和关键点数据链路可用，不代表跌倒准确率或
小车端到端实时性能。

## 2. 设备分工

- 队长电脑：运行 FastAPI 姿态推理服务器和 ONNX 模型。
- 小车 Loongson：读取小车摄像头、请求服务器、运行 `FallDetector`、发送
  UDP 特征。
- 运动进程：读取 UDP 特征并执行实际控制和安全停止逻辑。

小车不能把服务端地址写成 `127.0.0.1`，除非推理服务器也运行在小车
本机。正常联调时必须使用队长电脑可被小车访问的局域网 IP。

## 3. 获取代码

在队长电脑或小车上获取交接分支：

```bash
git clone --depth 1 --branch codex/loongson-remote-client https://github.com/lu343260/fall-detect.git
```

关键文件：

```text
AI-Fall-Detection-System/
├── collect_deployment_info.py
├── yolo11n-pose-256.onnx
├── deploy_server/
│   ├── server_inference.py
│   ├── benchmark_server_inference.py
│   ├── test_server_inference.py
│   ├── requirements.txt
│   ├── README.md
│   └── TEAM_HANDOFF.md
└── deploy_loongson_256/
    ├── pose_video.py
    ├── onnx_inference.py
    ├── tracking_features.py
    ├── motion_feature_receiver.py
    ├── fall_detector.py
    ├── detection_result.py
    ├── performance_logger.py
    ├── requirements.txt
    └── README.md
```

不要提交测试视频、摄像头图片、`__pycache__`、密码、令牌或设备凭据。

## 4. 队长电脑启动推理服务器

进入项目根目录，安装服务端依赖：

```bash
python -m pip install -r deploy_server/requirements.txt
```

启动服务：

```bash
python deploy_server/server_inference.py --model yolo11n-pose-256.onnx --host 0.0.0.0 --port 8000
```

另开一个终端检查服务：

```bash
curl http://127.0.0.1:8000/health
```

预期结果：

```json
{"status":"ok"}
```

记录队长电脑的实际局域网 IP，并确认操作系统防火墙允许小车访问 TCP
`8000` 端口：

```text
服务器 IP：____________________
服务端口：8000
```

## 5. 小车端检查摄像头

在小车 Loongson 终端执行：

```bash
ls /dev/video*
```

至少应出现一个摄像头设备，例如 `/dev/video0`。如果没有设备，先处理
摄像头供电、连接、驱动或容器/虚拟机设备透传，程序无法替代这一前置条件。

进入客户端目录并检查语法：

```bash
cd deploy_loongson_256
python3 -m py_compile pose_video.py onnx_inference.py tracking_features.py motion_feature_receiver.py
```

## 6. 启动 UDP 参考接收器

在小车终端一执行：

```bash
python3 motion_feature_receiver.py --host 127.0.0.1 --port 9001
```

启动时应显示：

```text
[MOTION] STOP (awaiting valid tracking data)
```

参考接收器会在以下情况下打印 STOP：

- 尚未收到有效目标；
- `target_valid=false`；
- 超过 0.5 秒没有收到新消息；
- `fall=true`。

该脚本只是协议验证工具，只打印结果，不会直接控制电机。真正的运动模块
必须实现对应的停止或保护动作。

## 7. 启动小车远程客户端

在小车终端二执行；将 `<SERVER_IP>` 替换为队长电脑的实际局域网 IP：

```bash
python3 pose_video.py --inference-mode remote --server-url http://<SERVER_IP>:8000/infer --timeout 10 --camera 0 --frame-skip 3 --no-display
```

如果摄像头画面方向已经正确，再加上：

```text
--no-rotate
```

客户端成功运行后，应同时看到：

- 客户端持续输出性能和跌倒状态；
- 服务端持续收到 `POST /infer` 且返回 200；
- UDP 接收器持续收到目标位置、距离比例和 `fall` 状态。

## 8. 必须完成的实车场景

- [ ] 正常站立：持续检测到目标，`fall=false`。
- [ ] 左右移动：`horizontal_error` 随人物方向合理变化。
- [ ] 远近移动：`distance_ratio` 随人物框高度合理变化。
- [ ] 无人或遮挡：`target_valid=false`，运动端进入 STOP。
- [ ] 模拟跌倒：状态机进入跌倒状态，`fall=true`，运动端进入 STOP。
- [ ] 跌倒后恢复：记录状态恢复过程和所需时间。
- [ ] 网络断开：超过 0.5 秒无有效数据，运动端进入 STOP。

测试跌倒动作时应确保人员和小车处于安全环境，不直接带载验证未经确认的
电机控制逻辑。

## 9. 队长需要回传的数据

大部分环境信息不用手填。请分别在项目根目录运行一次采集脚本，并把生成的
两个 JSON 文件发回。

队长电脑运行：

```bash
python collect_deployment_info.py --role server --output handoff_server.json
```

小车运行；将 `<SERVER_IP>` 换成队长电脑的局域网 IP：

```bash
python3 collect_deployment_info.py --role robot --server-url http://<SERVER_IP>:8000/infer --output handoff_robot.json
```

脚本会自动记录日期、Git 分支和 commit、系统与架构、Python、OpenCV、
ONNX Runtime、可用 Execution Provider、局域网 IP、`/health` 状态、摄像头
设备、实际采集分辨率和 `frame-skip`。报告不包含密码或令牌。

真实动作无法由配置脚本自动判断，只需人工填写下面这份简化记录：

```text
验证结果：
- 站立：
- 左右移动：
- 远近移动：
- 无人/遮挡：
- 模拟跌倒：
- 跌倒恢复：
- 网络断开：

性能：
- 服务端平均 inference_time_ms（从服务端日志复制）：
- 小车端 inference_ms / camera_fps / inference_fps（从客户端日志复制）：

异常和日志路径：
```

同时保留以下原始材料：

- 服务端启动与请求日志；
- 小车客户端至少 60 秒的控制台输出；
- UDP 接收器输出；
- `performance_log.csv`；
- 每个测试场景的短视频或关键截图，存放在版本库外。

## 10. 当前限制

- 当前默认使用第一个检测到的人，尚未实现稳定的多人目标跟踪。
- UDP 参考接收器不直接控制电机，真实运动控制闭环尚未验证。
- 目前没有准确率、召回率、误报率或告警延迟的完整实车结论。
- 远程性能必须记录端到端网络往返时间，不能只使用服务端模型推理时间。

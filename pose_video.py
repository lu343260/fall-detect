from onnx_inference import ONNXPoseDetector
import cv2
from fall_detector import FallDetector
import time
import platform
from collections import deque
from performance_logger import PerformanceLogger

# 根据运行平台自动选择模型和输入尺寸。
PLATFORM_MACHINE = platform.machine().lower()
if "loongarch" in PLATFORM_MACHINE:
    MODEL_NAME = "yolo11n-pose-256.onnx"
    INPUT_SIZE = 256
else:
    MODEL_NAME = "yolo11n-pose.onnx"
    INPUT_SIZE = 640

FRAME_SKIP = 1
frame_count = 0
camera_frame_times = deque(maxlen=30)
camera_fps = 0.0
#yolo推理统计
inference_count = 0
ai_start_time = time.perf_counter()
ai_fps = 0.0

inference_time_ms = 0.0
last_result = None

performance_logger = PerformanceLogger(
    "performance_log.csv",
    interval_seconds=5,
    model_name=MODEL_NAME,
    input_size=INPUT_SIZE,
)
total_inference_count = 0
inference_times_ms = []

DEBUG = False  # True 时打印每帧状态，方便调参


# 加载姿态模型
model = ONNXPoseDetector(MODEL_NAME)
fall_detector = FallDetector()


# 打开摄像头
video=cv2.VideoCapture(0)

video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cv2.namedWindow(
    "YOLO Pose",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "YOLO Pose",
    640,
    480
)


while True:
    frame_count += 1

    ret, frame = video.read()

    if not ret:
        break

    # 使用最近 30 次成功读取的时间戳，统计实时摄像头 FPS。
    camera_frame_times.append(time.perf_counter())
    if len(camera_frame_times) >= 2:
        time_span = camera_frame_times[-1] - camera_frame_times[0]
        if time_span > 0:
            camera_fps = (len(camera_frame_times) - 1) / time_span

    frame = cv2.rotate(frame, cv2.ROTATE_180)
    # 翻转画面，避免镜像
    #记录这一帧被读取时的单调时间
    # 记录当前帧被读取时的单调时间，供速度和持续时间计算使用。
    capture_time = time.monotonic()

    # 默认显示原始画面，即使当前帧没有检测到人
    annotated_frame = frame.copy()

    # 固定跳帧：摄像头仍然逐帧读取，但只在每 FRAME_SKIP 帧执行一次 YOLO。
    if frame_count % FRAME_SKIP == 0:
        inference_start = time.perf_counter()
        results = model(frame)
        inference_count += 1
        total_inference_count += 1
        inference_time_ms = (time.perf_counter() - inference_start) * 1000
        inference_times_ms.append(inference_time_ms)

        # 获取关键点
        keypoints_xy = results[0].keypoints.xy
        keypoints_conf = results[0].keypoints.conf

        # 默认认为当前帧没有可用人体，保证后面的显示逻辑仍然执行
        person = None
        points_valid = False

        if len(keypoints_xy) > 0:
            annotated_frame = results[0].plot()

            person = keypoints_xy[0]
            confidence = keypoints_conf[0]

            # 跌倒判断依赖肩膀和髋部，先确认这些关键点足够可靠。
            required_points = [5, 6, 11, 12]
            points_valid = all(
                confidence[i] >= 0.5
                for i in required_points
            )

        # 保持原有状态机调用条件和 FallDetector 逻辑不变。
        if points_valid and frame_count % 3 == 0:
            last_result = fall_detector.detect(person, capture_time)
            print(
                f"状态:{last_result.state} | "
                f"跌倒:{last_result.fall}"
            )
            print(f"髋间距离:{last_result.hip_width:.1f}")
            if last_result is not None and DEBUG:
                data = last_result.to_dict()
                print(
                    f"状态:{data['state']} | "
                    f"髋部({data['hip_x']:.1f},{data['hip_y']:.1f}) | "
                    f"宽度:{data['hip_width']:.1f} | "
                    f"速度:{data['speed']:.1f} | "
                    f"下降:{data['hip_drop']:.1f}"
                )

            if DEBUG:
                print("----------------")

    ai_elapsed = time.perf_counter() - ai_start_time
    if ai_elapsed >= 1:
        ai_fps = inference_count / ai_elapsed
        inference_count = 0
        ai_start_time = time.perf_counter()
    cv2.putText(
        annotated_frame,
        f"CameraFPS: {camera_fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    # 每 5 秒保存一次性能数据，用于 LoongArch 平台 AI 推理性能测试。
    current_time = time.perf_counter()
    if performance_logger.should_log(current_time):
        performance_logger.write(
            FRAME_SKIP,
            camera_fps,
            ai_fps,
            total_inference_count,
            inference_times_ms,
            current_time,
        )
    cv2.putText(
        annotated_frame,
        f"Inference Time: {inference_time_ms:.1f} ms",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        annotated_frame,
        f"Skip: {FRAME_SKIP}",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        annotated_frame,
        f"AI FPS: {ai_fps:.1f}",
        (10, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    # 绘制骨架
    if annotated_frame is not None:
        # 显示
        cv2.imshow(
            "YOLO Pose",
            annotated_frame
        )

    # 按q退出
    if cv2.waitKey(1) & 0xff == ord('q'):
            break


try:
    video.release()
    cv2.destroyAllWindows()
finally:
    performance_logger.close()

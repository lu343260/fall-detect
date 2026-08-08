from ultralytics import YOLO
import cv2
from fall_detector import FallDetector
import time

frame_count=0

DEBUG = False  # True 时打印每帧状态，方便调参


# 加载姿态模型
model = YOLO("yolov8n-pose.pt")
fall_detector = FallDetector()


# 打开摄像头
video=cv2.VideoCapture(1)

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
    frame_count+=1

    ret, frame = video.read()

    if not ret:
        break

    frame = cv2.rotate(frame, cv2.ROTATE_180)
    # 翻转画面，避免镜像
    #记录这一帧被读取时的单调时间
    # 记录当前帧被读取时的单调时间，供速度和持续时间计算使用。
    capture_time = time.monotonic()

    # 默认显示原始画面，即使当前帧没有检测到人
    annotated_frame = frame.copy()

    # YOLO姿态检测
    results = model(
        frame,
        conf=0.3,
        classes=[0],
        verbose=False
    )

    # 获取关键点
    keypoints_xy = results[0].keypoints.xy
    keypoints_conf = results[0].keypoints.conf

    # 默认认为当前帧没有可用人体，保证后面的显示逻辑仍然执行
    person = None
    points_valid = False

    if len(keypoints_xy) > 0:
    # 置信度检查
    # 跌倒检测
        annotated_frame = results[0].plot()

        person = keypoints_xy[0].cpu().numpy()
        confidence = keypoints_conf[0].cpu().numpy()

    # 跌倒判断依赖肩膀和髋部，先确认这些关键点足够可靠。
        required_points = [5, 6, 11, 12]

        points_valid = all(
            confidence[i] >= 0.5
            for i in required_points
        )


    if points_valid and frame_count % 3 == 0:

        result = fall_detector.detect(
             person,
             capture_time
             )
        print(
        f"状态:{result.state} | "
        f"跌倒:{result.fall}"
        )
        print(
        f"髋间距离:{result.hip_width:.1f}"
        )
        if result is not None and DEBUG:
            data = result.to_dict()
            print(
                f"状态:{data['state']} | "
                f"髋部({data['hip_x']:.1f},{data['hip_y']:.1f}) | "
                f"宽度:{data['hip_width']:.1f} | "
                f"速度:{data['speed']:.1f} | "
                f"下降:{data['hip_drop']:.1f}"
            )

        if DEBUG:
            print("----------------")
        

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


video.release()
cv2.destroyAllWindows()

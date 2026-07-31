from ultralytics import YOLO
import cv2
from fall_detector import FallDetector
import time

frame_count=0


# 加载姿态模型
model = YOLO("yolov8s-pose.pt")
fall_detector = FallDetector()


# 打开视频
video = cv2.VideoCapture("test_person2.mp4")


while True:
    frame_count+=1

    ret, frame = video.read()

    if not ret:
        break
    #记录这一帧被读取时的单调时间
    capture_time = time.monotonic()

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

    if len(keypoints_xy) == 0:
        continue

    person = keypoints_xy[0].cpu().numpy()
    confidence = keypoints_conf[0].cpu().numpy()

    required_points = [5, 6, 11, 12]

    if len(keypoints_xy) > 0:
    # 置信度检查
    # 跌倒检测
        annotated_frame = results[0].plot()
        cv2.imshow("YOLO Pose", annotated_frame)


    if frame_count % 3 == 0 :

        fallen = fall_detector.detect(
             person
             
             )

        if fallen:
            print("⚠️ 检测到可能跌倒")

        print("----------------")
        

    # 绘制骨架
    annotated_frame = results[0].plot()

    small_frame = cv2.resize(
    annotated_frame,
    (800,600)
)


    # 显示
    cv2.imshow(
        "YOLO Pose",
        small_frame
    )


    # 按q退出
    if cv2.waitKey(1) & 0xff == ord('q'):
            break


video.release()
cv2.destroyAllWindows()
from ultralytics import YOLO
import cv2
from fall_detector import FallDetector


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


    # YOLO姿态检测
    results = model(
        frame,
        conf=0.3,
        classes=[0],
        verbose=False
    )

    # 获取关键点
    keypoints = results[0].keypoints.xy


    if len(keypoints) > 0 and frame_count % 30 == 0 :

        person = keypoints[0]

        fallen = fall_detector.detect(person)

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
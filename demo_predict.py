from ultralytics import YOLO

# 1. 加载模型（这行代码会自动在你当前的文件夹下找 yolov8n.pt，如果没有它会自动去下）
model = YOLO("yolov8n.pt")

# 2. 预测你电脑里的某张图片（把路径换成你之前测试成功的那张 bus.jpg 路径）
results = model("C:/Users/pc/bus.jpg",save=True,conf=0.05)

# 3. 让识别结果在屏幕上弹窗显示出来
for r in results:
    r.show()
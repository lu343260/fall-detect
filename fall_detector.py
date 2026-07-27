import numpy as np
import time

class FallDetector:

    def __init__(self):
        # 记录疑似跌倒开始时间
        self.fall_start_time = None
        self.previous_hip_y = None
        self.previous_time = None


    def calculate_body_ratio(self, keypoints):
        """
        计算人体宽高比例
        """

        x_points = keypoints[:, 0]
        y_points = keypoints[:, 1]


        width = x_points.max() - x_points.min()

        height = y_points.max() - y_points.min()


        if height == 0:
            return 0


        ratio = width / height

        return ratio


    def calculate_hip_speed(self, keypoints):
        """
        计算髋部关键点的速度
        """

        # 获取左髋和右髋关键点
        left_hip = keypoints[11]
        right_hip = keypoints[12]

        # 计算髋部关键点的平均位置
        hip_y = (left_hip[1] + right_hip[1]) / 2

        current_time = time.time()

        speed = 0

        if self.previous_hip_y is not None :

            dy = hip_y - self.previous_hip_y

            dt = current_time - self.previous_time

            if dt > 0:

                speed = max(dy / dt, 0)


        self.previous_hip_y = hip_y
        self.previous_time = current_time

        return speed

    def calculate_body_angle(self, keypoints):
        """
        计算身体的倾斜角度
        """

        #左右肩
        left_shoulder = keypoints[5]

        right_shoulder = keypoints[6]

        #左右髋
        left_hip = keypoints[11]

        right_hip = keypoints[12]

        #肩中心
        shoulder_x=(left_shoulder[0] + right_shoulder[0]) / 2

        shoulder_y=(left_shoulder[1] + right_shoulder[1]) / 2

        #髋中心
        hip_x=(left_hip[0] + right_hip[0]) / 2

        hip_y=(left_hip[1] + right_hip[1]) / 2

        #向量
        dx=shoulder_x - hip_x

        dy=shoulder_y - hip_y

        #防止除0
        if dy == 0:
            return 90

        angle = np.arctan(abs(dx) / abs(dy))

        #弧度转角度
        angle = np.degrees(angle)

        return angle


    def detect(self, keypoints):
    
            ratio = self.calculate_body_ratio(keypoints)
            hip_speed = self.calculate_hip_speed(keypoints)
            angle = self.calculate_body_angle(keypoints)

            print("髋部下降速度:", hip_speed)
    
    
            print("身体宽高比例:", ratio)


            print("身体倾斜角度:", angle)
    
    
            # 横向程度超过阈值
            if (
                ratio > 1.0
                and angle > 60
                and hip_speed > 10
            ):
    
    
                # 第一次发现异常姿态
                if self.fall_start_time is None:
                    self.fall_start_time = time.time()
    
    
                duration = time.time() - self.fall_start_time
    
    
                print("疑似跌倒持续时间:", duration)
    
    
                if duration > 3:
                    return True
    
    
            else:
    
                # 恢复正常姿态
                self.fall_start_time = None
    
    
            return False

import numpy as np
import time

class FallState:
    NORMAL = 0
    FALLING = 1
    ON_GROUND = 2

class FallDetector:

    def __init__(self):
        # 当前状态
        self.state = FallState.NORMAL

        # 跌倒开始时间
        self.fall_start_time = None

        #状态确认计数
        self.falling_frames = 0
        self.ground_frames = 0

        # 髋部速度计算
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
    
            ratio = float(self.calculate_body_ratio(keypoints))
            speed = float(self.calculate_hip_speed(keypoints))
            angle = float(self.calculate_body_angle(keypoints))


            print("-----------------------")


            print("状态", self.state)


            print("髋部下降速度:", speed)


            print("身体宽高比例:", ratio)


            print("身体倾斜角度:", angle)


            #当前正常状态
            if self.state == FallState.NORMAL:

                #开始跌倒
                if  speed > 1 :

                    self.falling_frames += 1
                    print("falling_frames:", self.falling_frames)
                else:

                    self.falling_frames = 0

                if self.falling_frames >= 3:
                    self.state = FallState.FALLING

                    self.fall_start_time = time.time()

                    self.falling_frames = 0

                    print("检测到跌倒开始")

            #正在跌倒
            elif self.state == FallState.FALLING:

               duration = time.time() - self.fall_start_time

                 #已经倒地
               if ratio > 1 and angle > 60 and duration > 1:
                    self.ground_frames += 1

               else:
                    self.ground_frames = 0

                    if angle <30:
                    
                        self.state = FallState.NORMAL
                        self.fall_start_time = None
                        self.falling_frames = 0
                        print("检测恢复正常")

               if self.ground_frames >= 3:

                    self.state = FallState.ON_GROUND
                    self.ground_frames = 0
                    print("检测到已经倒地")

            #恢复正常
            

               

            #已经倒地
            elif self.state == FallState.ON_GROUND:

                #人再站起来

                if angle < 40 and ratio <1:

                    self.state = FallState.NORMAL

                    self.fall_start_time = None

                    print("检测到人已经站起来")

            return self.state == FallState.ON_GROUND



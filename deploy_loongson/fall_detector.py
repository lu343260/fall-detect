import numpy as np
from collections import deque
from detection_result import DetectionResult
from enum import Enum

class FallState(Enum):
    # 程序刚启动时，先收集稳定的人体姿态，不进行跌倒判断。
    INITIALIZING = -1
    NORMAL = 0
    FALLING = 1
    ON_GROUND = 2

class FallDetector:

    def __init__(self):
        # 当前状态
        # 初始状态不是 NORMAL，避免程序启动瞬间误报警。
        self.state = FallState.INITIALIZING

        # 跌倒开始时间
        self.fall_start_time = None

        #状态确认计数
        self.falling_frames = 0
        self.ground_frames = 0

        # 恢复确认计数
        self.recovery_frames = 0
        self.recovery_confirm_frames = 3  #连续满足恢复条件次数

        # 髋部速度计算
        self.previous_hip_y = None
        self.previous_time = None
        # 初始化阶段收集的髋部 y 坐标样本。
        self.hip_y_samples = []
        # 稳定初始化完成后，用中位数保存正常姿态下的髋部高度。
        self.baseline_hip_y = None
        self.calibration_sample_count = 10
        # hip_y 样本的标准差小于该值，才认为人体姿态足够稳定。
        self.calibration_std_threshold = 10.0

        # 跌倒判断阈值，集中放在这里方便后续用测试视频调参。
        self.speed_threshold = 1.0  #髋部下降速度
        self.drop_threshold = 40.0  #髋部相对基准
        self.ground_angle_threshold = 45.0    #身体倾斜角度
        self.ground_duration_threshold = 1.0  #疑似跌倒状态持续时间
        self.falling_confirm_frames = 2  #满足快速下降条件次数
        self.ground_confirm_frames = 3   #满足倒地条件次数

        self.hip_y_history = deque(maxlen=5)

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


    def calculate_hip_y(self, keypoints):
        """
        返回当前帧左右髋部中心的 y 坐标
        """
        return float((keypoints[11][1] + keypoints[12][1]) / 2)


    def get_smooth_hip_y(self, keypoints):
        """
        返回当前帧左右髋部中心的平滑 y 坐标
        """
        hip_y = self.calculate_hip_y(keypoints)
        self.hip_y_history.append(hip_y)
        smooth_hip_y = np.mean(self.hip_y_history)
        return smooth_hip_y

    def collect_baseline_sample(self, keypoints):
        """收集初始化样本，并在样本稳定后建立 baseline。"""
        hip_y = self.get_smooth_hip_y(keypoints)
        self.hip_y_samples.append(hip_y)  #相当于list.append()，在列表末尾添加元素

        print(
            f"初始化样本: {len(self.hip_y_samples)}"
            f"/{self.calibration_sample_count}"
        )

        if len(self.hip_y_samples) < self.calibration_sample_count:
            return False

        # 标准差越小，说明这段时间人体的髋部高度越稳定。
        hip_y_std = float(np.std(self.hip_y_samples))  #计算标准差

        if hip_y_std < self.calibration_std_threshold:
            # 中位数比单帧值更不容易受到关键点偶发抖动影响。
            self.baseline_hip_y = float(np.median(self.hip_y_samples))  #取中间值
            self.hip_y_samples.clear()
            self.state = FallState.NORMAL
            self.previous_hip_y = None
            self.previous_time = None
            self.falling_frames = 0
            self.ground_frames = 0
            self.recovery_frames = 0

            print("初始化成功")
            print("baseline_hip_y:", self.baseline_hip_y)
            print("hip_y_std:", hip_y_std)
            return True

        self.hip_y_samples.clear()
        print("初始化不稳定，重新采样")
        print("hip_y_std:", hip_y_std)
        return False


    def calculate_hip_speed(self, keypoints, current_time):
        """
        计算髋部关键点的速度
        """

        # 直接用原始髋部 y（不经过平滑），保证快速下落能被测到。
        hip_y = self.calculate_hip_y(keypoints)

        speed = 0.0

        if self.previous_hip_y is not None:

            dy = hip_y - self.previous_hip_y

            dt = current_time - self.previous_time

            if dt > 0:
                # y 变大表示向下移动；只保留向下速度，过滤向上移动。
                speed = dy / dt
                speed = max(speed, 0.0)
                speed = min(speed, 50.0)  # 限制最大速度，避免异常值影响判断。

        self.previous_hip_y = hip_y
        self.previous_time = current_time

        return speed

    def calculate_hip_drop(self, keypoints):
        """
        计算髋部垂直位移
        """
        left_hip = keypoints[11]
        right_hip = keypoints[12]

        # 当前帧髋部中心相对于初始化基准的位置。
        hip_y = self.get_smooth_hip_y(keypoints)

        if self.baseline_hip_y is None:
            return 0.0

        # 图像坐标中向下为正，因此向下移动时 hip_drop 为正值。
        drop = hip_y - self.baseline_hip_y

        return drop


    def update_baseline(self, keypoints):
        """在 NORMAL 状态下缓慢跟随人体位置更新 baseline。"""
        # current_hip_y 表示当前这一帧计算得到的髋部中心 y 坐标，
        # 不是固定不变的基准值。
        current_hip_y = self.calculate_hip_y(keypoints)
        # alpha 越小，baseline 更新越慢，越不容易被瞬时动作带偏。
        alpha = 0.05

        if self.baseline_hip_y is None:
            self.baseline_hip_y = current_hip_y
            return

        self.baseline_hip_y = (
            (1 - alpha) * self.baseline_hip_y
            + alpha * current_hip_y
        )

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

    def calculate_hip_info(self, keypoints):

        left_hip = keypoints[11]
        right_hip = keypoints[12]

        hip_x = (
        left_hip[0] + right_hip[0]
        ) / 2

        hip_y = (
        left_hip[1] + right_hip[1]
        ) / 2

        hip_width = abs(
        right_hip[0] - left_hip[0]
        )

        return hip_x, hip_y, hip_width


    def detect(self, keypoints, current_time):
        # 初始化阶段只收集样本，禁止速度、角度和 hip_drop 参与判断。
        if self.state == FallState.INITIALIZING:
            self.collect_baseline_sample(keypoints)
            result = DetectionResult()
            result.state = self.state.value
            result.initializing = True
            return result

        ratio = float(self.calculate_body_ratio(keypoints))
        speed = float(self.calculate_hip_speed(keypoints, current_time))
        angle = float(self.calculate_body_angle(keypoints))
        hip_drop = float(self.calculate_hip_drop(keypoints))
        hip_x, hip_y, hip_width = self.calculate_hip_info(keypoints)

        # 状态机：三种状态平级分支
        if self.state == FallState.NORMAL:
            self.update_baseline(keypoints)
            if (
                speed > self.speed_threshold
                and hip_drop > self.drop_threshold
            ):
                self.falling_frames += 1
            else:
                self.falling_frames = max(
                    0,
                    self.falling_frames - 1
                )

            if self.falling_frames >= self.falling_confirm_frames:
                self.state = FallState.FALLING
                self.fall_start_time = current_time
                self.falling_frames = 0
                self.recovery_frames = 0
                print("检测到跌倒开始")

        elif self.state == FallState.FALLING:
            duration = current_time - self.fall_start_time
            if (
                angle > self.ground_angle_threshold
                and hip_drop > self.drop_threshold
                and duration > self.ground_duration_threshold
            ):
                self.ground_frames += 1
                self.recovery_frames = 0
            else:
                self.ground_frames = 0
                if angle < 30 and hip_drop < 20:
                    self.recovery_frames += 1
                    if self.recovery_frames >= self.recovery_confirm_frames:
                        self.state = FallState.NORMAL
                        self.fall_start_time = None
                        self.falling_frames = 0
                        self.ground_frames = 0
                        self.recovery_frames = 0
                        self.hip_y_history.clear()
                        print("检测恢复正常")
                else:
                    self.recovery_frames = 0

            if self.ground_frames >= self.ground_confirm_frames:
                self.state = FallState.ON_GROUND
                self.ground_frames = 0
                self.recovery_frames = 0
                print("检测到已经倒地")

        elif self.state == FallState.ON_GROUND:
            if angle < 40 and ratio < 0.8:
                self.recovery_frames += 1
                if self.recovery_frames >= self.recovery_confirm_frames:
                    self.state = FallState.NORMAL
                    self.fall_start_time = None
                    self.falling_frames = 0
                    self.ground_frames = 0
                    self.recovery_frames = 0
                    self.hip_y_history.clear()
                    print("检测到人已经站起来")
            else:
                self.recovery_frames = 0

        result = DetectionResult()
        result.state = self.state.value
        result.fall = (
            self.state == FallState.FALLING
            or self.state == FallState.ON_GROUND
        )
        result.hip_x = hip_x
        result.hip_y = hip_y
        result.hip_width = hip_width
        result.angle = angle
        result.speed = speed
        result.ratio = ratio
        result.hip_drop = hip_drop
        return result

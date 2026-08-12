class DetectionResult:

    def __init__(self):
        
        # 状态
        self.state = None
        self.fall = False
        # 是否处于初始化采样阶段
        self.initializing = False

        # 髋部信息
        self.hip_x = 0
        self.hip_y = 0
        self.hip_width = 0

        # 身体特征
        self.angle = 0
        self.speed = 0
        self.ratio = 0
        self.hip_drop = 0

        # 置信度
        self.confidence = 0


    def to_dict(self):

        return {
            "state": self.state,
            "fall": self.fall,

            "hip_x": self.hip_x,
            "hip_y": self.hip_y,
            "hip_width": self.hip_width,

            "angle": self.angle,
            "speed": self.speed,
            "ratio": self.ratio,
            "hip_drop": self.hip_drop,

            "confidence": self.confidence
        }
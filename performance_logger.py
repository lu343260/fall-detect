"""跨平台 AI 推理性能日志模块。

该模块用于 PC 和 LoongArch 平台的 AI 推理性能测试与优化报告。
仅使用 Python 标准库，可直接在龙芯 Linux 环境运行。
"""

import csv
import platform
import time
from datetime import datetime
from pathlib import Path


class PerformanceLogger:
    """按固定时间间隔将性能数据追加保存到 CSV 文件。"""

    def __init__(self, path="performance_log.csv", interval_seconds=5,
                 model_name="unknown", input_size="unknown"):
        self.path = Path(path)
        self.interval_seconds = interval_seconds
        self.model_name = model_name
        self.input_size = (
            f"{input_size}x{input_size}"
            if isinstance(input_size, int)
            else str(input_size)
        )
        self.start_time = time.perf_counter()
        self.last_log_time = self.start_time

        self.file = self.path.open("a", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        if self.file.tell() == 0:
            self.writer.writerow([
                "timestamp",
                "platform_system",
                "platform_machine",
                "platform_release",
                "python_version",
                "model_name",
                "input_size",
                "frame_skip",
                "camera_fps",
                "ai_fps",
                "inference_count",
                "avg_inference_time_ms",
                "max_inference_time_ms",
                "min_inference_time_ms",
                "runtime_seconds",
            ])
            self.file.flush()

    def should_log(self, current_time=None):
        """判断是否达到下一次记录时间。"""
        current_time = time.perf_counter() if current_time is None else current_time
        return current_time - self.last_log_time >= self.interval_seconds

    def write(self, frame_skip, camera_fps, ai_fps, inference_count,
              inference_times_ms, current_time=None):
        """写入一行性能数据，并返回本次记录的单调时钟时间。"""
        current_time = time.perf_counter() if current_time is None else current_time
        times = list(inference_times_ms)
        avg_time = sum(times) / len(times) if times else 0.0
        max_time = max(times) if times else 0.0
        min_time = min(times) if times else 0.0

        self.writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            platform.system(),
            platform.machine(),
            platform.release(),
            platform.python_version(),
            self.model_name,
            self.input_size,
            frame_skip,
            f"{camera_fps:.2f}",
            f"{ai_fps:.2f}",
            inference_count,
            f"{avg_time:.2f}",
            f"{max_time:.2f}",
            f"{min_time:.2f}",
            f"{current_time - self.start_time:.2f}",
        ])
        self.file.flush()
        self.last_log_time = current_time
        return current_time

    def close(self):
        """关闭 CSV 文件。"""
        if not self.file.closed:
            self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

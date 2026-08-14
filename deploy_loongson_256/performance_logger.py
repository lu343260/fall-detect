"""CSV performance logger used by the standalone 256x256 deployment."""
import csv
import platform
import time
from datetime import datetime
from pathlib import Path


class PerformanceLogger:
    """Append periodic inference statistics using only the Python standard library."""

    def __init__(self, path="performance_log.csv", interval_seconds=5,
                 model_name="unknown", input_size="unknown"):
        self.path = Path(path)
        self.interval_seconds = interval_seconds
        self.model_name = model_name
        self.input_size = (
            f"{input_size}x{input_size}" if isinstance(input_size, int)
            else str(input_size)
        )
        self.start_time = time.perf_counter()
        self.last_log_time = self.start_time
        self.file = self.path.open("a", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        if self.file.tell() == 0:
            self.writer.writerow([
                "timestamp", "platform_system", "platform_machine",
                "platform_release", "python_version", "model_name",
                "input_size", "frame_skip", "total_frames", "camera_fps",
                "ai_fps", "inference_count", "avg_inference_time_ms",
                "fall_detection_result", "runtime_seconds",
            ])
            self.file.flush()

    def write(self, frame_skip, camera_fps, ai_fps, inference_count,
              inference_times_ms, current_time=None, total_frames=0,
              fall_detection_result="unknown"):
        current_time = time.perf_counter() if current_time is None else current_time
        times = list(inference_times_ms)
        average = sum(times) / len(times) if times else 0.0
        self.writer.writerow([
            datetime.now().isoformat(timespec="seconds"), platform.system(),
            platform.machine(), platform.release(), platform.python_version(),
            self.model_name, self.input_size, frame_skip, total_frames,
            f"{camera_fps:.2f}", f"{ai_fps:.2f}", inference_count,
            f"{average:.2f}", fall_detection_result,
            f"{current_time - self.start_time:.2f}",
        ])
        self.file.flush()
        self.last_log_time = current_time

    def close(self):
        if not self.file.closed:
            self.file.close()

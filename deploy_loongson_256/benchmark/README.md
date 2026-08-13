# OpenCV DNN 自动化性能测试

`benchmark_runner.py` 会复用 `benchmark_opencv_forward.py`，自动测试以下
OpenCV 线程配置：`0、1、2、4、8、12`。

其中：

- `0` 表示保持 OpenCV 默认线程配置；
- 其他数字会调用 `cv2.setNumThreads()`；
- 每组默认执行 1 次 benchmark，每次 benchmark 默认包含 20 次 warmup 和
  100 次 iteration；
- 延迟统计范围为 `setInput()` + `forward()`。

## 运行

在 `deploy_loongson_256` 目录执行：

```bash
python3 benchmark_runner.py
```

指定每组重复次数和每次采样次数：

```bash
python3 benchmark_runner.py --runs 3 --warmup 20 --iterations 100
```

只测试指定线程配置：

```bash
python3 benchmark_runner.py --threads 0 1 2 4
```

结果默认写入：

```text
benchmark/results/benchmark_result.csv
```

CSV 字段为：

```text
model_name,input_size,thread_num,avg_latency,min_latency,max_latency,fps,timestamp
```

也可以指定模型和输出文件：

```bash
python3 benchmark_runner.py \
  --model yolo11n-pose-256.onnx \
  --input-size 256 \
  --output benchmark/results/benchmark_result.csv
```

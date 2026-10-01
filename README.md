# T2520738
Android YOLO object detection experiment for CPU vs Android NNAPI acceleration.

Model: best.onnx, input 1x3x640x640, output 1x23x8400, 19 classes.

The NPU button requests ONNX Runtime NNAPI with CPU_DISABLED. This demonstrates a non-CPU NNAPI path request, but Android/NNAPI does not expose a portable per-app NPU-utilization percentage. The app therefore reports accelerator session status, CPU utilization, FPS, and inference latency rather than inventing NPU %.

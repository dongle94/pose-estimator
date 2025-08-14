# pose-estimator
Human Detection + Pose Estimation based Project
- YOLO Detector + HRNet or RTMPose


## Install (Test Evironments)
- Ubuntu 20.04, 22.04 (test version)
- python >= 3.8.x
  - test: 3.8.x / 3.10.x
- CUDA >= 11.8
  - test: 11.8 / 12.x
- torch 2.x
  - test: 2.0.1 / 2.4.1 / 2.8.0
- torchvision: torch version compatibility
- onnx >= 1.15.x
- onnxruntime-gpu
- tensorrt-cu11 10.0.1 (test version)

```shell
$ pip install -r ./requirements.txt 
```

## Modules
- medialoader
- object detectors
  - YOLOv5, YOLOv8, YOLOv10, YOLOv11, YOLOv12
    - pytorch, onnx, trt
- pose estimator
  - HRNet
    - pytorch, onnx, trt
  - RTMPose
    - onnx, trt
  - ViTPose
    - pytorch, onnx, trt

## Reference
- YOLOv5: https://docs.ultralytics.com/models/yolov5/
- YOLOv8: https://docs.ultralytics.com/models/yolov8/
- YOLOv10: https://github.com/THU-MIG/yolov10
  - https://docs.ultralytics.com/models/yolov10/
- YOLOv11: https://docs.ultralytics.com/models/yolo11/
- YOLOv12: https://docs.ultralytics.com/ko/models/yolo12/
- PoseHRNet: https://github.com/HRNet/HRNet-Human-Pose-Estimation
- RTMPose: https://github.com/open-mmlab/mmpose/tree/dev-1.x/projects/rtmpose
  - RTMPose-hand: https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose#hand-2d-21-keypoints
- ViTPose:
  - https://github.com/ViTAE-Transformer/ViTPose
  - https://github.com/JunkyByte/easy_ViTPose
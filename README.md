# pose-estimator
Human Detection + Pose Estimation based Project
- YOLO Detector + HRNet or RTMPose


## Install (Test Evironments)
- Ubuntu 20.04, 22.04 (test version)
- python >= 3.8.x (test version)
- CUDA >= 11.8 (test version)
- torch 2.0.1 (test version)
- torchvision 0.15.2 (test version)
- onnx >= 1.15.x
- onnxruntime-gpu
- tensorrt-cu11 10.0.1 (test version)

```shell
$ pip install -r ./requirements.txt 
```

## Modules
- medialoader
- object detectors
  - YOLOv5
    - pytorch, onnx, trt
  - YOLOv8
    - pytorch, onnx, trt
- pose estimator
  - HRNet
    - pytorch, onnx, trt
  - RTMPose
    - onnx, trt
  - ViTPose
    - pytorch

## Reference
- YOLOv5: https://github.com/ultralytics/yolov5
- YOLOv8(ultraytics): https://github.com/ultralytics/ultralytics
- PoseHRNet: https://github.com/HRNet/HRNet-Human-Pose-Estimation
- RTMPose: https://github.com/open-mmlab/mmpose/tree/dev-1.x/projects/rtmpose
  - RTMPose-hand: https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose#hand-2d-21-keypoints
- ViTPose:
  - https://github.com/ViTAE-Transformer/ViTPose
  - https://github.com/JunkyByte/easy_ViTPose
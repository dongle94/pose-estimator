# pose-estimator
Human Detection + Pose Estimation

## Introduce
We use detection for Human and joint keypoint estimator to predict human joints.
It use cropped image of a human area as an input for task.
Pose estimation task's raw output is specific size of array that proposal distribution of locating of joints.
Post processing change the raw heatmap output to a joint position such as (x, y).

And it can estimate hand joint keypoints using edit configuration options. 
You can use yolov8 detector for hand and rtmpose hand model.
In config, you can change parameter to 'coco-hand' for visualization.

## Install (Test Evironments)
- Ubuntu 20.04
- python 3.8.15
- CUDA 11.8
- torch 2.0.1
- onnx >= 1.15.x
- onnxruntime-gpu
- tensorrt-cu11 10.0.1

```shell
$ pip install -r ./requirements.txt
```

## Optimization
You can use scripts in tools. For Example,
```shell
$ python tools/rtmpose_onnx2trt.py -w ./weights/rtmpose/rtmpose-x_coco_384x288.onnx --verbose
```


## Reference
- YOLOv5: https://github.com/ultralytics/yolov5
- YOLOv8(ultraytics): https://github.com/ultralytics/ultralytics
- PoseHRNet: https://github.com/HRNet/HRNet-Human-Pose-Estimation
- RTMPose: https://github.com/open-mmlab/mmpose/tree/dev-1.x/projects/rtmpose
  - RTMPose-hand: https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose#hand-2d-21-keypoints

# pose-estimator
Human Detection + Pose Estimation

## Introduce
We use detection for human and joint keypoint estimator to predict human joints.
It use cropped image of a human area as an input for task.
Pose estimation task's raw output is specific size of array that proposal distribution of locating of joints.
Post processing change the raw heatmap output to a joint position such as (x, y).

## Install (Test Evironments)
- Ubuntu 20.04
- python 3.8.15

```shell
$ pip install -r ./requirements.txt
```


## Reference
- yolov5: https://github.com/ultralytics/yolov5
- YOLOv8(ultraytics): https://github.com/ultralytics/ultralytics

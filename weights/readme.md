## Model Download


### Objecct Detector 
####  YOLO(Ultralytics)
- YOLOv5: https://docs.ultralytics.com/models/yolov5/
  - pt(torch) download link
    - [yolov5nu](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov5nu.pt), [yolov5su](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov5su.pt), [yolov5mu](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov5mu.pt), [yolov5lu](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov5lu.pt), [yolov5xu](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov5xu.pt)
    - [yolov5n6u](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov5n6u.pt), [yolov5s6u](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov5s6u.pt), [yolov5m6u](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov5m6u.pt), [yolov5l6u](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov5l6u.pt), [yolov5x6u](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov5x6u.pt)

- YOLOv8: https://docs.ultralytics.com/models/yolov8/
  - pt(torch) download link
    - [yolo8n](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt), [yolo8s](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt), [yolo8s](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt), [yolo8s](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l.pt), [yolo8s](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt)

- YOLOv10: https://docs.ultralytics.com/models/yolov10/
  - pt(torch) download link
    - [YOLOv10-N](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10n.pt), [YOLOv10-S](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10s.pt)
    - [YOLOv10-M](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10m.pt), [YOLOv10-B](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10b.pt)
    - [YOLOv10-L](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10l.pt), [YOLOv10-X](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10x.pt)

- YOLOv11: https://docs.ultralytics.com/models/yolo11/
  - pt(torch) download link
    - [YOLO11n](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt), [YOLO11s](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt)
    - [YOLO11m](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt), [YOLO11l](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt), [YOLO11x](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt)


### PoseDetector
#### [HRNet](https://github.com/HRNet/deep-high-resolution-net.pytorch) (Pytorch)  
First, You need to convert model from python class architecture and weight to merged format.
You can download each hrnet model configuration file and weight file.
- Weights
  - COCO & MPII
    - [Google Drive](https://drive.google.com/drive/folders/14p2l1u19bLOm5p6esKyolYJyfE1_inLv)
    - [OneDrive](https://onedrive.live.com/?authkey=%21AEwfaSueYurmSRA&id=56B9F9C97F261712%2111775&cid=56B9F9C97F261712)

#### [ViTPose](https://github.com/ViTAE-Transformer/ViTPose/tree/main/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco?tab=readme-ov-file#results-from-this-repo-on-ms-coco-val-set-single-task-training) (Pytorch)
With classic decoder
- ViTPose-s
  - [HuggingFace](https://huggingface.co/JunkyByte/easy_ViTPose/blob/main/torch/coco/vitpose-s-coco.pth)
- ViTPose-b
  - [OneDrive](https://onedrive.live.com/?authkey=%21ACOnX82tXdVFKYo&id=E534267B85818129%21163&cid=E534267B85818129&parId=root&parQt=sharedby&o=OneUp)
- ViTPose-l
  - [OneDrive](https://onedrive.live.com/?authkey=%21AH2T%2DS6S0%2D2I%5FgU&id=E534267B85818129%21167&cid=E534267B85818129&parId=root&parQt=sharedby&o=OneUp)
- ViTPose-h
  - [OneDrive](https://onedrive.live.com/?authkey=%21AEswj6SSa818X%2DE&id=E534267B85818129%21168&cid=E534267B85818129&parId=root&parQt=sharedby&o=OneUp)
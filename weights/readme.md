## Model Download


### HumanDetector 
- Yolov5 r7.0
  - architecture
    - light(mobile, jetson, ...): `n`, `n6`, `s`, `s6`
    - pc, server: `m`, `m6`, `l`, `l6`, `x`, `x6`
  - link: https://github.com/ultralytics/yolov5/releases/tag/v7.0
- YOLOv8
  - link: https://github.com/ultralytics/ultralytics


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
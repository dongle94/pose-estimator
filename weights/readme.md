## Model Download


### HumanDetector 
- Yolov5 r7.0
  - architecture
    - light(mobile, jetson, ...): `n`, `n6`, `s`, `s6`
    - pc, server: `m`, `m6`, `l`, `l6`, `x`, `x6`
  - link: https://github.com/ultralytics/yolov5/releases/tag/v7.0



### PoseDetector
#### [HRNet](https://github.com/HRNet/deep-high-resolution-net.pytorch) (Pytorch)  
First, You need to convert model from python class architecture and weight to merged format.
You can download each hrnet model configuration file and weight file.
- Configuration
  - Downloads: [Official HRNet Configuration](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/tree/master/experiments/)
- Weights
  - COCO & MPII
    - [Google Drive](https://drive.google.com/drive/folders/14p2l1u19bLOm5p6esKyolYJyfE1_inLv)
    - [OneDrive](https://onedrive.live.com/?authkey=%21AEwfaSueYurmSRA&id=56B9F9C97F261712%2111775&cid=56B9F9C97F261712)

Second, You can use script to create model merged architecture and weights.
If, you download configuration file and wiehgt like the following structure,
```commandline
.(pose-estimator)
├── ...
├── tools
│   └── hrnet_merge_model.py
└── weights
    ├── pose_hrnet_w32_256x192.pth
    ├── readme.md
    └── w32_256x192_adam_lr1e-3.yaml
```
You can run as follows.

```shell
# path is fixed at repository root path you run script anywhere.

$ python hrnet_merge_model.py \
  -c ./weights/w32_256x192_adam_lr1e-3.yaml \
  -w ./weights/pose_hrnet_w32_256x192.pth \
  -s ./weights/hrnet_merge_w32_256x192.pth
```


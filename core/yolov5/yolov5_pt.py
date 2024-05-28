import sys
import time
import copy
import numpy as np
from typing import Union
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.yolov5 import YOLOV5
from core.yolov5.yolov5_utils.torch_utils import select_device
from core.yolov5.yolov5_utils.general import check_img_size, non_max_suppression, scale_boxes
from core.yolov5.yolov5_utils.augmentations import letterbox
from core.yolov5.models.experimental import attempt_load


class Yolov5Torch(YOLOV5):
    def __init__(self, weight: str, device: str = "cpu", img_size: int = 640, fp16: bool = False, auto: bool = False,
                 fuse: bool = True, gpu_num: int = 0, conf_thres=0.25, iou_thres=0.45, agnostic=False, max_det=100,
                 classes: Union[list, None] = None):
        super().__init__()
        self.device = select_device(device=device, gpu_num=gpu_num)
        self.cuda = torch.cuda.is_available() and device != "cpu"
        if fp16 is True and self.device.type != "cpu":
            self.fp16 = True
        else:
            self.fp16 = False
            print("Yolov5 pytorch model not support cpu version's fp16. It apply fp32.")
        model = attempt_load(weight, device=self.device, inplace=True, fuse=fuse)
        model.half() if self.fp16 else model.float()
        self.model = model
        self.stride = max(int(model.stride.max()), 32)
        self.img_size = check_img_size(img_size, s=self.stride)
        self.names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        self.auto = auto

        # parameter for postprocessing
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.agnostic = agnostic
        self.max_det = max_det

    def warmup(self, img_size=(1, 3, 640, 640)):
        im = torch.empty(*img_size, dtype=torch.half if self.fp16 else torch.float, device=self.device)
        t = self.get_time()
        self.infer(im)  # warmup
        print(f"-- Yolov5 Detector warmup: {self.get_time()-t:.6f} sec --")

    def preprocess(self, img: np.ndarray):
        im = letterbox(img, new_shape=self.img_size, auto=self.auto, stride=self.stride)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous

        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = torch.unsqueeze(im, dim=0)  # expand for batch dim
        return im, img

    def infer(self, im):
        y = self.model(im)
        return y

    def postprocess(self, pred, im_shape, im0_shape):
        pred = non_max_suppression(pred,
                                   conf_thres=self.conf_thres,
                                   iou_thres=self.iou_thres,
                                   classes=self.classes,
                                   agnostic=self.agnostic,
                                   max_det=self.max_det)[0]
        det = scale_boxes(im_shape[2:], copy.deepcopy(pred[:, :4]), im0_shape).round()
        det = torch.cat([det, pred[:, 4:]], dim=1)

        return pred, det

    def get_time(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        return time.time()


if __name__ == "__main__":
    import cv2
    from utils.config import set_config, get_config

    set_config('./configs/config.yaml')
    cfg = get_config()

    # device: cpu, cuda, mps
    yolov5 = Yolov5Torch(cfg.det_model_path, device=cfg.device, img_size=cfg.yolov5_img_size, fp16=cfg.det_half,
                         auto=False, gpu_num=cfg.gpu_num, conf_thres=cfg.det_conf_thres, iou_thres=cfg.yolov5_nms_iou,
                         agnostic=cfg.yolov5_agnostic_nms, max_det=cfg.yolov5_max_det, classes=cfg.det_obj_classes)
    yolov5.warmup()

    _im = cv2.imread('./data/images/sample.jpg')
    t0 = yolov5.get_time()
    _im, _im0 = yolov5.preprocess(_im)
    t1 = yolov5.get_time()
    _y = yolov5.infer(_im)
    t2 = yolov5.get_time()
    _pred, _det = yolov5.postprocess(_y, _im.size(), _im0.shape)
    t3 = yolov5.get_time()

    _det = _det.cpu().numpy()
    for d in _det:
        cv2.rectangle(
            _im0,
            (int(d[0]), int(d[1])),
            (int(d[2]), int(d[3])),
            (0, 0, 255),
            1,
            cv2.LINE_AA
        )
    cv2.imshow('result', _im0)
    cv2.waitKey(0)
    print(f"{_im0.shape} size image - pre: {t1-t0:.6f} / infer: {t2-t1:.6f} / post: {t3-t2:.6f}")

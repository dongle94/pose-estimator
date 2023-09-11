import sys
import copy
import os
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
os.chdir(ROOT)

from core.obj_detectors.models.yolo import check_img_size, letterbox, non_max_suppression, scale_boxes
from core.obj_detectors.models.torch_utils import select_device


class YoloDetector(nn.Module):
    def __init__(self, weight='yolov5s.pt', device: int or str = "cpu", img_size=640, fp16=False, auto=True, fuse=True,
                 classes=None):
        super().__init__()

        device = "cpu" if device == "" else device
        self.device = select_device(device)
        self.cuda = torch.cuda.is_available() and device
        self.fp16 = True if fp16 and self.device.type != "cpu" else False
        _model = attempt_load(weight, device=device, inplace=True, fuse=fuse)
        _model.half() if self.fp16 else _model.float()
        self.model = _model
        self.stride = max(int(_model.stride.max()), 32)
        self.img_size = check_img_size(img_size, s=self.stride)
        self.names = _model.module.names if hasattr(_model, 'module') else _model.names  # get class names
        self.auto = auto
        self.classes = classes

    def warmup(self, imgsz=(1, 3, 640, 640)):
        _im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
        self.infer(_im)  # warmup
        print("-- Yolov5 Detector warmup --")

    def preprocess(self, _img):
        _im = letterbox(_img, self.img_size, stride=self.stride, auto=self.auto)[0]  # padded resize
        _im = _im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        _im = np.ascontiguousarray(_im)  # contiguous

        _im = torch.from_numpy(_im).to(self.device)
        _im = _im.half() if self.fp16 else _im.float()  # uint8 to fp16/32
        _im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(_im.shape) == 3:
            _im = torch.unsqueeze(_im, dim=0)  # expand for batch dim

        return _im, _img

    def infer(self, _im):
        if self.fp16 and _im.dtype != torch.float16:
            _im = _im.half()  # to FP16
        y = self.model(_im)

        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def postprocess(self, pred, im_shape, im0_shape, conf_thres=0.25, nms_iou=0.45, agnostic_nms=False, max_det=100):
        pred = non_max_suppression(pred, classes=self.classes, conf_thres=conf_thres, iou_thres=nms_iou,
                                   agnostic=agnostic_nms, max_det=max_det)[0]
        det = scale_boxes(im_shape[2:], copy.deepcopy(pred[:, :4]), im0_shape).round()
        det = torch.cat([det, pred[:, 4:]], dim=1)

        return pred.cpu().numpy(), det.cpu().numpy()

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x


def attempt_load(weight, device=None, inplace=True, fuse=True):

    ckpt = torch.load(str(weight), map_location='cpu')
    ckpt = (ckpt.get('ema') or ckpt['model']).to(device).float()

    # Model compatibility updates
    if not hasattr(ckpt, 'stride'):
        ckpt.stride = torch.tensor([32.])
    if hasattr(ckpt, 'names') and isinstance(ckpt.names, (list, tuple)):
        ckpt.names = dict(enumerate(ckpt.names))  # convert to dict
    ckpt = ckpt.fuse().eval() if fuse and hasattr(ckpt, 'fuse') else ckpt.eval()

    from models.yolo import Detect, Model
    # Module compatibility updates
    for m in ckpt.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
            m.inplace = inplace  # torch 1.7.0 compatibility
            if t is Detect and not isinstance(m.anchor_grid, list):
                delattr(m, 'anchor_grid')
                setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
        elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # return model
    return ckpt


if __name__ == "__main__":
    model = YoloDetector(weight='./weights/yolov5n.pt', device=0, img_size=640)
    model.warmup()

    import cv2
    img = cv2.imread('./data/images/army.jpg')
    im, im0 = model.preprocess(img)

    pred = model.infer(im)
    pred, det = model.postprocess(pred, im.shape, im0.shape)
    print(det)
    for d in det:
        x1, y1, x2, y2 = map(int, d[:4])
        cv2.rectangle(im0, (x1, y1), (x2, y2), (128, 128, 128), thickness=2, lineType=cv2.LINE_AA)

    cv2.imshow('_', im0)
    cv2.waitKey(0)

import copy
import os
import numpy as np
import torch
import torch.nn as nn

from models.yolo import check_img_size, letterbox, non_max_suppression, scale_boxes

class YoloDetector(nn.Module):
    def __init__(self, weight='yolov5s.pt', device=torch.device('cpu'), img_size=640, fp16=False, auto=True, fuse=True):
        super().__init__()

        self.device = self.select_device(device)
        self.cuda = torch.cuda.is_available() and device
        self.fp16 = fp16
        model = attempt_load(weight, device=device, inplace=True, fuse=fuse)
        model.half() if fp16 else model.float()
        self.model = model
        self.stride = max(int(model.stride.max()), 32)
        self.img_size = check_img_size(img_size, s=self.stride)
        self.names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        self.auto = auto

    def warmup(self, imgsz=(1, 3, 640, 640)):
        im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
        self.forward(im)  # warmup
        print("-- Yolov5 Detector warmup --")

    def preprocess(self, img):
        im = letterbox(img, self.img_size, stride=self.stride, auto=self.auto)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous

        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        return im, img

    def forward(self, im):
        b, ch, h, w = im.shape

        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16
        y = self.model(im)

        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def postprocess(self, pred, im_shape, im0_shape, max_det=100):
        pred = non_max_suppression(pred, classes=[0], max_det=max_det)[0]
        det = scale_boxes(im_shape[2:], copy.deepcopy(pred[:, :4]), im0_shape).round()
        det = torch.cat([det, pred[:, 4:]], dim=1)

        return pred, det

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    def select_device(self, device=''):
        device = str(device).strip().lower().replace('cuda', '').replace('none', '')
        cpu = device == 'cpu'
        if cpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        elif device:  # non-cpu device requested
            os.environ[
                'CUDA_VISIBLE_DEVICES'] = device  # set environment variable - must be before assert is_available()

        if not cpu and torch.cuda.is_available():
            device = device if device else '0'
            arg = 'cuda:0'
        else:
            arg = 'cpu'

        return torch.device(arg)

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
    model = YoloDetector(weight='../weights/yolov5n.pt', device='cpu', img_size=640)
    model.warmup()

    import cv2
    img = cv2.imread('../army.jpg')
    im, im0 = model.preprocess(img)

    pred = model.forward(im)
    pred, det = model.postprocess(pred, im.shape, im0.shape)
    print(det)
    for d in det:
        x1, y1, x2, y2 = map(int, d[:4])
        cv2.rectangle(im0, (x1, y1), (x2, y2), (128, 128, 128), thickness=2, lineType=cv2.LINE_AA)

    cv2.imshow('_', im0)
    cv2.waitKey(0)
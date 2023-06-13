import math
import numpy as np
import torch
import torch.nn as nn


class YoloDetector(nn.Module):
    def __init__(self, weight='yolov5s.pt', device=torch.device('cpu'), fp16=False, fuse=True):
        super().__init__()

        self.cuda = torch.cuda.is_available() and device
        self.fp16 = fp16
        model = attempt_load(weight, device=device, inplace=True, fuse=fuse)
        self.stride = max(int(model.stride.max()), 32)

        self.names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        model.half() if fp16 else model.float()
        self.model = model

    def forward(self, im, ):
        b, ch, h, w = im.shape

        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16
        y = self.model(im)

        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=(1, 3, 640, 640)):
        im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
        self.forward(im)  # warmup
        print("-- Yolov5 Detector warmup --")


def attempt_load(weight, device=None, inplace=True, fuse=True):

    # model = Ensemble()
    if device not in ["cpu", None]:
        device = f"cuda:{device}"
    ckpt = torch.load(str(weight), map_location='cpu')
    ckpt = (ckpt.get('ema') or ckpt['model']).to(device).float()

    # Model compatibility updates
    if not hasattr(ckpt, 'stride'):
        ckpt.stride = torch.tensor([32.])
    if hasattr(ckpt, 'names') and isinstance(ckpt.names, (list, tuple)):
        ckpt.names = dict(enumerate(ckpt.names))  # convert to dict
    ckpt = ckpt.fuse().eval() if fuse and hasattr(ckpt, 'fuse') else ckpt.eval()

    # return model
    return ckpt

if __name__ == "__main__":
    model = YoloDetector(weight='../weights/yolov5n.pt')

    # print(dir(model), type(model))
import sys
import time
import numpy as np
import torch
from typing import Union
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.yolo.util.torch_utils import select_device
from core.yolo.nn.tasks import attempt_load_weights
from core.yolo.util.checks import check_imgsz
from core.yolo.data.augment import LetterBox
from core.yolo.util.ops import scale_boxes


class Yolov10Torch(object):
    def __init__(self, weight: str, device: str = 'cpu', fp16: bool = False, fuse: bool = True, auto: bool = False,
                 img_size: int = 640, conf_thres=0.25, classes: Union[list, None] = None, **kwargs):
        super(Yolov10Torch, self).__init__()
        self.device = select_device(device=device)
        self.cuda = torch.cuda.is_available() and device != "cpu"
        if fp16 is True and self.device.type != "cpu":
            self.fp16 = True
        else:
            self.fp16 = False

        model = attempt_load_weights(weight, device=self.device, inplace=True, fuse=fuse)
        model.half() if self.fp16 else model.float()
        for p in model.parameters():
            p.requires_grad = False
        self.model = model
        self.model.eval()
        self.stride = max(int(model.stride.max()), 32)
        self.names = model.module.names if hasattr(model, "module") else model.names
        self.img_size = check_imgsz(img_size, stride=self.model.stride, min_dim=2)
        self.auto = auto
        self.letter_box = LetterBox(self.img_size, auto=self.auto, stride=self.stride)

        # parameter for postprocessing
        self.conf_thres = conf_thres
        self.classes = classes

    def warmup(self, img_size=(1, 3, 640, 640)):
        im = torch.empty(*img_size, dtype=torch.half if self.fp16 else torch.float, device=self.device)
        t = self.get_time()
        self.infer(im)
        print(f"-- YOLOv10 Detector warmup: {time.time() - t:.6f} sec --")

    def preprocess(self, img):
        im = self.letter_box(image=img)
        im = im[..., ::-1].transpose((2, 0, 1))
        im = np.ascontiguousarray(im)  # contiguous

        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.fp16 else im.float()
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = torch.unsqueeze(im, dim=0)  # expand for batch dim
        return im, img

    def infer(self, img):
        y = self.model(img)
        return y

    def postprocess(self, pred, im_shape, im0_shape):
        if isinstance(pred, dict):
            pred = pred["one2one"]

        if isinstance(pred, (list, tuple)):
            pred = pred[0]  # 1, 300, 6

        if pred.shape[-1] == 6:
            pass
        else:
            raise

        mask = pred[..., 4] > self.conf_thres
        if self.classes is not None:
            mask = mask & (pred[..., 5:6] == torch.tensor(self.classes, device=pred.device).unsqueeze(0)).any(2)

        pred = [p[mask[idx]] for idx, p in enumerate(pred)][0]
        pred[:, :4] = scale_boxes(im_shape[2:], pred[:, :4], im0_shape)

        return None, pred

    def get_time(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        return time.time()


if __name__ == "__main__":
    import cv2
    from utils.config import set_config, get_config

    set_config('./configs/config.yaml')
    cfg = get_config()

    yolov10 = Yolov10Torch(cfg.det_model_path, device=cfg.device, fp16=cfg.det_half, img_size=cfg.yolo_img_size,
                           conf_thres=cfg.det_conf_thres, classes=cfg.det_obj_classes)
    yolov10.warmup()

    _im = cv2.imread('./data/images/sample.jpg')
    t0 = yolov10.get_time()
    _im, _im0 = yolov10.preprocess(_im)
    t1 = yolov10.get_time()
    _y = yolov10.infer(_im)
    t2 = yolov10.get_time()
    _, _pred = yolov10.postprocess(_y, _im.size(), _im0.shape)
    t3 = yolov10.get_time()

    _det = _pred.cpu().numpy()
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
    print(f"{_im0.shape} size image - pre: {t1 - t0:.6f} / infer: {t2 - t1:.6f} / post: {t3 - t2:.6f}")

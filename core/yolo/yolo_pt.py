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

from core.yolo import YOLO
from core.yolo.util.torch_utils import select_device
from core.yolo.util.checks import check_imgsz
from core.yolo.util.ops import non_max_suppression, scale_boxes
from core.yolo.nn.tasks import attempt_load_weights
from core.yolo.data.augment import LetterBox
from utils.logger import get_logger


class YoloTorch(YOLO):
    def __init__(self, weight: str, device: str = 'cpu', gpu_num: int = 0, img_size: int = 640, fp16: bool = False,
                 auto: bool = False, fuse: bool = True, conf_thres=0.25, iou_thres=0.45, agnostic=False, max_det=100,
                 classes: Union[list, None] = None, **kwargs):
        super(YoloTorch, self).__init__()

        self.logger = get_logger()
        self.device = select_device(device=device, gpu_num=gpu_num, logger=get_logger())
        self.cuda = torch.cuda.is_available() and device != "cpu"
        if fp16 is True and self.device.type != "cpu":
            self.fp16 = True
        else:
            self.fp16 = False
            self.logger.info(f"{kwargs['model_type']} pytorch will not use fp16. It will apply fp32 precision.")

        model = attempt_load_weights(weight, device=self.device, inplace=True, fuse=fuse)
        self.model = model.half() if self.fp16 else model.float()
        self.model.eval()
        self.stride = max(int(model.stride.max()), 32)
        self.img_size = check_imgsz(img_size, stride=self.stride, min_dim=2)
        self.names = model.module.names if hasattr(model, 'module') else model.names
        self.auto = auto
        self.letter_box = LetterBox(self.img_size, auto=self.auto, stride=self.stride)

        # parameter for postprocessing
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.agnostic = agnostic
        self.max_det = max_det
        self.kwargs = kwargs

    def warmup(self, img_size=None):
        if img_size is None:
            img_size = (1, 3, self.img_size[0], self.img_size[1])
        im = torch.empty(*img_size, dtype=torch.half if self.fp16 else torch.float, device=self.device)

        t = self.get_time()
        for _ in range(2):
            self.infer(im)  # warmup
        self.logger.info(f"-- {self.kwargs['model_type']} Pytorch Detector warmup: {time.time()-t:.6f} sec --")

    def preprocess(self, img):
        im = self.letter_box(image=img)
        im = im[..., ::-1].transpose((2, 0, 1))      # BGR to RGB, HWC to CHW, (3, h, w)
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im).to(self.device)

        im = im.half() if self.fp16 else im.float()     # uint8 to fp16/32
        im /= 255       # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = torch.unsqueeze(im, dim=0)  # expand for batch dim
        return im, img

    def infer(self, img):
        y = self.model(img)
        return y

    def postprocess(self, pred, im_shape, im0_shape):
        pred = non_max_suppression(
            pred,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            classes=self.classes,
            agnostic=self.agnostic,
            max_det=self.max_det
        )[0]
        det = scale_boxes(im_shape[2:], copy.deepcopy(pred[:, :4]), im0_shape)
        det = torch.cat([det, pred[:, 4:]], dim=1)
        det = det.cpu().detach().numpy()

        return det

    def get_time(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        return time.time()


if __name__ == "__main__":
    import cv2
    from utils.logger import init_logger
    from utils.config import set_config, get_config

    set_config('./configs/config.yaml')
    cfg = get_config()

    init_logger(cfg)

    _detector = YoloTorch(
        cfg.det_model_path,
        device=cfg.device,
        gpu_num=cfg.gpu_num,
        img_size=cfg.yolo_img_size,
        fp16=cfg.det_half,
        conf_thres=cfg.det_conf_thres,
        iou_thres=cfg.yolo_nms_iou,
        agnostic=cfg.yolo_agnostic_nms,
        max_det=cfg.yolo_max_det,
        classes=cfg.det_obj_classes,
        model_type=cfg.det_model_type
    )
    _detector.warmup()

    _im = cv2.imread('./data/images/sample.jpg')

    t0 = _detector.get_time()
    _im, _im0 = _detector.preprocess(_im)
    t1 = _detector.get_time()
    _y = _detector.infer(_im)
    t2 = _detector.get_time()
    _det = _detector.postprocess(_y, _im.size(), _im0.shape)
    t3 = _detector.get_time()

    for d in _det:
        x1, y1, x2, y2 = map(int, d[:4])
        cls = int(d[5])
        cv2.rectangle(_im0, (x1, y1), (x2, y2), (96, 96, 216), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(_im0, str(_detector.names[cls]), (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (96, 96, 96), thickness=1, lineType=cv2.LINE_AA)

    cv2.imshow('result', _im0)
    cv2.waitKey(0)
    get_logger().info(f"{_im0.shape} size image - pre: {t1 - t0:.6f} / infer: {t2 - t1:.6f} / post: {t3 - t2:.6f}")

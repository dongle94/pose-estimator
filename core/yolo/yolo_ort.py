import sys
import time
import copy
import numpy as np
from typing import Union
from pathlib import Path

import torch
import onnxruntime as ort

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.yolo import YOLO
from core.yolo.util.checks import check_imgsz
from core.yolo.data.augment import LetterBox
from core.yolo.util.ops import scale_boxes, non_max_suppression_np, non_max_suppression
from utils.logger import get_logger


class YoloORT(YOLO):
    def __init__(self, weight: str, device: str = 'cpu', gpu_num: int = 0, img_size: int = 640, fp16: bool = False,
                 auto: bool = False, conf_thres=0.25, iou_thres=0.45, agnostic=False, max_det=100,
                 classes: Union[list, None] = None, **kwargs):
        super(YoloORT, self).__init__()

        self.logger = get_logger()
        self.device = device
        self.gpu_num = gpu_num
        self.cuda = ort.get_device() == 'GPU' and device == 'cuda'
        self.fp16 = True if fp16 is True else False

        providers = ['CPUExecutionProvider']
        if self.cuda is True:
            cuda_provider = (
                "CUDAExecutionProvider", {
                    "device_id": gpu_num,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'cudnn_conv_algo_search': 'HEURISTIC',
                }
            )
            providers.insert(0, cuda_provider)
        self.sess = ort.InferenceSession(weight, providers=providers)
        self.io_binding = self.sess.io_binding()
        if device == 'cuda':
            self.io_binding.bind_output('output0')

        self.input_name = self.sess.get_inputs()[0].name
        self.output_names = [o.name for o in self.sess.get_outputs()]
        self.output_shapes = [output.shape for output in self.sess.get_outputs()]
        self.output_types = [output.type for output in self.sess.get_outputs()]

        meta = self.sess.get_modelmeta().custom_metadata_map  # metadata
        if 'stride' in meta:
            self.stride, self.names = int(meta['stride']), eval(meta['names'])

        self.stride = max(int(self.stride), 32)
        self.img_size = check_imgsz(img_size, stride=self.stride, min_dim=2)
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
        im = np.zeros(img_size, dtype=np.float16 if self.fp16 else np.float32)
        if self.device == 'cuda':
            im_ortval = ort.OrtValue.ortvalue_from_numpy(im, 'cuda', self.gpu_num)
            self.io_binding.bind_input(
                name='images', device_type=im_ortval.device_name(), device_id=self.gpu_num, element_type=im.dtype,
                shape=im_ortval.shape(), buffer_ptr=im_ortval.data_ptr()
            )

        t = self.get_time()
        self.infer(im)
        self.logger.info(f"-- {self.kwargs['model_type']} Onnx Detector warmup: {self.get_time() - t:.6f} sec --")

    def preprocess(self, img):
        im = self.letter_box(image=img)
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im).astype(np.float32)  # contiguous

        im /= 255.0
        im = im.astype(np.float16) if self.fp16 else im.astype(np.float32)
        if len(im.shape) == 3:
            im = np.expand_dims(im, axis=0)  # expand for batch dim

        if self.device == 'cuda':
            im_ortval = ort.OrtValue.ortvalue_from_numpy(im, self.device, self.gpu_num)
            self.io_binding.bind_input(
                name='images', device_type=im_ortval.device_name(), device_id=self.gpu_num, element_type=im.dtype,
                shape=im_ortval.shape(), buffer_ptr=im_ortval.data_ptr())
        return im, img

    def infer(self, img):
        if self.device == 'cuda':
            self.sess.run_with_iobinding(self.io_binding)
            ret = None  # output i/o gpu->cpu will execute in postprocess
        else:
            ret = self.sess.run(self.output_names, {self.input_name: img})
        return ret

    def postprocess(self, pred, im_shape, im0_shape):
        if self.device == 'cuda':
            pred = self.io_binding.copy_outputs_to_cpu()[0]
            pred = torch.from_numpy(pred)
            pred = non_max_suppression(
                prediction=pred,
                iou_thres=self.iou_thres,
                conf_thres=self.conf_thres,
                classes=self.classes,
                agnostic=self.agnostic,
                max_det=self.max_det
            )[0]
            det = scale_boxes(im_shape[2:], copy.deepcopy(pred[:, :4]), im0_shape).round()
            det = torch.cat([det, pred[:, 4:]], dim=1)
            det = det.cpu().detach().numpy()
        else:
            pred = non_max_suppression_np(
                prediction=pred,
                iou_thres=self.iou_thres,
                conf_thres=self.conf_thres,
                classes=self.classes,
                agnostic=self.agnostic,
                max_det=self.max_det
            )[0]
            det = scale_boxes(im_shape[2:], copy.deepcopy(pred[:, :4]), im0_shape).round()
            det = np.concatenate([det, pred[:, 4:]], axis=1)

        return det

    def get_time(self):
        return time.time()


if __name__ == "__main__":
    import cv2
    from utils.logger import init_logger
    from utils.config import set_config, get_config

    set_config('./configs/config.yaml')
    cfg = get_config()

    init_logger(cfg)

    _detector = YoloORT(
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
    _det = _detector.postprocess(_y, _im.shape, _im0.shape)
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

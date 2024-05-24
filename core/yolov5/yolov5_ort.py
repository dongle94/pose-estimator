import os
import sys
import time
import copy
import numpy as np
from typing import Union
from pathlib import Path

import onnxruntime as ort
from onnxconverter_common import float16

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.yolov5 import YOLOV5
from core.yolov5.yolov5_utils.general import non_max_suppression, non_max_suppression_np, scale_boxes
from core.yolov5.yolov5_utils.augmentations import letterbox


class Yolov5ORT(YOLOV5):
    def __init__(self, weight: str, device: str = "cpu", img_size: int = 640, fp16: bool = False, auto: bool = False,
                 gpu_num: int = 0, conf_thres=0.25, iou_thres=0.45, agnostic=False, max_det=100,
                 classes: Union[list, None] = None):
        super().__init__()
        self.img_size = img_size
        self.auto = auto
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

        if self.fp16:
            weight_dir = os.path.dirname(weight)
            weight_name, weight_ext = os.path.splitext(os.path.basename(weight))
            half_model = os.path.join(weight_dir, f"{weight_name}-fp16{weight_ext}")
            if not os.path.exists(half_model):
                import onnx
                model = onnx.load(weight)
                model_fp16 = float16.convert_float_to_float16(
                    model=model,
                    min_positive_val=float('-inf'),
                    max_finite_val=float('inf'),
                    keep_io_types=False
                )
                onnx.save(model_fp16, half_model)
                print("Onnx model convert fp16 model.")
            weight = half_model
            print("Onnx model get fp16.")
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

        # parameter for postprocessing
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.agnostic = agnostic
        self.max_det = max_det

    def warmup(self, img_size=(1, 3, 640, 640)):
        im = np.zeros(img_size, dtype=np.float16 if self.fp16 else np.float32)
        if self.device == 'cuda':
            im_ortval = ort.OrtValue.ortvalue_from_numpy(im, 'cuda', self.gpu_num)
            element_type = np.float16 if self.fp16 else np.float32
            self.io_binding.bind_input(
                name='images', device_type=im_ortval.device_name(), device_id=self.gpu_num, element_type=element_type,
                shape=im_ortval.shape(), buffer_ptr=im_ortval.data_ptr())

        t = time.time()
        self.infer(im)
        print(f"-- Yolov5 Onnx Detector warmup: {time.time()-t:.6f} sec --")

    def preprocess(self, img: np.ndarray):
        im = letterbox(img, new_shape=self.img_size, auto=self.auto, stride=self.stride)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im).astype(np.float32)  # contiguous

        im /= 255.0
        im = im.astype(np.float16) if self.fp16 else im.astype(np.float32)
        if len(im.shape) == 3:
            im = np.expand_dims(im, axis=0)  # expand for batch dim

        if self.device == 'cuda':
            im_ortval = ort.OrtValue.ortvalue_from_numpy(im, self.device, self.gpu_num)
            element_type = np.float16 if self.fp16 else np.float32
            self.io_binding.bind_input(
                name='images', device_type=im_ortval.device_name(), device_id=self.gpu_num, element_type=element_type,
                shape=im_ortval.shape(), buffer_ptr=im_ortval.data_ptr())
        return im, img

    def infer(self, img):
        if self.device == 'cuda':
            self.sess.run_with_iobinding(self.io_binding)
            ret = None      # output i/o gpu->cpu will execute in postprocess
        else:
            ret = self.sess.run(self.output_names, {self.input_name: img})
        return ret

    def postprocess(self, pred, im_shape, im0_shape):
        if self.device == 'cuda':
            pred = self.io_binding.copy_outputs_to_cpu()[0]
        pred = non_max_suppression_np(prediction=pred,
                                      iou_thres=self.iou_thres,
                                      conf_thres=self.conf_thres,
                                      classes=self.classes,
                                      agnostic=self.agnostic,
                                      max_det=self.max_det)[0]
        det = scale_boxes(im_shape[2:], copy.deepcopy(pred[:, :4]), im0_shape).round()
        det = np.concatenate([det, pred[:, 4:]], axis=1)

        return pred, det


if __name__ == "__main__":
    import cv2
    from utils.config import set_config, get_config

    set_config('./configs/config.yaml')
    cfg = get_config()

    yolov5 = Yolov5ORT(
        cfg.det_model_path, device=cfg.device, img_size=cfg.yolov5_img_size, fp16=cfg.det_half, auto=False,
        gpu_num=cfg.gpu_num, conf_thres=cfg.det_conf_thres, iou_thres=cfg.yolov5_nms_iou,
        agnostic=cfg.yolov5_agnostic_nms, max_det=cfg.yolov5_max_det, classes=cfg.det_obj_classes
    )
    yolov5.warmup(imgsz=(1, 3, cfg.yolov5_img_size, cfg.yolov5_img_size))

    _im = cv2.imread('./data/images/sample.jpg')
    t0 = yolov5.get_time()
    _im, _im0 = yolov5.preprocess(_im)
    t1 = yolov5.get_time()
    _y = yolov5.infer(_im)
    t2 = yolov5.get_time()
    _pred, _det = yolov5.postprocess(_y, _im.shape, _im0.shape)
    t3 = yolov5.get_time()

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

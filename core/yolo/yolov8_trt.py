import os
import sys
import time
import copy
import numpy as np
from typing import Union
from pathlib import Path

import torch
from cuda import cudart
import tensorrt as trt

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.yolo.data.augment import LetterBox
from core.yolo.util.ops import scale_boxes, non_max_suppression_np, non_max_suppression


class Yolov8TRT(object):
    def __init__(self, weight: str, device: str = "cpu", img_size: int = 640, fp16: bool = False, auto: bool = False,
                 gpu_num: int = 0, conf_thres=0.25, iou_thres=0.45, agnostic=False, max_det=100,
                 classes: Union[list, None] = None):
        super(Yolov8TRT, self).__init__()
        self.device = device
        self.img_size = img_size
        self.auto = auto
        self.gpu_num = gpu_num
        self.fp16 = True if fp16 is True else False

        self.trt_logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")

        with open(weight, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = self.engine.get_tensor_dtype(name)
            shape = self.engine.get_tensor_shape(name)
            is_input = False
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                is_input = True
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = cudart.cudaMalloc(size)[1]
            binding = {
                'index': i,
                'name': name,
                'dtype': np.dtype(trt.nptype(dtype)),
                'shape': list(shape),
                'allocation': allocation,
                'size': size
            }
            self.allocations.append(allocation)
            if is_input:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)
        self.names = {i: f'class{i}' for i in range(999)}
        self.letter_box = LetterBox(self.img_size, auto=self.auto)

        # prams for postprocessing
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.agnostic = agnostic
        self.max_det = max_det

    def warmup(self, img_size=(1, 3, 640, 640)):
        im = np.zeros(img_size, dtype=np.float16 if self.fp16 else np.float32)  # input
        t = self.get_time()
        self.infer(im)  # warmup
        print(f"-- Yolov8 TRT Detector warmup: {self.get_time() - t:.6f} sec --")

    def preprocess(self, img):
        im = self.letter_box(image=img)
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im).astype(np.float32)  # contiguous

        im /= 255.0
        im = im.astype(np.float16) if self.fp16 else im.astype(np.float32)
        if len(im.shape) == 3:
            im = np.expand_dims(im, axis=0)

        return im, img

    def infer(self, img):
        outputs = []
        for shape, dtype in self.output_spec():
            outputs.append(np.zeros(shape, dtype))

        device_ptr = self.inputs[0]['allocation']
        host_arr = img
        nbytes = host_arr.size * host_arr.itemsize
        cudart.cudaMemcpy(device_ptr, host_arr.data, nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

        self.context.execute_v2(self.allocations)
        for o in range(len(outputs)):
            host_arr = outputs[o]
            device_ptr = self.outputs[o]['allocation']
            nbytes = host_arr.size * host_arr.itemsize
            cudart.cudaMemcpy(host_arr, device_ptr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

        return outputs[0]

    def postprocess(self, pred, im_shape, im0_shape):
        if self.fp16:
            pred = torch.from_numpy(pred)
            pred = non_max_suppression(prediction=pred,
                                       iou_thres=self.iou_thres,
                                       conf_thres=self.conf_thres,
                                       classes=self.classes,
                                       agnostic=self.agnostic,
                                       max_det=self.max_det)[0]
            det = scale_boxes(im_shape[2:], copy.deepcopy(pred[:, :4]), im0_shape).round()
            det = torch.cat([det, pred[:, 4:]], dim=1)
            det = det.cpu().numpy()
        else:
            pred = non_max_suppression_np(prediction=pred,
                                          iou_thres=self.iou_thres,
                                          conf_thres=self.conf_thres,
                                          classes=self.classes,
                                          agnostic=self.agnostic,
                                          max_det=self.max_det)[0]
            det = scale_boxes(im_shape[2:], copy.deepcopy(pred[:, :4]), im0_shape).round()
            det = np.concatenate([det, pred[:, 4:]], axis=1)

        return pred, det

    def get_time(self):
        return time.time()

    def output_spec(self):
        """
        Get the specs for the output tensors of the network. Useful to prepare memory allocations.
        :return: A list with two items per element, the shape and (numpy) datatype of each output tensor.
        """
        specs = []
        for o in self.outputs:
            specs.append((o['shape'], o['dtype']))
        return specs


if __name__ == "__main__":
    import cv2
    from utils.config import set_config, get_config

    set_config('./configs/config.yaml')
    cfg = get_config()

    yolov8 = Yolov8TRT(
        cfg.det_model_path, device=cfg.device, img_size=cfg.yolo_img_size, fp16=cfg.det_half, auto=False,
        gpu_num=cfg.gpu_num, conf_thres=cfg.det_conf_thres, iou_thres=cfg.yolo_nms_iou,
        agnostic=cfg.yolo_agnostic_nms, max_det=cfg.yolo_max_det, classes=cfg.det_obj_classes
    )
    yolov8.warmup(img_size=(1, 3, cfg.yolo_img_size, cfg.yolo_img_size))

    _im = cv2.imread('./data/images/sample.jpg')
    t0 = yolov8.get_time()
    _im, _im0 = yolov8.preprocess(_im)
    t1 = yolov8.get_time()
    _y = yolov8.infer(_im)
    t2 = yolov8.get_time()
    _pred, _det = yolov8.postprocess(_y, _im.shape, _im0.shape)
    t3 = yolov8.get_time()

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

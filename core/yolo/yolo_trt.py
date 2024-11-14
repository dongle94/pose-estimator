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

from core.yolo import YOLO
from core.yolo.data.augment import LetterBox
from core.yolo.util.ops import scale_boxes, non_max_suppression
from utils.logger import get_logger


class YoloTRT(YOLO):
    def __init__(self, weight: str, device: str = 'cpu', gpu_num: int = 0, img_size: int = 640, fp16: bool = False,
                 auto: bool = False, conf_thres=0.25, iou_thres=0.45, agnostic=False, max_det=100,
                 classes: Union[list, None] = None, **kwargs):
        super(YoloTRT, self).__init__()

        self.logger = get_logger()
        self.device = device
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
            is_input = True if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else False
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
            self.inputs.append(binding) if is_input else self.outputs.append(binding)

        self.img_size = img_size
        self.auto = auto
        self.names = {i: f'class{i}' for i in range(999)}
        self.letter_box = LetterBox(self.img_size, auto=self.auto)

        # parameter for postprocessing
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.agnostic = agnostic
        self.max_det = max_det
        self.kwargs = kwargs

    def warmup(self, img_size=None):
        if img_size is None:
            img_size = (1, 3, self.img_size, self.img_size)
        im = np.zeros(img_size, dtype=np.float16 if self.fp16 else np.float32)

        t = self.get_time()
        for _ in range(2):
            self.infer(im)  # warmup
        self.logger.info(f"-- {self.kwargs['model_type']} TRT Detector warmup: {self.get_time() - t:.6f} sec --")

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

        return det

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
    from utils.logger import init_logger
    from utils.config import set_config, get_config

    set_config('./configs/config.yaml')
    cfg = get_config()

    init_logger(cfg)

    _detector = YoloTRT(
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

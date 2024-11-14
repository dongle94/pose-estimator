import sys
import cv2
import time
import numpy as np
from pathlib import Path

from cuda import cudart
import tensorrt as trt

FILE = Path(__file__).resolve()
ROOT_PATH = FILE.parents[2]
if ROOT_PATH not in sys.path:
    sys.path.append(str(ROOT_PATH))

from core.vitpose import ViTPoseBase
from core.vitpose.vit_utils.inference import pad_image
from core.vitpose.vit_utils.top_down_eval import keypoints_from_heatmaps
from utils.logger import get_logger


class ViTPoseTRT(ViTPoseBase):
    def __init__(self, weight: str, device: str = 'cpu', gpu_num: int = 0, img_size: list = None, fp16: bool = False,
                 dataset_format: str = 'coco', **kwargs):
        super(ViTPoseTRT, self).__init__()

        self.logger = get_logger()
        self.device = device
        self.gpu_num = gpu_num
        self.fp16 = True if fp16 is True else False

        self.img_size = img_size
        self.dataset = dataset_format

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
            shape = list(self.engine.get_tensor_shape(name))
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
                'shape': shape,
                'allocation': allocation,
                'size': size
            }
            self.allocations.append(allocation)
            if is_input:
                self.context.set_input_shape(name, shape)
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.kwargs = kwargs

    def warmup(self, img_size=None):
        if img_size is None:
            img_size = (1, 3, self.img_size[0], self.img_size[1])
        im = np.zeros(img_size, dtype=np.float16 if self.fp16 else np.float32)  # input

        t = self.get_time()
        self.infer(im)  # warmup
        self.logger.info(f"-- {self.kwargs['model_type']} TRT Estimator warmup: {self.get_time() - t:.6f} sec --")

    def preprocess(self, im, boxes):
        pad_bbox = 10

        model_inputs = []
        orig_wh = []
        pads = []
        bboxes = []

        boxes = boxes[:, :4].round().astype(int)
        for bbox in boxes:
            # Slightly bigger bbox
            bbox[[0, 2]] = np.clip(bbox[[0, 2]] + [-pad_bbox, pad_bbox], 0, im.shape[1])
            bbox[[1, 3]] = np.clip(bbox[[1, 3]] + [-pad_bbox, pad_bbox], 0, im.shape[0])

            img_inp = im[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            img_inp, (left_pad, top_pad) = pad_image(img_inp, 3 / 4)

            org_h, org_w = img_inp.shape[:2]
            img_input = cv2.resize(img_inp, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_LINEAR)
            img_input = img_input.astype(np.float32)
            img_input /= 255.0
            img_input = ((img_input - self.mean) / self.std)
            img_input = img_input.transpose(2, 0, 1)[::-1]
            model_inputs.append(img_input)
            orig_wh.append([org_w, org_h])
            pads.append([left_pad, top_pad])
            bboxes.append(bbox)

        model_inputs = np.stack(model_inputs, axis=0)
        model_inputs = np.ascontiguousarray(model_inputs)
        model_inputs = model_inputs.astype(np.float16) if self.fp16 else model_inputs.astype(np.float32)

        return model_inputs, orig_wh, pads, bboxes

    def infer(self, inputs):
        outputs = []
        for img in inputs:
            outs = []
            for shape, dtype in self.output_spec():
                outs.append(np.zeros(shape, dtype))

            device_ptr = self.inputs[0]['allocation']
            host_arr = img
            nbytes = host_arr.size * host_arr.itemsize
            cudart.cudaMemcpy(device_ptr, host_arr.data, nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

            self.context.execute_v2(self.allocations)
            for o in range(len(outs)):
                host_arr = outs[o][0]
                device_ptr = self.outputs[o]['allocation']
                nbytes = host_arr.size * host_arr.itemsize
                cudart.cudaMemcpy(host_arr, device_ptr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
                outputs.append(host_arr)
        outputs = np.array(outputs)

        return outputs

    def postprocess(self, preds, orig_wh, pads, bboxes):
        frame_kpts = []
        for pred, (orig_w, orig_h), (l_pad, t_pad), bbox in zip(preds, pads, orig_wh, bboxes):
            points, prob = keypoints_from_heatmaps(heatmaps=np.expand_dims(pred, axis=0).astype(np.float32),
                                                   center=np.array([[orig_w // 2,
                                                                     orig_h // 2]]),
                                                   scale=np.array([[orig_w, orig_h]]),
                                                   unbiased=True, use_udp=True)

            kpts = np.concatenate([points, prob], axis=2)[0]
            kpts[:, :2] += bbox[:2] - [l_pad, t_pad]
            frame_kpts.append(kpts)

        return frame_kpts

    def output_spec(self):
        """
        Get the specs for the output tensors of the network. Useful to prepare memory allocations.
        :return: A list with two items per element, the shape and (numpy) datatype of each output tensor.
        """
        specs = []
        for o in self.outputs:
            specs.append((o['shape'], o['dtype']))
        return specs


if __name__ == '__main__':
    from core.obj_detector import ObjectDetector
    from utils.logger import init_logger
    from utils.config import set_config, get_config
    from utils.visualization import vis_pose_result

    set_config('./configs/config.yaml')
    _cfg = get_config()

    init_logger(_cfg)

    _detector = ObjectDetector(cfg=_cfg)
    _estimator = ViTPoseTRT(
        weight=_cfg.kept_model_path,
        device=_cfg.device,
        gpu_num=_cfg.gpu_num,
        img_size=_cfg.kept_img_size,
        fp16=_cfg.kept_half,
        dataset_format=_cfg.kept_format,
        model_size=_cfg.vitpose_name,
        model_type=_cfg.kept_model_type
    )
    _estimator.warmup()

    _img = cv2.imread('./data/images/army.jpg')

    t0 = _detector.detector.get_time()
    _det = _detector.run(_img)
    t1 = _detector.detector.get_time()

    t2 = _estimator.get_time()
    _kept_inputs, _pads, _orig_wh, _bboxes = _estimator.preprocess(_img, _det)
    t3 = _estimator.get_time()
    _kept_pred = _estimator.infer(_kept_inputs)
    t4 = _estimator.get_time()
    _kept_pred = _estimator.postprocess(_kept_pred, _orig_wh, _pads, _bboxes)
    t5 = _estimator.get_time()

    for d in _det:
        x1, y1, x2, y2 = map(int, d[:4])
        cls = int(d[5])
        cv2.rectangle(_img, (x1, y1), (x2, y2), (96, 96, 216), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(_img, str(_detector.names[cls]), (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (96, 96, 96), thickness=1, lineType=cv2.LINE_AA)

    if len(_kept_pred):
        _img = vis_pose_result(_img, pred_kepts=_kept_pred, model=_estimator.dataset)

    cv2.imshow('_', _img)
    cv2.waitKey(0)
    get_logger().info(f"Detector: {t1 - t0:.6f} / pre:{t3 - t2:.6f} / infer: {t4 - t3:.6f} / post: {t5 - t4:.6f}")

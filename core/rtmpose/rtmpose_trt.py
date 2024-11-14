import cv2
import sys
import time
import numpy as np
from pathlib import Path

from cuda import cudart
import tensorrt as trt

FILE = Path(__file__).resolve()
ROOT_PATH = FILE.parents[2]
if ROOT_PATH not in sys.path:
    sys.path.append(str(ROOT_PATH))

from core.rtmpose import RMTPose
from core.rtmpose.rtmpose_utils.preprocess import bbox_xyxy2cs, top_down_affine
from core.rtmpose.rtmpose_utils.postprocess import decode
from utils.logger import get_logger


class RMTPoseTRT(RMTPose):
    def __init__(self, weight: str, device: str = 'cpu',gpu_num: int = 0, img_size: list = None, fp16: bool = False,
                 dataset_format: str = 'coco', **kwargs):
        super(RMTPoseTRT, self).__init__()

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
            is_input = True if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else False
            size = np.dtype(trt.nptype(dtype)).itemsize

            for idx, s in enumerate(shape):
                if s == -1:
                    s = 1
                    shape[idx] = 1
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

        self.mean = (123.675, 116.28, 103.53)
        self.std = (58.395, 57.12, 57.375)
        self.kwargs = kwargs

    def warmup(self, img_size=None):
        if img_size is None:
            img_size = (1, 3, self.img_size[0], self.img_size[1])
        im = np.zeros(img_size, dtype=np.float16 if self.fp16 else np.float32)  # input

        t = self.get_time()
        for _ in range(2):
            self.infer(im)  # warmup
        self.logger.info(f"-- {self.kwargs['model_type']} TRT Estimator warmup: {self.get_time() - t:.6f} sec --")

    def preprocess(self, im, boxes):
        centers = []
        scales = []
        inputs = []
        for box in boxes[:, :4]:
            center, scale = bbox_xyxy2cs(box, padding=1.25)

            # do affine transformation
            input_size = [self.img_size[1], self.img_size[0]]
            resized_img, scale = top_down_affine(input_size, scale, center, im)
            centers.append(center)
            scales.append(scale)

            # normalize image
            resized_img = (resized_img - self.mean) / self.std
            resized_img = resized_img.transpose((2, 0, 1))[::-1]
            resized_img = resized_img.astype(np.float16) if self.fp16 else resized_img.astype(np.float32)
            inputs.append(resized_img)

        inputs = np.stack(inputs, axis=0)
        inputs = np.ascontiguousarray(inputs)
        inputs = inputs.astype(np.float16) if self.fp16 else inputs.astype(np.float32)

        return inputs, centers, scales

    def infer(self, imgs):
        outputs = []
        simcc_x = []
        simcc_y = []
        for img in imgs:
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
                if self.outputs[o]['name'] == 'simcc_x':
                    simcc_x.append(host_arr)
                elif self.outputs[o]['name'] == 'simcc_y':
                    simcc_y.append(host_arr)
        simcc_x, simcc_y = np.array(simcc_x), np.array(simcc_y)

        outputs.append(simcc_x)
        outputs.append(simcc_y)

        return outputs

    def postprocess(self, preds, centers, scales, simcc_split_ratio: float = 2.0):
        simcc_x, simcc_y = preds
        heatmaps = np.einsum('ijk,ijm->ijmk', simcc_x, simcc_y)
        keypoints, scores = decode(simcc_x, simcc_y, simcc_split_ratio)

        # rescale keypoints
        _keypoints = []
        input_size = [self.img_size[1], self.img_size[0]]
        for keypoint, center, scale in zip(keypoints, centers, scales):
            keypoint = keypoint / input_size * scale
            keypoint = keypoint + center - scale / 2
            _keypoints.append(keypoint)
        kepts = np.array(_keypoints)
        preds = np.concatenate((kepts, np.expand_dims(scores, axis=-1)), axis=2)

        return preds, heatmaps

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
    _estimator = RMTPoseTRT(
        weight=_cfg.kept_model_path,
        device=_cfg.device,
        gpu_num=_cfg.gpu_num,
        img_size=_cfg.kept_img_size,
        fp16=_cfg.kept_half,
        dataset_format=_cfg.kept_format,
        model_type=_cfg.kept_model_type
    )
    _estimator.warmup()

    _img = cv2.imread('./data/images/sample.jpg')

    t0 = _detector.detector.get_time()
    _det = _detector.run(_img)
    t1 = _detector.detector.get_time()

    t2 = _estimator.get_time()
    _kept_inputs, _centers, _scales = _estimator.preprocess(_img, _det)
    t3 = _estimator.get_time()
    _kept_pred = _estimator.infer(_kept_inputs)
    t4 = _estimator.get_time()
    _kept_pred, _raw_score = _estimator.postprocess(_kept_pred, np.asarray(_centers), np.asarray(_scales))
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

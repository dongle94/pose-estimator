import sys
import cv2
import math
import numpy as np
from pathlib import Path

from cuda import cudart
import tensorrt as trt

FILE = Path(__file__).resolve()
ROOT_PATH = FILE.parents[2]
if ROOT_PATH not in sys.path:
    sys.path.append(str(ROOT_PATH))

from core.hrnet_pose import PoseHRNet
from core.hrnet_pose.hrnet_utils.transforms import box_to_center_scale, get_affine_transform, transform_preds
from core.hrnet_pose.hrnet_utils.inference import get_max_preds
from utils.logger import get_logger


class PoseHRNetTRT(PoseHRNet):
    def __init__(self, weight: str, device: str = 'cpu', gpu_num: int = 0, img_size: list = None, fp16: bool = False,
                 channel: int = 32, dataset_format: str = 'coco', **kwargs):
        super(PoseHRNetTRT, self).__init__()

        self.logger = get_logger()
        self.device = device
        self.gpu_num = gpu_num
        self.fp16 = True if fp16 is True else False

        self.channel = channel
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

        self.mean = [0.485, 0.456, 0.406],
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
        centers = []
        scales = []
        rotation = 0
        model_inputs = []
        for box in boxes[:, :5]:
            center, scale = box_to_center_scale(box, self.img_size)
            centers.append(center)
            scales.append(scale)

            # do affine transformation
            trans = get_affine_transform(center, scale, rotation, (self.img_size[1], self.img_size[0]))
            input_img = cv2.warpAffine(
                im,
                trans,
                (int(self.img_size[1]), int(self.img_size[0])),
                flags=cv2.INTER_LINEAR
            )

            # normalize image
            input_img = (input_img - self.mean) / self.std
            input_img = input_img.transpose((2, 0, 1))[::-1]
            input_img = np.ascontiguousarray(input_img).astype(np.float32)
            input_img /= 255.0
            input_img = np.expand_dims(input_img, 0)
            model_inputs.append(input_img)

        inputs = np.array(model_inputs, dtype=np.float16 if self.fp16 else np.float32)

        return inputs, centers, scales

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

    def postprocess(self, preds, centers, scales):
        # raw_heatmaps -> coordinates
        batch_heatmaps = preds
        coords, maxvals = get_max_preds(batch_heatmaps)

        heatmap_height = batch_heatmaps.shape[2]
        heatmap_width = batch_heatmaps.shape[3]

        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                    diff = np.array(
                        [
                            hm[py][px + 1] - hm[py][px - 1],
                            hm[py + 1][px] - hm[py - 1][px]
                        ]
                    )
                    coords[n][p] += np.sign(diff) * .25

        preds = coords.copy()

        # Transform back
        for i in range(coords.shape[0]):
            preds[i] = transform_preds(
                coords[i], centers[i], scales[i], [heatmap_width, heatmap_height]
            )
        preds = np.concatenate((preds, maxvals), axis=2)

        return preds, batch_heatmaps

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
    _estimator = PoseHRNetTRT(
        weight=_cfg.kept_model_path,
        device=_cfg.device,
        gpu_num=_cfg.gpu_num,
        img_size=_cfg.kept_img_size,
        fp16=_cfg.kept_half,
        dataset_format=_cfg.kept_format,
        model_type=_cfg.kept_model_type
    )
    _estimator.warmup()

    _img = cv2.imread('./data/images/army.jpg')

    t0 = _detector.detector.get_time()
    _det = _detector.run(_img)
    t1 = _detector.detector.get_time()

    _input_img = _img.copy()
    t2 = _estimator.get_time()
    _kept_inputs, _centers, _scales = _estimator.preprocess(_input_img, _det)
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

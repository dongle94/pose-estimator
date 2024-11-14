import cv2
import sys
import time
import numpy as np
from pathlib import Path

import onnxruntime as ort

FILE = Path(__file__).resolve()
ROOT_PATH = FILE.parents[2]
if ROOT_PATH not in sys.path:
    sys.path.append(str(ROOT_PATH))

from core.rtmpose import RMTPose
from core.rtmpose.rtmpose_utils.preprocess import bbox_xyxy2cs, top_down_affine
from core.rtmpose.rtmpose_utils.postprocess import decode
from utils.logger import get_logger


class RMTPoseORT(RMTPose):
    def __init__(self, weight: str, device: str = 'cpu', gpu_num: int = 0, img_size: list = None, fp16: bool = False,
                 dataset_format: str = 'coco', **kwargs):
        super(RMTPoseORT, self).__init__()

        self.logger = get_logger()
        self.device = device
        self.gpu_num = gpu_num
        self.cuda = ort.get_device() == 'GPU' and device == 'cuda'
        self.fp16 = True if fp16 is True else False

        self.img_size = img_size
        self.dataset = dataset_format

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
        self.sess = ort.InferenceSession(path_or_bytes=weight, providers=providers)

        self.input_name = self.sess.get_inputs()[0].name
        self.input_shape = self.sess.get_inputs()[0].shape
        self.output_names = [o.name for o in self.sess.get_outputs()]
        self.output_shapes = [output.shape for output in self.sess.get_outputs()]
        self.output_types = [output.type for output in self.sess.get_outputs()]

        self.mean = (123.675, 116.28, 103.53)
        self.std = (58.395, 57.12, 57.375)
        self.kwargs = kwargs

    def warmup(self, img_size=None):
        if img_size is None:
            img_size = (1, 3, self.img_size[0], self.img_size[1])
        im = np.zeros(img_size, dtype=np.float16 if self.fp16 else np.float32)

        t = self.get_time()
        for _ in range(2):
            self.infer(im)  # warmup
        self.logger.info(f"-- {self.kwargs['model_type']} Onnx Estimator warmup: {time.time() - t:.6f} sec --")

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
            resized_img = np.ascontiguousarray(resized_img).astype(np.float32)
            inputs.append(resized_img)

        inputs = np.array(inputs, dtype=np.float32)

        return inputs, centers, scales

    def infer(self, inputs):
        ret = self.sess.run(self.output_names, {self.input_name: inputs})
        return ret

    def postprocess(self, preds, centers, scales, simcc_split_ratio: float = 2.0):
        """Postprocess for RTMPose model output.

            Args:
                preds (np.ndarray): Output of RTMPose model.
                centers (tuple): Center of bbox in shape (x, y).
                scales (tuple): Scale of bbox in shape (w, h).
                simcc_split_ratio (float): Split ratio of simcc.

            Returns:
                tuple:
                - keypoints (np.ndarray): Rescaled keypoints.
                - scores (np.ndarray): Model predict scores.
            """
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


if __name__ == '__main__':
    from core.obj_detector import ObjectDetector
    from utils.logger import init_logger
    from utils.config import set_config, get_config
    from utils.visualization import vis_pose_result

    set_config('./configs/config.yaml')
    _cfg = get_config()

    init_logger(_cfg)

    _detector = ObjectDetector(cfg=_cfg)
    _estimator = RMTPoseORT(
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

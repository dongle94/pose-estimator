import sys
import cv2
import math
import numpy as np
from pathlib import Path

import onnxruntime as ort

FILE = Path(__file__).resolve()
ROOT_PATH = FILE.parents[2]
if ROOT_PATH not in sys.path:
    sys.path.append(str(ROOT_PATH))

from core.hrnet_pose import PoseHRNet
from core.hrnet_pose.hrnet_utils.transforms import box_to_center_scale, get_affine_transform, transform_preds
from core.hrnet_pose.hrnet_utils.inference import get_max_preds
from utils.logger import get_logger


class PoseHRNetORT(PoseHRNet):
    def __init__(self, weight: str, device: str = 'cpu', gpu_num: int = 0, img_size: list = None, fp16: bool = False,
                 channel: int = 32,  dataset_format: str = 'coco', **kwargs):
        super(PoseHRNetORT, self).__init__()

        self.logger = get_logger()
        self.device = device
        self.gpu_num = gpu_num
        self.cuda = ort.get_device() == 'GPU' and device == 'cuda'
        self.fp16 = True if fp16 is True else False

        self.channel = channel
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
        self.sess = ort.InferenceSession(weight, providers=providers)
        self.io_binding = self.sess.io_binding()
        if self.cuda:
            self.io_binding.bind_output('outputs')

        self.input_name = self.sess.get_inputs()[0].name
        self.output_names = [o.name for o in self.sess.get_outputs()]
        self.output_shapes = [output.shape for output in self.sess.get_outputs()]
        self.output_types = [output.type for output in self.sess.get_outputs()]

        self.mean = [0.485, 0.456, 0.406],
        self.std = [0.229, 0.224, 0.225]
        self.kwargs = kwargs

    def warmup(self, img_size=None):
        if img_size is None:
            img_size = (1, 3, self.img_size[0], self.img_size[1])
        im = np.zeros(img_size, dtype=np.float16 if self.fp16 else np.float32)

        t = self.get_time()
        for _ in range(2):
            self.infer(im)  # warmup
        self.logger.info(f"-- {self.kwargs['model_type']} Onnx Estimator warmup: {self.get_time()-t:.6f} sec --")

    def preprocess(self, im, boxes):
        """
        HRNet Pre-processing

        :param im: ndarray - original input image array
        :param boxes: ndarray - [batch, 6] 6 is [x,y,x,y,conf,class]
        :return:
        """

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
            input_img = np.ascontiguousarray(input_img).astype(np.float32)
            input_img /= 255.0
            input_img = (input_img - self.mean) / self.std
            input_img = input_img.transpose((2, 0, 1))[::-1]
            input_img = np.expand_dims(input_img, 0)
            model_inputs.append(input_img)

        inputs = np.array(model_inputs, dtype=np.float16 if self.fp16 else np.float32)

        return inputs, centers, scales

    def infer(self, inputs):
        rets = []
        for img in inputs:
            if self.device == 'cuda':
                im_ortval = ort.OrtValue.ortvalue_from_numpy(img, self.device, self.gpu_num)
                self.io_binding.bind_input(
                    name=self.input_name, device_type=im_ortval.device_name(), device_id=self.gpu_num,
                    element_type=img.dtype, shape=im_ortval.shape(), buffer_ptr=im_ortval.data_ptr())

                self.sess.run_with_iobinding(self.io_binding)

                ret = self.io_binding.copy_outputs_to_cpu()[0][0]
                rets.append(ret)
            else:
                ret = self.sess.run(self.output_names, {self.input_name: img})
                rets.append(np.squeeze(ret))
        rets = np.array(rets)

        return rets

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


if __name__ == '__main__':
    from core.obj_detector import ObjectDetector
    from utils.logger import init_logger
    from utils.config import set_config, get_config
    from utils.visualization import vis_pose_result

    set_config('./configs/config.yaml')
    _cfg = get_config()

    init_logger(_cfg)

    _detector = ObjectDetector(cfg=_cfg)
    _estimator = PoseHRNetORT(
        weight=_cfg.kept_model_path,
        device=_cfg.device,
        gpu_num=_cfg.gpu_num,
        img_size=_cfg.kept_img_size,
        fp16=_cfg.kept_half,
        channel=_cfg.hrnet_channel,
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
    _kept_pred, _raw_heatmaps = _estimator.postprocess(_kept_pred, np.asarray(_centers), np.asarray(_scales))
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

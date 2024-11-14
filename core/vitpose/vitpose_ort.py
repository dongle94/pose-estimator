import sys
import cv2
import time
import numpy as np
from pathlib import Path

import onnxruntime as ort

FILE = Path(__file__).resolve()
ROOT_PATH = FILE.parents[2]
if ROOT_PATH not in sys.path:
    sys.path.append(str(ROOT_PATH))

from core.vitpose import ViTPoseBase
from core.vitpose.vit_utils.inference import pad_image
from core.vitpose.vit_utils.top_down_eval import keypoints_from_heatmaps
from utils.logger import get_logger


class ViTPoseORT(ViTPoseBase):
    def __init__(self, weight: str, device: str = 'cpu', gpu_num: int = 0, img_size: list = None, fp16: bool = False,
                 dataset_format: str = 'coco', **kwargs):
        super(ViTPoseORT, self).__init__()

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
        self.io_binding = self.sess.io_binding()
        if self.cuda:
            self.io_binding.bind_output('output_0')

        self.input_name = self.sess.get_inputs()[0].name
        self.input_shape = self.sess.get_inputs()[0].shape
        self.output_names = [o.name for o in self.sess.get_outputs()]
        self.output_shapes = [output.shape for output in self.sess.get_outputs()]
        self.output_types = [output.type for output in self.sess.get_outputs()]

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.kwargs = kwargs

    def warmup(self, img_size=None):
        if img_size is None:
            img_size = (1, 3, self.img_size[0], self.img_size[1])
        im = np.zeros(img_size, dtype=np.float16 if self.fp16 else np.float32)

        t = self.get_time()
        self.infer([im])
        self.logger.info(f"-- {self.kwargs['model_type']} Onnx Estimator warmup: {time.time() - t:.6f} sec --")

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
            img_input = np.ascontiguousarray(img_input).astype(np.float32)
            img_input /= 255.0
            img_input = ((img_input - self.mean) / self.std)
            img_input = img_input.transpose(2, 0, 1)[::-1]
            img_input = np.expand_dims(img_input, 0)

            model_inputs.append(img_input)
            orig_wh.append([org_w, org_h])
            pads.append([left_pad, top_pad])
            bboxes.append(bbox)

        model_inputs = np.stack(model_inputs, axis=0)
        model_inputs = model_inputs.astype(np.float16) if self.fp16 else model_inputs.astype(np.float32)

        return model_inputs, orig_wh, pads, bboxes

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
        return rets

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


if __name__ == "__main__":
    from core.obj_detector import ObjectDetector
    from utils.logger import init_logger
    from utils.config import set_config, get_config
    from utils.visualization import vis_pose_result

    set_config('./configs/config.yaml')
    _cfg = get_config()

    init_logger(_cfg)

    _detector = ObjectDetector(cfg=_cfg)
    _estimator = ViTPoseORT(
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

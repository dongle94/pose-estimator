import sys
import time
import cv2
import numpy as np
from pathlib import Path

import torch
from torchvision import transforms

FILE = Path(__file__).resolve()
ROOT_PATH = FILE.parents[2]
if ROOT_PATH not in sys.path:
    sys.path.append(str(ROOT_PATH))

from core.vitpose import ViTPoseBase
from core.vitpose.models.model import ViTPose
from core.vitpose.vit_utils.util import dyn_model_import
from core.vitpose.vit_utils.top_down_eval import keypoints_from_heatmaps
from core.vitpose.vit_utils.inference import pad_image
from utils.torch_utils import select_device
from utils.logger import get_logger


class ViTPoseTorch(ViTPoseBase):
    def __init__(self, weight: str, device: str = 'cpu', gpu_num: int = 0, img_size: list = None,
                 fp16: bool = False, dataset_format: str = 'coco', model_size: str = None, **kwargs):
        super(ViTPoseTorch, self).__init__()

        self.logger = get_logger()
        self.device = select_device(device=device, gpu_num=gpu_num)
        if fp16 is True and self.device.type != "cpu":
            self.fp16 = True
        else:
            self.fp16 = False
            self.logger.info("ViTPose model will run using fp16 precision")

        self.img_size = img_size
        self.dataset = dataset_format
        self.model_size = model_size

        assert model_size in [None, 's', 'b', 'l', 'h'], \
            f'The model name {model_size} is not valid'
        model_cfg = dyn_model_import(self.dataset, model_size)
        model = ViTPose(model_cfg)
        ckpt = torch.load(weight, map_location=self.device)
        if 'state_dict' in ckpt:
            model.load_state_dict(ckpt['state_dict'])
        else:
            model.load_state_dict(ckpt)
        self.model = model.half() if self.fp16 else model.float()
        self.model = self.model.to(self.device).eval()

        self.pose_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.kwargs = kwargs

    def warmup(self, img_size=None):
        if img_size is None:
            img_size = (1, 3, self.img_size[0], self.img_size[1])
        im = torch.empty(*img_size, dtype=torch.half if self.fp16 else torch.float, device=self.device)

        t = self.get_time()
        for _ in range(2):
            self.infer(im)  # warmup
        self.logger.info(f"-- {self.kwargs['model_type']} Pytorch Estimator warmup: {time.time() - t:.6f} sec --")

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
            img_input = img_input[..., ::-1]
            img_input = self.pose_transform(img_input.copy())

            model_inputs.append(img_input)
            orig_wh.append([org_w, org_h])
            pads.append([left_pad, top_pad])
            bboxes.append(bbox)

        model_inputs = torch.stack(model_inputs)
        inputs = model_inputs.to(self.device).float()
        if self.fp16:
            inputs = inputs.half()

        return inputs, orig_wh, pads, bboxes

    def infer(self, inputs):
        outputs = self.model(inputs)
        return outputs

    def postprocess(self, preds, orig_wh, pads, bboxes):
        batch_heatmaps = preds.float().cpu().detach().numpy()

        frame_kpts = []
        for pred, (orig_w, orig_h), (l_pad, t_pad), bbox in zip(batch_heatmaps, pads, orig_wh, bboxes):
            points, prob = keypoints_from_heatmaps(heatmaps=np.expand_dims(pred, axis=0),
                                                   center=np.array([[orig_w // 2,
                                                                     orig_h // 2]]),
                                                   scale=np.array([[orig_w, orig_h]]),
                                                   unbiased=True, use_udp=True)

            kpts = np.concatenate([points, prob], axis=2)[0]
            kpts[:, :2] += bbox[:2] - [l_pad, t_pad]
            frame_kpts.append(kpts)

        return frame_kpts

    def get_time(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        return time.time()


if __name__ == "__main__":
    from core.obj_detector import ObjectDetector
    from utils.logger import init_logger
    from utils.config import set_config, get_config
    from utils.visualization import vis_pose_result

    set_config('./configs/config.yaml')
    _cfg = get_config()

    init_logger(_cfg)

    _detector = ObjectDetector(cfg=_cfg)
    _estimator = ViTPoseTorch(
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

    _input_img = _img.copy()
    t2 = _estimator.get_time()
    _kept_inputs, _pads, _orig_wh, _bboxes = _estimator.preprocess(_input_img, _det)
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

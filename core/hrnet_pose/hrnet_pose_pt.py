import os
import sys
import time
import cv2
import torch
import math
import numpy as np
from torchvision import transforms
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT_PATH = FILE.parents[2]
if ROOT_PATH not in sys.path:
    sys.path.append(str(ROOT_PATH))

from core.hrnet_pose import PoseHRNet
from core.hrnet_pose.models.pose_hrnet import PoseHighResolutionNet
from core.hrnet_pose.hrnet_utils.torch_utils import select_device
from core.hrnet_pose.hrnet_utils.transforms import box_to_center_scale, get_affine_transform, transform_preds
from core.hrnet_pose.hrnet_utils.inference import get_max_preds


class PoseHRNetTorch(PoseHRNet):
    def __init__(self, weight: str, device: str = 'cpu', channel: int = 32, img_size: list = None, gpu_num: int = 0,
                 fp16: bool = False):
        super(PoseHRNetTorch, self).__init__()

        self.device = select_device(device=device, gpu_num=gpu_num)
        self.channel = channel
        self.img_size = img_size

        if fp16 is True and self.device.type != "cpu":
            self.fp16 = True
        else:
            self.fp16 = False
            print("HRNet_pose pytorch model not support cpu version's fp16. It apply fp32.")

        self.dataset = None
        self.weight_cfg = self.get_cfg_file()
        print(f"Weight config path: {self.weight_cfg}")

        # get model
        from core.hrnet_pose.cfg.default import _C as pose_cfg
        pose_cfg.defrost()
        pose_cfg.merge_from_file(self.weight_cfg)
        pose_cfg.freeze()
        model = PoseHighResolutionNet(cfg=pose_cfg)
        model.load_state_dict(torch.load(weight, map_location=self.device), strict=False)
        self.model = model.half() if self.fp16 else model.float()
        self.model = self.model.to(self.device).eval()

        self.pose_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def warmup(self, img_size=None):
        if img_size is None:
            img_size = (1, 3, self.img_size[0], self.img_size[1])
        im = torch.empty(*img_size, dtype=torch.half if self.fp16 else torch.float, device=self.device)
        t = self.get_time()
        self.infer(im)
        print(f"-- HRNetPose Estimator warmup: {time.time()-t:.6f} sec --")

    def preprocess(self, im, boxes):
        """
        HRNet Pre-processing

        :param im: ndarray - original input image array
        :param boxes: ndarray - [batch, 6] 6 is [x,y,x,y,conf,class]
        :return:
        """

        centers = []
        scales = []
        for box in boxes[:, :5]:
            center, scale = box_to_center_scale(box, self.img_size)
            centers.append(center)
            scales.append(scale)

        rotation = 0
        model_inputs = []
        for center, scale in zip(centers, scales):
            trans = get_affine_transform(center, scale, rotation, (self.img_size[1], self.img_size[0]))
            # Crop smaller image of people
            model_input = cv2.warpAffine(
                im,
                trans,
                (int(self.img_size[1]), int(self.img_size[0])),
                flags=cv2.INTER_LINEAR)
            # hwc -> 1chw
            model_input = self.pose_transform(model_input)  # .unsqueeze(0)
            model_inputs.append(model_input)

        # n * 1chw -> nchw
        model_inputs = torch.stack(model_inputs)
        inputs = model_inputs.to(self.device)
        if self.fp16:
            inputs = inputs.half()

        return inputs, centers, scales

    def infer(self, inputs):
        output = self.model(inputs)
        return output

    def postprocess(self, preds, centers, scales):
        batch_heatmaps = preds.cpu().detach().numpy()

        # raw_heatmaps -> coordinates
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

    def get_cfg_file(self):
        img_size = self.img_size
        channel = self.channel
        if img_size is None:
            img_size = [256, 192]
        if img_size[0] == img_size[1]:
            self.dataset = "mpii"
        else:
            self.dataset = "coco"
        cfg_file = f"{self.dataset}_w{channel}_{img_size[0]}x{img_size[1]}.yaml"
        cfg_path = os.path.join(os.path.dirname(__file__), 'cfg', cfg_file)

        return cfg_path

    def get_time(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        return time.time()


if __name__ == '__main__':
    from core.obj_detector import ObjectDetector
    from utils.logger import init_logger, get_logger
    from utils.config import set_config, get_config
    from utils.visualization import vis_pose_result

    set_config('./configs/config.yaml')
    _cfg = get_config()

    init_logger(_cfg)
    _logger = get_logger()

    _detector = ObjectDetector(cfg=_cfg)
    _estimator = PoseHRNetTorch(
        weight=_cfg.kept_model_path,
        device=_cfg.device,
        channel=_cfg.hrnet_channel,
        img_size=_cfg.kept_img_size,
        gpu_num=_cfg.gpu_num,
        fp16=_cfg.kept_half
    )
    _estimator.warmup()

    img = cv2.imread('./data/images/army.jpg')

    t0 = _detector.detector.get_time()
    _det = _detector.run(img)
    _det = _det.detach().cpu().numpy()
    t1 = _detector.detector.get_time()

    input_img = img.copy()
    t2 = _estimator.get_time()
    _kept_inputs, _centers, _scales = _estimator.preprocess(input_img, _det)
    t3 = _estimator.get_time()
    _kept_pred = _estimator.infer(_kept_inputs)
    t4 = _estimator.get_time()
    _kept_pred, _raw_heatmaps = _estimator.postprocess(_kept_pred, np.asarray(_centers), np.asarray(_scales))
    t5 = _estimator.get_time()

    for d in _det:
        x1, y1, x2, y2 = map(int, d[:4])
        cls = int(d[5])
        cv2.rectangle(img, (x1, y1), (x2, y2), (96, 96, 216), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(img, str(_detector.names[cls]), (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (96, 96, 96), thickness=1, lineType=cv2.LINE_AA)

    if len(_kept_pred):
        img = vis_pose_result(img, pred_kepts=_kept_pred, model=_estimator.dataset)

    cv2.imshow('_', img)
    cv2.waitKey(0)
    print(f"Detector: {t1 - t0:.6f} / pre:{t3 - t2:.6f} / infer: {t4 - t3:.6f} / post: {t5 - t4:.6f}")

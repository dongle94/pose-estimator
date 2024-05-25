import os
import sys
import numpy as np

from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
os.chdir(ROOT)

from utils.logger import get_logger


class PoseEstimator(object):
    def __init__(self, cfg=None):
        self.logger = get_logger()
        self.cfg = cfg

        # Keypoint Detector model configuration
        if os.path.abspath(cfg.kept_model_path) != cfg.kept_model_path:
            weight = os.path.abspath(os.path.join(ROOT, cfg.kept_model_path))
        else:
            weight = os.path.abspath(cfg.kept_model_path)
        self.estimator_type = cfg.kept_model_type.lower()

        self.framework = None

        device = cfg.device
        gpu_num = cfg.gpu_num
        fp16 = cfg.kept_half
        img_size = cfg.kept_img_size

        if self.estimator_type == "hrnet":
            channel = cfg.hrnet_channel
            ext = os.path.splitext(weight)[1]
            if ext in [".pt", ".pth"]:
                from core.hrnet_pose.hrnet_pose_pt import PoseHRNetTorch
                model = PoseHRNetTorch
                self.framework = 'torch'
            else:
                raise FileNotFoundError("No pose_hrnet weight File!")
            self.estimator = model(
                weight=weight,
                device=device,
                channel=channel,
                img_size=img_size,
                gpu_num=gpu_num,
                fp16=fp16
            )

        else:
            raise NotImplementedError(f'Unknown estimator type: {self.estimator_type}')

        # warm up
        self.estimator.warmup(img_size=(1, 3, img_size[0], img_size[1]))
        get_logger().info(f"Successfully loaded weight from {weight}")

        # logging
        self.f_cnt = 0
        self.ts = [0., 0., 0.]

    def run(self, img, boxes):
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().detach().numpy()

        if self.estimator_type in ['hrnet']:
            # boxes coords are ltrb
            t0 = self.estimator.get_time()
            inp, centers, scales = self.estimator.preprocess(img, boxes)

            t1 = self.estimator.get_time()
            kept_pred = self.estimator.infer(inp)

            t2 = self.estimator.get_time()
            kept_pred, raw_heatmaps = self.estimator.postprocess(
                preds=kept_pred, center=np.asarray(centers), scale=np.asarray(scales))
            t3 = self.estimator.get_time()

            # calculate time & logging
            self.f_cnt += 1
            self.ts[0] += t1 - t0
            self.ts[1] += t2 - t1
            self.ts[2] += t3 - t2
            if self.f_cnt % self.cfg.console_log_interval == 0:
                self.logger.debug(
                    f"{self.estimator_type} estimator {self.f_cnt} Frames average time - "
                    f"preproc: {self.ts[0] / self.f_cnt:.6f} sec / "
                    f"infer: {self.ts[1] / self.f_cnt:.6f} sec / "
                    f"postproc: {self.ts[2] / self.f_cnt:.6f} sec")
        else:
            kept_pred, raw_heatmaps = None, None

        return kept_pred, raw_heatmaps

if __name__ == "__main__":
    import cv2
    import time
    from core.obj_detector import ObjectDetector
    from core.media_loader import MediaLoader
    from utils.logger import init_logger
    from utils.config import set_config, get_config

    set_config('./configs/config.yaml')
    _cfg = get_config()

    init_logger(_cfg)
    _logger = get_logger()

    _detector = ObjectDetector(cfg=_cfg)
    _estimator = PoseEstimator(cfg=_cfg)

    _bgr = getattr(_cfg, 'media_bgr', True)
    _realtime = getattr(_cfg, 'media_realtime', False)
    media_loader = MediaLoader(_cfg.media_source,
                               logger=_logger,
                               realtime=_realtime,
                               bgr=_bgr,
                               opt=_cfg)
    wt = 1 / media_loader.dataset.fps

    while True:
        st = time.time()
        frame = media_loader.get_frame()

        _det = _detector.run(frame)

        if len(_det):
            kept_preds, heatmaps = _estimator.run(frame, _det)

        for d in _det:
            x1, y1, x2, y2 = map(int, d[:4])
            cls = int(d[5])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (96, 96, 216), thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(frame, str(_detector.names[cls]), (x1, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (96, 96, 96), thickness=1, lineType=cv2.LINE_AA)

        cv2.imshow('_', frame)
        if cv2.waitKey(1) == ord('q'):
            print("-- CV2 Stop --")
            break

        et = time.time()
        if et - st < wt:
            time.sleep(wt - (et - st))
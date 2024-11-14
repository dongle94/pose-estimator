import os
import sys
import numpy as np
import torch
from pathlib import Path

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
        self.kept_format = cfg.kept_format

        if self.estimator_type == "hrnet":
            channel = cfg.hrnet_channel
            ext = os.path.splitext(weight)[1]
            if ext in [".pt", ".pth"]:
                from core.hrnet_pose.hrnet_pose_pt import PoseHRNetTorch
                model = PoseHRNetTorch
                self.framework = 'torch'
            elif ext in [".onnx"]:
                from core.hrnet_pose.hrnet_pose_ort import PoseHRNetORT
                model = PoseHRNetORT
                self.framework = 'onnx'
            elif ext in [".engine"]:
                from core.hrnet_pose.hrnet_pose_trt import PoseHRNetTRT
                model = PoseHRNetTRT
                self.framework = 'trt'
            else:
                raise FileNotFoundError("No HRNet Pose weight File!")
            self.estimator = model(
                weight=weight,
                device=device,
                gpu_num=gpu_num,
                img_size=img_size,
                fp16=fp16,
                channel=channel,
                dataset_format=self.kept_format,
                model_type=self.estimator_type
            )
        elif self.estimator_type == "rtmpose":
            ext = os.path.splitext(weight)[1]
            if ext in [".onnx"]:
                from core.rtmpose.rtmpose_ort import RMTPoseORT
                model = RMTPoseORT
                self.framework = 'onnx'
            elif ext in [".engine"]:
                from core.rtmpose.rtmpose_trt import RMTPoseTRT
                model = RMTPoseTRT
                self.framework = 'trt'
            else:
                raise FileNotFoundError("No rtmpose weight File!")
            self.estimator = model(
                weight=weight,
                device=device,
                img_size=img_size,
                gpu_num=gpu_num,
                fp16=fp16,
                dataset_format=self.kept_format,
                model_type=self.estimator_type
            )
        elif self.estimator_type == 'vitpose':
            model_name = cfg.vitpose_name
            ext = os.path.splitext(weight)[1]
            if ext in [".pt", ".pth"]:
                from core.vitpose.vitpose_pt import ViTPoseTorch
                model = ViTPoseTorch
                self.framework = 'torch'
            elif ext in [".onnx"]:
                from core.vitpose.vitpose_ort import ViTPoseORT
                model = ViTPoseORT
                self.framework = 'onnx'
            elif ext in [".engine"]:
                from core.vitpose.vitpose_trt import ViTPoseTRT
                model = ViTPoseTRT
                self.framework = 'trt'
            else:
                raise FileNotFoundError("No vitpose weight File!")
            self.estimator = model(
                weight=weight,
                device=device,
                gpu_num=gpu_num,
                img_size=img_size,
                fp16=fp16,
                dataset_format=self.kept_format,
                model_size=model_name,
                model_type=self.estimator_type
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

        if self.estimator_type in ['hrnet', 'rtmpose']:
            # boxes coords are ltrb
            t0 = self.estimator.get_time()
            inp, centers, scales = self.estimator.preprocess(img, boxes)

            t1 = self.estimator.get_time()
            kept_pred = self.estimator.infer(inp)

            t2 = self.estimator.get_time()
            kept_pred, raw_heatmaps = self.estimator.postprocess(
                preds=kept_pred, centers=np.asarray(centers), scales=np.asarray(scales)
            )

            t3 = self.estimator.get_time()
        elif self.estimator_type in ['vitpose']:
            t0 = self.estimator.get_time()
            inp, pads, orig_wh, bboxes = self.estimator.preprocess(img, boxes)

            t1 = self.estimator.get_time()
            kept_pred = self.estimator.infer(inp)

            t2 = self.estimator.get_time()
            kept_pred = self.estimator.postprocess(
                kept_pred, orig_wh, pads, bboxes
            )
            kept_pred = np.array(kept_pred)
            raw_heatmaps = None
            t3 = self.estimator.get_time()
        else:
            raise NotImplementedError(f'Unknown estimator type: {self.estimator_type}')

        # calculate time & logging
        self.f_cnt += 1
        self.ts[0] += t1 - t0
        self.ts[1] += t2 - t1
        self.ts[2] += t3 - t2
        if self.f_cnt % self.cfg.console_log_interval == 0:
            self.logger.debug(
                f"{self.estimator_type} estimator {self.f_cnt} Frames average time - "
                f"Total: {sum(self.ts) / self.f_cnt:.6f} sec / "
                f"preproc: {self.ts[0] / self.f_cnt:.6f} sec / "
                f"infer: {self.ts[1] / self.f_cnt:.6f} sec / "
                f"postproc: {self.ts[2] / self.f_cnt:.6f} sec")

        return kept_pred, raw_heatmaps


if __name__ == "__main__":
    import cv2
    import time
    from core.obj_detector import ObjectDetector
    from core.media_loader import MediaLoader
    from utils.logger import init_logger
    from utils.config import set_config, get_config
    from utils.visualization import vis_pose_result

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
    wt = 0 if media_loader.is_imgs else 1 / media_loader.dataset.fps

    while True:
        st = time.time()
        frame = media_loader.get_frame()
        if frame is None:
            break

        _det = _detector.run(frame)

        _kept_preds = None
        if len(_det):
            _kept_preds, _heatmaps = _estimator.run(frame, _det)

        for d in _det:
            x1, y1, x2, y2 = map(int, d[:4])
            cls = int(d[5])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (96, 96, 216), thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(frame, str(_detector.names[cls]), (x1, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (96, 96, 96), thickness=1, lineType=cv2.LINE_AA)

        if _kept_preds is not None:
            frame = vis_pose_result(frame,
                                    pred_kepts=_kept_preds,
                                    model=_estimator.estimator.dataset,
                                    radius=max(frame.shape[0] // 300, 1),
                                    thickness=max(frame.shape[0] // 1000, 1))

        et = time.time()
        if media_loader.is_imgs:
            t = 0
        else:
            if et - st < wt:
                t = int((wt - (et - st)) * 1000) + 1
            else:
                t = 1

        cv2.imshow('_', frame)
        if cv2.waitKey(t) == ord('q'):
            print("-- CV2 Stop --")
            break

    print("-- Stop program --")

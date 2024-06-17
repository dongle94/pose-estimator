import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.logger import get_logger


class ObjectDetector(object):
    def __init__(self, cfg=None):
        self.logger = get_logger()
        self.cfg = cfg

        weight = os.path.abspath(cfg.det_model_path)
        self.detector_type = cfg.det_model_type.lower()

        device = cfg.device
        gpu_num = cfg.gpu_num
        fp16 = cfg.det_half
        conf_thres = cfg.det_conf_thres
        classes = cfg.det_obj_classes

        self.framework = None

        if self.detector_type in ["yolov5", "yolov8"]:
            img_size = cfg.yolov5_img_size
            iou_thres = cfg.yolov5_nms_iou
            agnostic = cfg.yolov5_agnostic_nms
            max_det = cfg.yolov5_max_det
            self.im_shape = None
            self.im0_shape = None
            if self.detector_type == "yolov5":
                # model load with weight
                ext = os.path.splitext(weight)[1]
                if ext in ['.pt', '.pth']:
                    from core.yolov5.yolov5_pt import Yolov5Torch
                    model = Yolov5Torch
                    self.framework = 'torch'
                elif ext == '.onnx':
                    from core.yolov5.yolov5_ort import Yolov5ORT
                    model = Yolov5ORT
                    self.framework = 'onnx'
                elif ext in ['.engine', '.bin']:
                    from core.yolov5.yolov5_trt import Yolov5TRT
                    model = Yolov5TRT
                    self.framework = 'trt'
                else:
                    raise FileNotFoundError('No Yolov5 weight File!')
            elif self.detector_type == "yolov8":
                # model load with weight
                ext = os.path.splitext(weight)[1]
                if ext in ['.pt', '.pth']:
                    from core.yolov8.yolov8_pt import Yolov8Torch
                    model = Yolov8Torch
                    self.framework = 'torch'
                elif ext == '.onnx':
                    from core.yolov8.yolov8_ort import Yolov8ORT
                    model = Yolov8ORT
                    self.framework = 'onnx'
                elif ext in ['.engine', '.bin']:
                    from core.yolov8.yolov8_trt import Yolov8TRT
                    model = Yolov8TRT
                    self.framework = 'trt'
                else:
                    raise FileNotFoundError('No Yolov8 weight File!')
            else:
                raise NotImplementedError(f'Unknown detector type: {self.detector_type}')
            self.detector = model(
                weight=weight,
                device=device,
                img_size=img_size,
                fp16=fp16,
                gpu_num=gpu_num,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                agnostic=agnostic,
                max_det=max_det,
                classes=classes
            )
            self.names = self.detector.names

            # warm up
            self.detector.warmup(img_size=(1, 3, img_size, img_size))
            self.logger.info(f"Successfully loaded weight from {weight}")

        # logging
        self.f_cnt = 0
        self.ts = [0., 0., 0.]

    def run(self, img):
        if self.detector_type in ["yolov5", "yolov8"]:
            t0 = self.detector.get_time()

            img, orig_img = self.detector.preprocess(img)
            im_shape = img.shape
            im0_shape = orig_img.shape
            t1 = self.detector.get_time()

            preds = self.detector.infer(img)
            t2 = self.detector.get_time()

            pred, det = self.detector.postprocess(preds, im_shape, im0_shape)
            t3 = self.detector.get_time()

            # calculate time & logging
            self.f_cnt += 1
            self.ts[0] += t1 - t0
            self.ts[1] += t2 - t1
            self.ts[2] += t3 - t2
            if self.f_cnt % self.cfg.console_log_interval == 0:
                self.logger.debug(
                    f"{self.detector_type} detector {self.f_cnt} Frames average time - "
                    f"preproc: {self.ts[0]/self.f_cnt:.6f} sec / "
                    f"infer: {self.ts[1] / self.f_cnt:.6f} sec / " 
                    f"postproc: {self.ts[2] / self.f_cnt:.6f} sec")

        else:
            pred, det = None, None

        return det

    def run_np(self, img):
        det = self.run(img)
        if self.framework == 'torch':
            det = det.cpu().numpy()
        return det


if __name__ == "__main__":
    import cv2
    import time
    from core.media_loader import MediaLoader
    from utils.logger import init_logger
    from utils.config import set_config, get_config

    set_config('./configs/config.yaml')
    _cfg = get_config()

    init_logger(_cfg)
    _logger = get_logger()

    _detector = ObjectDetector(cfg=_cfg)

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

        _det = _detector.run(frame)

        for d in _det:
            x1, y1, x2, y2 = map(int, d[:4])
            cls = int(d[5])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (96, 96, 216), thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(frame, str(_detector.names[cls]), (x1, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (96, 96, 96), thickness=1, lineType=cv2.LINE_AA)

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

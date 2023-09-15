import os
import sys
import numpy as np

from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
os.chdir(ROOT)

from utils.logger import get_logger


class PoseDetector(object):
    def __init__(self, cfg=None):
        self.cfg = cfg

        # Keypoint Detector model configuration
        if os.path.abspath(cfg.KEPT_MODEL_PATH) != cfg.KEPT_MODEL_PATH:
            weight = os.path.abspath(os.path.join(ROOT, cfg.KEPT_MODEL_PATH))
        else:
            weight = os.path.abspath(cfg.KEPT_MODEL_PATH)
        self.estimator_type = cfg.KEPT_MODEL_TYPE.lower()

        if self.estimator_type == "hrnet":
            device = cfg.DEVICE
            fp16 = cfg.KEPT_HALF
            img_size = cfg.KEPT_IMG_SIZE
            weight_cfg = cfg.KEPT_MODEL_CNF

            # model load with weight
            ext = os.path.splitext(weight)[1]
            if ext in [".pt", ".pth"]:
                from core.pose_estimator.hrnet import HRNet
                self.detector = HRNet(weight=weight, weight_cfg=weight_cfg, device=device, img_size=img_size, fp16=fp16)

            # warm up
            self.detector.warmup(imgsz=(1, 3, img_size[0], img_size[1]))
            get_logger().info(f"Successfully loaded weight from {weight}")

    def preprocess(self, img, boxes):
        # boxes coords are ltrb
        inp, _centers, _scales = self.detector.preprocess(img, boxes)
        return inp, _centers, _scales

    def detect(self, inputs):
        _preds = self.detector.infer(inputs)

        return _preds

    def postprocess(self, _preds, _centers, _scales):
        _preds, _raw_heatmaps = self.detector.postprocess(_preds, _centers, _scales)
        return _preds, _raw_heatmaps


if __name__ == "__main__":
    import time
    import cv2
    from utils.config import _C as _cfg
    from utils.config import update_config
    from utils.logger import init_logger
    from core.obj_detectors import ObjectDetector
    from utils.medialoader import MediaLoader
    from utils.visualization import vis_pose_result, get_heatmaps, merge_heatmaps

    # get config
    update_config(_cfg, args='./configs/config.yaml')
    init_logger(cfg=_cfg)
    logger = get_logger()

    # get detectors
    obj_detector = ObjectDetector(cfg=_cfg)
    kept_detector = PoseDetector(cfg=_cfg)

    # get media loader by params
    s = sys.argv[1]
    media_loader = MediaLoader(s, realtime=True)
    media_loader.start()

    while media_loader.is_frame_ready() is False:
        time.sleep(0.001)
        continue

    f_cnt = 0
    ts = [0., 0., 0.]
    while True:
        # Get Input frame
        frame = media_loader.get_frame()

        if frame is None:
            logger.info("Frame is None -- Break main loop")
            break

        # Human Detection
        t0 = time.time()
        im = obj_detector.preprocess(frame)
        pred = obj_detector.detect(im)
        pred, det = obj_detector.postprocess(pred)
        t1 = time.time()

        # Pose Detection
        if len(det):
            inps, centers, scales = kept_detector.preprocess(frame, det)
            preds = kept_detector.detect(inps)
            rets, raw_heatmaps = kept_detector.postprocess(preds, centers, scales)

            # Keypoints process
            new_raw_heatmaps = raw_heatmaps[:]
            raw_heatmaps = np.asarray(new_raw_heatmaps)

            heatmaps = get_heatmaps(raw_heatmaps, colormap=None, draw_index=None)
            heatmap = merge_heatmaps(heatmaps, det, frame.shape)
        else:
            rets = None
            heatmap = None
        t2 = time.time()

        # Show Processed Videos
        for d in det:
            x1, y1, x2, y2 = map(int, d[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (96, 96, 216), thickness=2, lineType=cv2.LINE_AA)
        # if rets is not None:
        #     frame = vis_pose_result(model=None, img=frame, result=rets)
        cv2.imshow('_', frame)

        # Show Processed Heatmap
        if heatmap is not None:
            if len(heatmap.shape) == 2:
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            new_heatmap = cv2.add((0.4 * heatmap).astype(np.uint8), frame)
            cv2.imshow('heatmap', new_heatmap)
        t3 = time.time()

        ts[0] += (t1 - t0)
        ts[1] += (t2 - t1)
        ts[2] += (t3 - t2)

        if cv2.waitKey(1) == ord('q'):
            logger.info("-- CV2 Stop by Keyboard Input --")
            break

        f_cnt += 1
        if f_cnt % _cfg.CONSOLE_LOG_INTERVAL == 0:
            logger.debug(
                f"{f_cnt} Frame - obj_det: {ts[0] / f_cnt:.4f} / kept_det: {ts[1] / f_cnt:.4f} / vis: {ts[2] / f_cnt:.4f}")

        time.sleep(0.0001)
    media_loader.stop()
    logger.info("-- Stop program --")


import os
import sys
import numpy as np

from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
os.chdir(ROOT)

from detectors.hrnet import HRNet

class PoseDetector(object):
    def __init__(self, cfg=None):
        # Keypoint Detector model configuration
        if os.path.abspath(cfg.KEPT_MODEL_PATH) != cfg.KEPT_MODEL_PATH:
            weight = os.path.abspath(os.path.join(ROOT, cfg.KEPT_MODEL_PATH))
        else:
            weight = os.path.abspath(cfg.KEPT_MODEL_PATH)
        device = cfg.DEVICE
        fp16 = cfg.KEPT_HALF
        img_size = cfg.KEPT_IMG_SIZE

        # Model load with weight
        self.detector = HRNet(weight=weight, device=device, img_size=img_size, fp16=fp16)

        # warm up
        self.detector.warmup(imgsz=(1, 3, img_size[0], img_size[1]))

    def preprocess(self, img, boxes):
        # boxes coords are ltrb
        inp, centers, scales = self.detector.preprocess(img, boxes)
        return inp, centers, scales

    def detect(self, inputs):
        preds = self.detector.forward(inputs)

        return preds

    def postprocess(self, preds, centers, scales):
        preds, raw_heatmaps = self.detector.postprocess(preds, centers, scales)
        return preds, raw_heatmaps


def test():
    import time
    import cv2
    from detectors.obj_detector import HumanDetector
    from utils.medialoader import MediaLoader
    from utils.visualization import vis_pose_result, get_heatmaps, merge_heatmaps

    # get detectors
    obj_detector = HumanDetector(cfg=cfg)
    kept_detector = PoseDetector(cfg=cfg)

    # get media loader by params
    s = sys.argv[1]
    media_loader = MediaLoader(s)

    while media_loader.img is None:
        time.sleep(0.001)
        continue

    while media_loader.cap.isOpened():
        # Get Input frame
        frame = media_loader.get_frame()
        if frame is None:
            logger.info("Frame is None -- Break main loop")
            break

        # Human Detection
        im = obj_detector.preprocess(frame)
        pred = obj_detector.detect(im)
        pred, det = obj_detector.postprocess(pred)

        # Pose Detection
        if det.size()[0]:
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

        # Show Processed Videos
        for d in det:
            x1, y1, x2, y2 = map(int, d[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (96, 96, 216), thickness=2, lineType=cv2.LINE_AA)
        if rets is not None:
            frame = vis_pose_result(model=None, img=frame, result=rets)
        cv2.imshow('_', frame)

        # Show Processed Heatmap
        if heatmap is not None:
            if len(heatmap.shape) == 2:
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            new_heatmap = cv2.add((0.4 * heatmap).astype(np.uint8), frame)
            cv2.imshow('heatmap', new_heatmap)

        if cv2.waitKey(1) == ord('q'):
            logger.info("-- CV2 Stop by Keyboard Input --")
            break

        time.sleep(0.001)
    media_loader.stop()
    logger.info("-- Stop program --")


if __name__ == "__main__":
    from utils.config import _C as cfg
    from utils.config import update_config
    from utils.logger import init_logger, get_logger

    # get config
    update_config(cfg, args='./configs/config.yaml')
    print(cfg)

    # get logger
    init_logger(cfg=cfg)
    logger = get_logger()

    test()

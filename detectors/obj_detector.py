import yaml
import os
import sys

from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
os.chdir(ROOT)

from detectors.yolov5_pt import YoloDetector

class HumanDetector(object):
    def __init__(self, cfg=None):
        # Detection model configuration
        if os.path.abspath(cfg.DET_MODEL_PATH) != cfg.DET_MODEL_PATH:
            weight = os.path.abspath(os.path.join(ROOT, cfg.DET_MODEL_PATH))
        else:
            weight = os.path.abspath(cfg.DET_MODEL_PATH)
        device = cfg.DEVICE

        # model load with weight
        self.detector = YoloDetector(weight=weight, device=device, img_size=cfg.IMG_SIZE, fp16=cfg.HALF)

        # warm up
        img_size = cfg.IMG_SIZE
        self.detector.warmup(imgsz=(1, 3, img_size, img_size))

    def preprocess(self, img):
        img, orig_img = self.detector.preprocess(img)
        self.im_shape = img.shape
        self.im0_shape = orig_img.shape

        return img

    def detect(self, img):
        pred = self.detector.forward(img)

        return pred

    def postprocess(self, pred):
        max_det = 100
        ret = self.detector.postprocess(pred=pred, im_shape=self.im_shape,
                                        im0_shape=self.im0_shape, max_det=max_det)

        return ret


if __name__ == "__main__":
    import time
    import cv2
    from utils.config import _C as cfg
    from utils.config import update_config
    from utils.medialoader import MediaLoader

    update_config(cfg, args='./configs/config.yaml')
    detector = HumanDetector(cfg=cfg)

    s = sys.argv[1]
    media_loader = MediaLoader(s)
    time.sleep(1)
    while True:
        frame = media_loader.get_frame()

        im = detector.preprocess(frame)
        pred = detector.detect(im)
        ret = detector.postprocess(pred)

        for d in ret[1]:
            x1, y1, x2, y2 = map(int, d[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (96, 96, 216), thickness=2, lineType=cv2.LINE_AA)

        cv2.imshow('_', frame)
        if cv2.waitKey(1) == ord('q'):
            print("-- CV2 Stop --")
            break

    media_loader.stop()
    print("-- Stop program --")

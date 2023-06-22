import os
import sys
import torch

from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
os.chdir(ROOT)

from detectors.hrnet import HRNet

class PoseDetector(object):
    def __init__(self, cfg=None):
        if os.path.abspath(cfg.KEPT_MODEL_PATH) != cfg.KEPT_MODEL_PATH:
            weight = os.path.abspath(os.path.join(ROOT, cfg.KEPT_MODEL_PATH))
        else:
            weight = os.path.abspath(cfg.KEPT_MODEL_PATH)
        device = cfg.DEVICE
        fp16 = cfg.KEPT_HALF
        self.detector = HRNet(weight=weight, device=device, fp16=fp16)


    def preprocess(self, img, boxes):
        # boxes coords are ltrb
        self.detector.preprocess()
        pass

    def forward(self):
        pass

    def postprocess(self):
        pass


def test():
    print(cfg)

    pose_model = PoseDetector(cfg=cfg)

if __name__ == "__main__":
    from utils.config import _C as cfg
    from utils.config import update_config

    update_config(cfg, args='./configs/config.yaml')
    test()
# Ultralytics YOLO 🚀, AGPL-3.0 license
import time


__version__ = "8.3.161"

__all__ = "__version__"


class YOLO(object):
    def __init__(self):
        pass

    def warmup(self, img_size):
        pass

    def preprocess(self, img):
        pass

    def infer(self, img):
        pass

    def postprocess(self, pred, im_shape, im0_shape):
        pass

    def get_time(self):
        return time.time()

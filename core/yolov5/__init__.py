import time


class YOLOV5(object):
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

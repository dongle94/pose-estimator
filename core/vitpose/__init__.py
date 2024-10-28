import time


class ViTPoseBase(object):
    def __init__(self):
        pass

    def warmup(self, img_size):
        pass

    def preprocess(self, im, boxes):
        pass

    def infer(self, inputs):
        pass

    def postprocess(self, preds, orig_wh, pads, bboxes):
        pass

    def get_time(self):
        return time.time()

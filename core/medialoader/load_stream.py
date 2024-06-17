import cv2
import math
import time
import numpy as np
import platform
from threading import Thread

from core.medialoader import LoadSample


class LoadStream(LoadSample):
    def __init__(self, source, stride=1, opt=None, bgr=True, logger=None):
        super().__init__()

        self.logger = logger
        self.stride = stride
        self.bgr = bgr
        source = eval(source) if source.isnumeric() else source

        if platform.system() == 'Windows':
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(source)
        assert cap.isOpened(), f'Failed to open {source}'

        self.mode = 'webcam'

        if opt is not None and opt.media_opt_auto is False:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*opt.media_fourcc))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, opt.media_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, opt.media_height)
            cap.set(cv2.CAP_PROP_FPS, opt.media_fps)
        self.w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.fps = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30  # 30 FPS fallback
        self.frame = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')
        self.enable_param = opt.media_enable_param
        if self.enable_param:
            for param in opt.media_cv2_params:
                k, v = param.popitem()
                cap.set(eval(k), v)
        else:
            cap.set(cv2.CAP_PROP_SETTINGS, 0)

        if self.logger is not None:
            self.logger.info(f"-- Load Stream: {self.w}*{self.h}, FPS: {self.fps} --")
        else:
            print(f"-- Load Stream: {self.w}*{self.h}, FPS: {self.fps} --")

        _, self.img = cap.read()
        self.thread = Thread(target=self.update, args=(cap, source,), daemon=True)
        self.thread.start()

    def update(self, cap, stream):
        n, f = 0, self.frame
        while cap.isOpened() and n < f:
            n += 1

            cap.grab()
            if n % self.stride == 0:
                success, im = cap.retrieve()
                if success:
                    self.img = im
                else:
                    self.img = np.zeros_like(self.img)
                    cap.open(stream)
            time.sleep(0.0)  # wait time

    def __iter__(self):
        return self

    def __next__(self):
        if not self.thread.is_alive() or cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            raise StopIteration

        im = self.img.copy()
        if self.bgr is False:       # rgb
            im = im[..., ::-1]

        return im

    def __len__(self):
        pass


if __name__ == "__main__":
    s = '0'
    loader = LoadStream(s)
    for _im in loader:
        _im = _im[..., ::-1]
        cv2.imshow('.', _im)

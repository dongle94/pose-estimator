import cv2
import math
import time
import numpy as np
import platform
import threading
from threading import Thread

from core.medialoader import LoadSample


class LoadStream(LoadSample):
    def __init__(self, source, stride=1, opt=None, bgr=True, logger=None):
        super().__init__()

        self.logger = logger
        self.bgr = bgr
        self.stride = stride
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
            self.logger.info(f"-- Load {self.mode.title()}: {self.w}x{self.h}, FPS: {self.fps} --")
        else:
            print(f"-- Load {self.mode.title()}: {self.w}x{self.h}, FPS: {self.fps} --")

        self.cap = cap
        _, self.img = cap.read()
        self.img_lock = threading.Lock()
        self.thread = Thread(target=self.update, args=(cap, source,), daemon=True)
        self.thread.start()

    def update(self, cap, stream):
        n, f = 0, self.frame
        while cap.isOpened() and n < f:
            n += 1
            success = cap.grab()
            if not success:
                time.sleep(0.1)
                continue

            if n % self.stride == 0:
                success, im = cap.retrieve()
                with self.img_lock:
                    if success:
                        self.img = im
                    else:
                        cap.release()
                        if platform.system() == 'Windows':
                            self.cap = cap = cv2.VideoCapture(stream, cv2.CAP_DSHOW)
                        else:
                            self.cap = cap = cv2.VideoCapture(stream)
                        time.sleep(0.1)

    def __iter__(self):
        return self

    def __next__(self):
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            return None

        with self.img_lock:
            if self.img is None:
                return None
            im = self.img.copy()

        if self.bgr is False:       # rgb
            im = im[..., ::-1]

        return im

    def __len__(self):
        return float('inf')
    
    def __del__(self):
        if hasattr(self, 'thread') and self.thread.is_alive():
            # 스레드 종료 대기
            self.thread.join(timeout=1.0)
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    s = '0'
    loader = LoadStream(s)
    for _im in loader:
        _im = _im[..., ::-1]
        cv2.imshow('.', _im)

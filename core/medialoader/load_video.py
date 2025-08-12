import os
import cv2
import math
import time
import threading
from threading import Thread

from core.medialoader import LoadSample

VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv', 'webm'


class LoadVideo(LoadSample):
    def __init__(self, path, stride=1, realtime=False, bgr=True, logger=None):
        super().__init__()

        self.logger = logger
        self.bgr = bgr
        self.stride = stride
        self.realtime = realtime

        path = os.path.abspath(path)
        if path.split('.')[-1].lower() not in VID_FORMATS:
            raise FileNotFoundError(f"File ext is invalid: {path}")

        cap = cv2.VideoCapture(path)
        assert cap.isOpened(), f'Failed to open {path}'

        self.mode = 'video'

        self.w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.fps = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30  # 30 FPS fallback
        self.frame = 0
        self.frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.stride)

        if self.logger is not None:
            self.logger.info(f"-- Load {self.mode.title()}: {self.w}x{self.h}, FPS: {self.fps} --")
        else:
            print(f"-- Load {self.mode.title()}: {self.w}x{self.h}, FPS: {self.fps} --")

        self.wait_ms = 1 / self.fps
        self.cap = cap
        if self.realtime is True:
            _, self.img = cap.read()
            self.img_lock = threading.Lock()
            self.thread = Thread(target=self.update, args=(cap,), daemon=True)
            self.thread.start()

    def update(self, cap):
        n, f = 0, self.frames
        while cap.isOpened() and n < f:
            n += 1
            st = time.time()
            for _ in range(self.stride):
                cap.grab()
            ret, im = cap.retrieve()
            if not ret:
                cap.release()
                break

            with self.img_lock:
                self.img = im

            et = time.time()
            wait_t = max(0, self.wait_ms - (et - st))
            time.sleep(wait_t)
        
        with self.img_lock:
            self.img = None

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            return None

        if self.realtime is True:
            with self.img_lock:
                if self.img is None:
                    return None
                im = self.img.copy()
        else:
            for _ in range(self.stride):
                self.cap.grab()
            ret, im = self.cap.retrieve()
            if not ret:
                self.cap.release()
                return None

        if self.bgr is False:
            im = im[..., ::-1]

        return im

    def __len__(self):
        return self.frames

    def __del__(self):
        if hasattr(self, 'thread') and self.thread.is_alive():
            # 스레드 종료 대기
            self.thread.join(timeout=1.0)
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    p1 = './data/videos/sample.mp4'
    loader = LoadVideo(p1)
    for _im in loader:
        _im = _im[..., ::-1]
        cv2.imshow('.', _im)

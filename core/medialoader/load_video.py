import os
import cv2
import math
import time
import platform
from threading import Thread

from core.medialoader import LoadSample

VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv', 'webm'


class LoadVideo(LoadSample):
    def __init__(self, path, stride=1, realtime=False, bgr=True):
        super().__init__()

        self.stride = stride
        self.realtime = realtime
        self.bgr = bgr

        path = os.path.abspath(path)
        if path.split('.')[-1].lower() not in VID_FORMATS:
            raise FileNotFoundError(f"File ext is invalid: {path}")

        cap = cv2.VideoCapture(path)
        assert cap.isOpened(), f'Failed to open {path}'

        self.mode = 'video'

        self.w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.cap = cap
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.fps = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30  # 30 FPS fallback
        self.frame = 0
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.stride)

        self.wait_ms = 1 / self.fps
        if self.realtime is True:
            _, self.img = cap.read()
            self.thread = Thread(target=self.update, args=(cap,), daemon=True)
            self.thread.start()

    def update(self, cap):
        n, f = 0, self.frames
        while cap.isOpened() and n < f:
            n += 1

            st = time.time()
            for _ in range(self.stride):
                self.cap.grab()
            ret, im = self.cap.retrieve()
            while not ret:
                self.cap.release()
                break

            self.img = im
            et = time.time()
            wait_t = self.wait_ms - (et - st)
            time.sleep(wait_t)

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            raise StopIteration("User Stop Video")

        if self.realtime is True:
            if self.img is None:
                raise StopIteration("Video End")
            im = self.img.copy()
        else:
            for _ in range(self.stride):
                self.cap.grab()
            ret, im = self.cap.retrieve()
            while not ret:
                return None
            #     self.cap.release()
            #     raise StopIteration("Video End")

        if self.bgr is False:
            im = im[..., ::-1]

        return im

    def __len__(self):
        pass


if __name__ == "__main__":
    p1 = './data/videos/sample.mp4'
    loader = LoadVideo(p1)
    for _im in loader:
        _im = _im[..., ::-1]
        cv2.imshow('.', _im)

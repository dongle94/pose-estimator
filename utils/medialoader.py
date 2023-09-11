import os
import cv2
import math
import numpy as np
from pathlib import Path
from threading import Thread
import time


# include image suffixes
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'
# include video suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv', 'webm'


def check_sources(source):
    is_file, is_url, is_webcam = False, False, False
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    is_webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)

    return is_file, is_url, is_webcam


class MediaLoader(object):
    def __init__(self, source, stride=1, logger=None, realtime=True, opt=None, fast=False):
        self.stride = stride
        self.realtime = realtime
        self.is_file, self.is_url, self.is_webcam = check_sources(source)

        source = os.path.abspath(source) if os.path.isfile(source) else source
        self.source = str(source)
        self.img, self.fps, self.frame, self.thread = None, 0, 0, None

        self.logger = logger

        source = eval(source) if source.isnumeric() else source
        cap = cv2.VideoCapture(source)
        assert cap.isOpened(), f"Failed to open {source}"

        # Metadata
        self.cap = cap
        if opt is not None and opt.MEDIA_OPT_AUTO is False:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, opt.MEDIA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, opt.MEDIA_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, opt.MEDIA_FPS)
        self.w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.fps = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30  # 30 FPS fallback
        self.frame = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')

        _, self.img = cap.read()

        wait_ms = 1 / self.fps if fast is False else 0
        self.thread = Thread(target=self.update, args=([cap, source, wait_ms]), daemon=True)
        print(f"-- Success ({self.frame} frames {self.w}x{self.h} at {self.fps:.2f} FPS)")
        self.alive = True
        self.bpause = False

    def start(self):
        self.bpause = False
        self.thread.start()

    def update(self, cap, stream, wait_ms=0.01):
        n, f = 0, self.frame
        while cap.isOpened() and n < f and self.alive:
            # pause action
            if self.bpause is True:
                time.sleep(0.001)
                continue
            # if not realtime, process when img array is exist.
            if self.realtime is False and self.img is not None:
                time.sleep(0.001)
                continue
            # process
            n += 1
            cap.grab()
            if n % self.stride == 0:
                success, im = cap.retrieve()
                if success:
                    self.img = im
                else:
                    self.img = np.zeros_like(self.img)
                    cap.open(stream)
            time.sleep(wait_ms)
        # break loop
        self.img = None

    def is_frame_ready(self):
        return True if self.img is not None else False

    def get_one_frame(self):
        ret, f = self.cap.read()
        return f

    def get_frame(self):
        # get frame
        if self.realtime is False:
            # if image array empty, wait for image
            while self.img is None:
                return None
        else:   # realtime is true
            if self.img is None:
                return None
        orig_im = self.img.copy()

        # if not realtime, frame reset
        if self.realtime is False:
            self.img = None

        return orig_im

    def show_frame(self, wait_sec:int=0):
        frame = self.get_frame()
        cv2.imshow("frame", frame)
        if cv2.waitKey(wait_sec) == ord('q'):
            if self.logger is not None:
               self.logger.info("-- Quit Show frames")
            raise StopIteration

    def wait_frame(self):
        frame = None
        while frame is None:
            frame = self.get_frame()
        return frame

    def stop(self):
        self.alive = False
        if self.logger is not None:
            self.logger.info("Stop Update thread")
        self.thread.join(timeout=1)

    def pause(self):
        self.bpause = True
        if self.logger is not None:
            self.logger.info("Pause Update thread")

    def is_pause(self):
        return self.bpause

    def restart(self):
        self.bpause = False
        if self.logger is not None:
            self.logger.info("Restart Update thread")

    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys

    s = sys.argv[1]      # video file, webcam, rtsp stream.. etc

    medialoader = MediaLoader(s)
    medialoader.start()
    while medialoader.is_frame_ready() is False:
        time.sleep(0.01)
        continue
    print("-- MediaLoader is ready")
    _frame = medialoader.get_frame()
    print("-- Frame Metadata:", _frame.shape, _frame.dtype)
    t = time.time()
    while True:
        cur_t = time.time()
        if 5 < cur_t - t < 10:
            medialoader.pause()
        else:
            if medialoader.is_pause() is True:
                medialoader.restart()
            else:
                medialoader.show_frame(wait_sec=1)
            time.sleep(0.005)

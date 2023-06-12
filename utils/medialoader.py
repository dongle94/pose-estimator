import os
import cv2
import math
import numpy as np
from pathlib import Path
from threading import Thread
import time

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'       # include image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'   # include video suffixes

def check_sources(s):
    is_file, is_url, is_webcam = False, False, False
    is_file = Path(s).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = s.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    is_webcam = s.isnumeric() or s.endswith('.streams') or (is_url and not is_file)

    return is_file, is_url, is_webcam

class MediaLoader(object):
    def __init__(self, source, save_result=False, save_path="", stride=1):
        self.stride = stride
        self.is_file, self.is_url, self.is_webcam = check_sources(source)

        source = Path(source).read_text().rsplit() if os.path.isfile(source) else source
        self.source = str(source)
        self.img, self.fps, self.frame, self.thread = None, 0, 0, None

        source = eval(source) if source.isnumeric() else source
        cap = cv2.VideoCapture(source)
        assert cap.isOpened(), f"Failed to open {source}"

        # Metadata
        self.cap = cap
        self.w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.fps = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30  # 30 FPS fallback
        self.frame = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')

        _, self.imgs = cap.read()
        self.thread = Thread(target=self.update, args=([cap, source]), daemon=True)
        print(f"-- Success ({self.frame} frames {self.w}x{self.h} at {self.fps:.2f} FPS)")
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
            time.sleep(0.0)

    def is_frame_ready(self):
        return True if self.img is not None else False

    def get_frame(self):
        orig_im = self.img.copy()

        return orig_im

    def show_frame(self, wait_sec:int=0):
        frame = self.get_frame()
        cv2.imshow("frame", frame)
        cv2.waitKey(wait_sec)

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys

    s = sys.argv[1]
    medialoader = MediaLoader(s)
    time.sleep(1)
    _frame = medialoader.get_frame()
    print(_frame.shape, _frame.dtype)

    medialoader.show_frame(0)

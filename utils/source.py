import os
import cv2
import math
import numpy as np
import time
from pathlib import Path
from urllib.parse import urlparse
from threading import Thread

from utils.augmentations import letterbox

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'       # include image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'   # include video suffixes


# source check
def check_sources(s):
    is_file, is_url, is_webcam = False, False, False
    is_file = Path(s).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = s.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    is_webcam = s.isnumeric() or s.endswith('.streams') or (is_url and not is_file)

    return is_file, is_url, is_webcam


class YoloLoadStreams:
    # YOLOv5 streamloader, i.e. `python detect.py --source 'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP streams`
    def __init__(self, sources='file.streams', img_size=640, stride=32, auto=True, transforms=None, vid_stride=1):
        # torch.backends.cudnn.benchmark = True  # faster for fixed-size inference
        # self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride
        self.vid_stride = vid_stride  # video frame-rate stride
        sources = Path(sources).read_text().rsplit() if os.path.isfile(sources) else sources

        # n = len(sources)
        # self.sources = [clean_str(x) for x in sources]  # clean source names for later
        self.sources = sources
        # self.imgs, self.fps, self.frames, self.threads = [None] * n, [0] * n, [0] * n, [None] * n
        self.imgs, self.fps, self.frames, self.threads = None, 0, 0, None

        # Start thread to read frames from video stream
        if urlparse(sources).hostname in ('www.youtube.com', 'youtube.com', 'youtu.be'):  # if source is YouTube video
            # YouTube format i.e. 'https://www.youtube.com/watch?v=Zgi9g1ksQHc' or 'https://youtu.be/Zgi9g1ksQHc'
            # check_requirements(('pafy', 'youtube_dl==2020.12.2'))
            import pafy
            sources = pafy.new(sources).getbest(preftype='mp4').url  # YouTube URL
        sources = eval(sources) if sources.isnumeric() else sources  # i.e. s = '0' local webcam
        cap = cv2.VideoCapture(sources)
        assert cap.isOpened(), f'Failed to open {sources}'

        self.cap = cap
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)  # warning: may return 0 or nan
        self.frames = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')  # infinite stream fallback
        self.fps = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30 # 30 FPS fallback

        _, self.imgs = cap.read()
        self.threads =  Thread(target=self.update, args=([cap, sources]), daemon=True)
        print(f"-- Success ({self.frames} frames {w}x{h} at {self.fps:.2f} FPS)")
        self.threads.start()

        s = np.stack([letterbox(self.imgs, img_size, stride=stride, auto=auto)[0].shape])
        self.rect = np.unique(s, axis=0).shape[0] == 1
        self.auto = auto and self.rect
        self.transforms = transforms
        if not self.rect:
            print("-- WARNING Stream shapes differ.")
        # for i, s in enumerate(sources):  # index, source
        #     # Start thread to read frames from video stream
        #     st = f'{i + 1}/{n}: {s}... '
        #     if urlparse(s).hostname in ('www.youtube.com', 'youtube.com', 'youtu.be'):  # if source is YouTube video
        #         # YouTube format i.e. 'https://www.youtube.com/watch?v=Zgi9g1ksQHc' or 'https://youtu.be/Zgi9g1ksQHc'
        #         # check_requirements(('pafy', 'youtube_dl==2020.12.2'))
        #         import pafy
        #         s = pafy.new(s).getbest(preftype='mp4').url  # YouTube URL
        #     s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
        #     if s == 0:
        #         assert not is_colab(), '--source 0 webcam unsupported on Colab. Rerun command in a local environment.'
        #         assert not is_kaggle(), '--source 0 webcam unsupported on Kaggle. Rerun command in a local environment.'
        #     cap = cv2.VideoCapture(s)
        #     assert cap.isOpened(), f'{st}Failed to open {s}'
        #     self.cap = cap
        #     w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        #     h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #     fps = cap.get(cv2.CAP_PROP_FPS)  # warning: may return 0 or nan
        #     self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')  # infinite stream fallback
        #     self.fps[i] = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30  # 30 FPS fallback
        #
        #     _, self.imgs[i] = cap.read()  # guarantee first frame
        #     self.threads[i] = Thread(target=self.update, args=([i, cap, s]), daemon=True)
        #     LOGGER.info(f'{st} Success ({self.frames[i]} frames {w}x{h} at {self.fps[i]:.2f} FPS)')
        #     self.threads[i].start()
        # LOGGER.info('')  # newline
        #
        # # check for common shapes
        # s = np.stack([letterbox(x, img_size, stride=stride, auto=auto)[0].shape for x in self.imgs])
        # self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        # self.auto = auto and self.rect
        # self.transforms = transforms  # optional
        # if not self.rect:
        #     LOGGER.warning('WARNING ⚠️ Stream shapes differ. For optimal performance supply similarly-shaped streams.')

    def update(self, cap, stream):
        n, f = 0, self.frames
        while cap.isOpened() and n < f:
            n += 1
            cap.grab()
            if n % self.vid_stride == 0:
                success, im = cap.retrieve()
                if success:
                    self.imgs = im
                else:
                    self.imgs = np.zeros_like(self.imgs)
                    cap.open(stream)
            time.sleep(0.0)
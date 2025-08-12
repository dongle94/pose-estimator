import os
import sys
import cv2
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
os.chdir(ROOT)

from core.medialoader import load_images, load_video, load_stream


def check_sources(source):
    img_formats = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'
    vid_formats = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv', 'webm'
    is_imgs, is_vid, is_stream = False, False, False
    if os.path.isdir(source) or '*' in source:
        is_imgs = True
    elif os.path.isfile(source) and Path(source).suffix[1:] in img_formats:
        is_imgs = True
    elif os.path.isfile(source) and Path(source).suffix[1:] in vid_formats:
        is_vid = True
    elif source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')):
        is_stream = True
    elif source.isnumeric() or source.endswith('.streams') or source.startswith('/dev'):
        is_stream = True

    return is_imgs, is_vid, is_stream


class MediaLoader(object):
    def __init__(self, source, stride=1, logger=None, realtime=False, opt=None, bgr=True):

        self.stride = stride
        self.logger = logger
        self.realtime = realtime
        self.opt = opt
        self.bgr = bgr

        self.is_imgs, self.is_vid, self.is_stream = check_sources(source)

        if self.is_imgs:
            dataset = load_images.LoadImages(source, bgr=self.bgr, logger=logger)
            # 첫 번째 이미지로부터 크기 정보 획득
            if dataset.num_files > 0:
                # 첫 번째 이미지 로드해서 크기 확인
                first_frame = next(iter(dataset))
                if first_frame is not None:
                    self.height, self.width = first_frame.shape[:2]
                    dataset.__iter__()  # iterator 리셋
                else:
                    # fallback to config or default
                    self.width = getattr(opt, 'media_width', 640) if opt else 640
                    self.height = getattr(opt, 'media_height', 480) if opt else 480
            else:
                self.width = getattr(opt, 'media_width', 640) if opt else 640
                self.height = getattr(opt, 'media_height', 480) if opt else 480
        elif self.is_vid:
            dataset = load_video.LoadVideo(source, stride=self.stride, realtime=self.realtime, bgr=self.bgr,
                                           logger=logger)
            self.width, self.height = dataset.w, dataset.h
        elif self.is_stream:
            dataset = load_stream.LoadStream(source, stride=self.stride, opt=self.opt, bgr=self.bgr, logger=logger)
            self.width, self.height = dataset.w, dataset.h
        else:
            raise NotImplementedError(f'Invalid input: {source}')

        self.dataset = dataset

    def get_frame(self):
        im = self.dataset.__next__()
        if im is None:
            raise StopIteration("No more frames available")
        return im

    def show_frame(self, title: str = 'frame', wait_sec: int = 0):
        try:
            frame = self.get_frame()
            if self.bgr is False:
                frame = frame[..., ::-1]
            cv2.imshow(title, frame)
            if cv2.waitKey(wait_sec) == ord('q'):
                if self.logger is not None:
                    self.logger.info("-- Quit Show frames")
                raise StopIteration
            return frame
        except StopIteration:
            if self.logger is not None:
                self.logger.info("-- No more frames to show")
            cv2.destroyAllWindows()
            raise

    def __del__(self):
        if hasattr(self, 'dataset'):
            del self.dataset


if __name__ == "__main__":
    from utils.logger import init_logger, get_logger
    from utils.config import set_config, get_config

    # s = sys.argv[1]      # video file, webcam, rtsp stream... 0etc
    set_config('./configs/config.yaml')
    _cfg = get_config()

    init_logger(_cfg)
    _logger = get_logger()

    _media_loader = MediaLoader(_cfg.media_source,
                                logger=_logger,
                                realtime=_cfg.media_realtime,
                                bgr=_cfg.media_bgr,
                                opt=_cfg)
    print(f"-- Frame Metadata: {_media_loader.width}x{_media_loader.height}, FPS: {_media_loader.dataset.fps}")
    print("-- MediaLoader is ready")

    _title = 'frame'
    wt = int((0 if _media_loader.is_imgs else 1 / _media_loader.dataset.fps) * 1000)
    try: 
        while True:
            _frame = _media_loader.show_frame(title=_title, wait_sec=wt)
    except StopIteration:
        print("-- MediaLoader finished")
    except KeyboardInterrupt:
        print("-- MediaLoader interrupted by user")
    finally:
        cv2.destroyAllWindows()

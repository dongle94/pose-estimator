import os
import sys
import cv2
import time
import numpy as np
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
os.chdir(ROOT)

from core.media_loader import MediaLoader
from core.obj_detector import ObjectDetector
from core.pose_estimator import PoseEstimator
from utils.logger import get_logger, init_logger
from utils.config import set_config, get_config
from utils.visualization import vis_pose_result


def main():
    set_config('./configs/config.yaml')
    _cfg = get_config()

    init_logger(_cfg)
    _logger = get_logger()

    _detector = ObjectDetector(cfg=_cfg)
    _estimator = PoseEstimator(cfg=_cfg)

    _bgr = getattr(_cfg, 'media_bgr', True)
    _realtime = getattr(_cfg, 'media_realtime', False)
    media_loader = MediaLoader(_cfg.media_source,
                               logger=_logger,
                               realtime=_realtime,
                               bgr=_bgr,
                               opt=_cfg)
    wt = 0 if media_loader.is_imgs else 1 / media_loader.dataset.fps

    sign_points = []
    m_w, m_h = _cfg.media_width, _cfg.media_height
    roi_x1, roi_y1 = int(m_w * 0.25), int(m_h * 0.25)
    roi_x2, roi_y2 = int(m_w * 0.75), int(m_h * 0.75)

    while True:
        st = time.time()
        frame = media_loader.get_frame()
        cv2.flip(frame, 1, frame)

        _det = _detector.run(frame)

        _kept_preds = None
        if len(_det):
            _kept_preds, _heatmaps = _estimator.run(frame, _det)

        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (50, 50, 180), thickness=2)

        valid_boxes = []
        for idx, d in enumerate(_det):
            x1, y1, x2, y2 = map(int, d[:4])
            cls = int(d[5])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (96, 96, 216), thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(frame, str(_detector.names[cls]), (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (96, 96, 96), thickness=1, lineType=cv2.LINE_AA)
            if cls == 0:
                valid_boxes.append(idx)

        valid_idx = -1
        if _kept_preds is not None and valid_boxes:
            area = 0
            for idx in valid_boxes:
                hand_kept = _kept_preds[idx]
                det = _det[idx]
                x1, y1, x2, y2 = map(int, det[:4])
                bench_x, bench_y = hand_kept[8][:2]
                if (roi_x1 <= bench_x <= roi_x2 and roi_y1 <= bench_y <= roi_y2) and area < (x2 - x1) * (y2 - y1):
                    area = (x2 - x1) * (y2 - y1)
                    valid_idx = idx

        if valid_idx != -1:
            kept_preds = np.expand_dims(np.array(_kept_preds)[valid_idx], axis=0)
            frame = vis_pose_result(frame,
                                    pred_kepts=kept_preds,
                                    model=_estimator.estimator.dataset)
            x, y, score = _kept_preds[valid_idx][8]
            if score > 0.5:
                sign_points.append([int(x), int(y)])

        for idx, pt in enumerate(sign_points):
            cv2.circle(frame, (pt[0], pt[1]), 1, (0, 255, 0), -1)
            if idx > 0 and len(sign_points):
                cv2.line(frame,
                         (sign_points[idx-1][0], sign_points[idx-1][1]),
                         (pt[0], pt[1]),
                         (255, 0, 0))

        et = time.time()
        if media_loader.is_imgs:
            t = 0
        else:
            if et - st < wt:
                t = int((wt - (et - st)) * 1000)
            else:
                t = 1

        cv2.imshow('_', frame)
        key = cv2.waitKey(t)
        if key == ord('q'):
            print("-- CV2 Stop --")
            break
        elif key == ord(' '):
            sign_points = []
        elif key == ord('s'):
            bg = np.full(frame.shape, 255, dtype=np.uint8)
            cv2.rectangle(bg, (roi_x1, roi_y1), (roi_x2, roi_y2), (50, 50, 180), thickness=2)
            for idx, pt in enumerate(sign_points):
                cv2.circle(bg, (pt[0], pt[1]), 1, (0, 255, 0), -1)
                if idx > 0 and len(sign_points):
                    cv2.line(bg,
                             (sign_points[idx - 1][0], sign_points[idx - 1][1]),
                             (pt[0], pt[1]),
                             (255, 0, 0))
            cv2.imwrite('./sign.jpg', bg)

    print("-- Stop program --")


if __name__ == "__main__":
    main()

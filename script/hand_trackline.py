import argparse
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


def main(opt):
    set_config('./configs/config.yaml')
    cfg = get_config()

    init_logger(cfg)
    logger = get_logger()

    detector = ObjectDetector(cfg=cfg)
    estimator = PoseEstimator(cfg=cfg)

    bgr = getattr(cfg, 'media_bgr', True)
    realtime = getattr(cfg, 'media_realtime', False)
    media_loader = MediaLoader(cfg.media_source,
                               logger=logger,
                               realtime=realtime,
                               bgr=bgr,
                               opt=cfg)
    wt = 0 if media_loader.is_imgs else 1 / media_loader.dataset.fps

    sign_points = []
    m_w, m_h = media_loader.width, media_loader.height
    roi_x1, roi_y1 = int(m_w * 0.25), int(m_h * 0.25)
    roi_x2, roi_y2 = int(m_w * 0.75), int(m_h * 0.75)

    if opt.save_path:
        os.makedirs(opt.save_path, exist_ok=True)

    frame_idx = 0
    while True:
        st = time.time()
        frame_idx += 1
        frame = media_loader.get_frame()
        cv2.flip(frame, 1, frame)

        det = detector.run(frame)
        _kept_preds = None
        if len(det):
            _kept_preds, _heatmaps = estimator.run(frame, det)

        if not opt.show_all:
            bg_frame = np.full(frame.shape, 255, dtype=np.uint8)
            frame = cv2.addWeighted(frame, 0.1, bg_frame, 0.9, 0)
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (50, 50, 180), thickness=2)

        # Select Right hand
        valid_hand_boxes = []
        for idx, d in enumerate(det):
            x1, y1, x2, y2 = map(int, d[:4])
            cls = int(d[5])
            if cls == 0:
                valid_hand_boxes.append(idx)
            if opt.show_all:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (96, 96, 216), thickness=2, lineType=cv2.LINE_AA)
                cv2.putText(frame, str(detector.names[cls]), (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (96, 96, 96), thickness=1, lineType=cv2.LINE_AA)

        # Select index finger in valid hands
        valid_hand_kept_idx = -1
        if valid_hand_boxes and _kept_preds is not None:
            area = 0
            for idx in valid_hand_boxes:
                hand_kept = _kept_preds[idx]
                d = det[idx]
                x1, y1, x2, y2 = map(int, d[:4])
                bench_x, bench_y = hand_kept[8][:2]
                if (roi_x1 <= bench_x <= roi_x2 and roi_y1 <= bench_y <= roi_y2) and area < (x2 - x1) * (y2 - y1):
                    area = (x2 - x1) * (y2 - y1)
                    valid_hand_kept_idx = idx

        if valid_hand_kept_idx != -1:
            bench_y_0 = _kept_preds[valid_hand_kept_idx][18][1]
            bench_y_1 = _kept_preds[valid_hand_kept_idx][14][1]
            bench_y_2 = _kept_preds[valid_hand_kept_idx][10][1]
            x, y, score = _kept_preds[valid_hand_kept_idx][8]       # 8: index_finger tip
            if score > opt.confidence and y <= bench_y_0 and y <= bench_y_1 and y <= bench_y_2:
                sign_points.append([int(x), int(y)])
            if opt.show_all:
                kept_preds = np.expand_dims(np.array(_kept_preds)[valid_hand_kept_idx], axis=0)
                frame = vis_pose_result(frame,
                                        pred_kepts=kept_preds,
                                        model=estimator.estimator.dataset)

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

        if opt.save_path:
            cv2.imwrite(os.path.join(opt.save_path, str(f"{frame_idx:04d}") + '.jpg'), frame)
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
        elif key == ord('c'):
            opt.show_all = not opt.show_all

    print("-- Stop program --")


def args_parse():
    parser = argparse.ArgumentParser(description="Hand finger sign script")
    parser.add_argument('-c', '--confidence', default=0.5, type=float,
                        help='hand keypoint score threshold')
    parser.add_argument('-s', '--save_path', default="", type=str,
                        help='if it is not empty string, save sequence image in save path')
    parser.add_argument('--show_all', action='store_true')
    _args = parser.parse_args()
    return _args


if __name__ == "__main__":
    args = args_parse()
    main(args)

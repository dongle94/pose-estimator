import argparse
import os
import sys
import cv2
import time
import math
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
    roi_x1, roi_y1 = int(m_w * (0.5 - opt.roi_width / 2)), int(m_h * (0.5 - opt.roi_height / 2))
    roi_x2, roi_y2 = int(m_w * (0.5 + opt.roi_width / 2)), int(m_h * (0.5 + opt.roi_height / 2))

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
            hand_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            hand_frame = cv2.resize(hand_frame, (int(frame.shape[1]*0.2), int(frame.shape[0]*0.2)))
            frame = np.full(frame.shape, 255, dtype=np.uint8)
            hf_h, hf_w = hand_frame.shape[:2]
            frame[0:hf_h, m_w-hf_w:m_w] = hand_frame

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
            roi_area = (roi_x2-roi_x1) * (roi_y2-roi_y1)
            bx1, by1, bx2, by2 = det[valid_hand_kept_idx][:4].cpu().numpy()
            hand_area = (bx2-bx1) * (by2-by1)

            ret = is_valid_hand_point(_kept_preds[valid_hand_kept_idx], roi_area, hand_area,
                                      kept_score=opt.confidence, box_thres=opt.bbox_area, angle=opt.angle)

            if ret:
                x, y, score = _kept_preds[valid_hand_kept_idx][8]  # 8: index_finger tip
                sign_points.append([int(x), int(y)])
            if opt.show_all:
                kept_preds = np.expand_dims(np.array(_kept_preds)[valid_hand_kept_idx], axis=0)
                frame = vis_pose_result(frame,
                                        pred_kepts=kept_preds,
                                        model=estimator.estimator.dataset)

        for idx, pt in enumerate(sign_points):
            cv2.circle(frame, (pt[0], pt[1]), 1, (0, 255, 0), -1)
            if idx > 0 and len(sign_points):
                pre_x, pre_y = sign_points[idx - 1][0], sign_points[idx - 1][1]
                cur_x, cur_y = pt[0], pt[1]
                dist = math.dist([pre_x, pre_y], [cur_x, cur_y])
                if dist > 100:
                    thick = 4
                elif dist > 50:
                    thick = 5
                else:
                    thick = 6
                cv2.line(
                    frame,
                    (sign_points[idx-1][0], sign_points[idx-1][1]),
                    (pt[0], pt[1]),
                    (255, 0, 0),
                    thickness=thick
                )

        et = time.time()
        if media_loader.is_imgs:
            t = 0
        else:
            if et - st < wt:
                t = int((wt - (et - st)) * 1000) + 1
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
            cv2.imwrite('./sign.jpg', frame)
        elif key == ord('c'):
            opt.show_all = not opt.show_all

    print("-- Stop program --")


def is_valid_hand_point(hand_point, roi_area, hand_box_area, kept_score=0.3, box_thres=0.05, angle=90):
    ret = True
    x, y, score = hand_point[8]  # 8: index_finger tip

    # Score condition
    if score < kept_score:
        ret = False

    # Area of Box condition
    if round(hand_box_area/roi_area, 4) < box_thres:
        print("area:", round(hand_box_area / roi_area, 4))
        ret = False

    # Angle condition
    a = hand_point[8][:2]
    b = hand_point[6][:2]
    c = hand_point[5][:2]
    ba, bc = a - b, c - b
    dot = np.dot(ba, bc)
    ba_norm, bc_norm = np.linalg.norm(ba), np.linalg.norm(bc)
    radi = np.arccos(np.clip(dot / (ba_norm * bc_norm) + 1e-8, -1.0, 1.0))
    angle = np.abs(radi * 180.0 / np.pi)
    if angle < 90:
        print("angle", angle)
        ret = False

    bench_y_0 = hand_point[18][1]
    bench_y_1 = hand_point[14][1]
    bench_y_2 = hand_point[10][1]
    bench_x = hand_point[18][0]

    return ret


def args_parse():
    parser = argparse.ArgumentParser(description="Hand finger sign script")
    parser.add_argument('-s', '--save_path', default="", type=str,
                        help='if it is not empty string, save sequence image in save path')
    parser.add_argument('--show_all', action='store_true')

    parser.add_argument('--roi_width', default=0.7, type=float)
    parser.add_argument('--roi_height', default=0.5, type=float)
    parser.add_argument('-c', '--confidence', default=0.5, type=float,
                        help='hand keypoint score threshold')
    parser.add_argument('-b', '--bbox_area', default=0.02, type=float)
    parser.add_argument('--angle', default=90, type=int)
    _args = parser.parse_args()
    return _args


if __name__ == "__main__":
    args = args_parse()
    main(args)

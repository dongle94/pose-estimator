import os
import sys
import cv2
import csv
import time
import argparse
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
os.chdir(ROOT)

from utils.config import _C as cfg
from utils.config import update_config
from utils.logger import get_logger, init_logger

from utils.medialoader import MediaLoader
from core.obj_detectors import ObjectDetector
from core.pose_estimator import PoseDetector



def main(opt):
    logger = get_logger()
    logger.info(f"Start dv6 annotation script.")
    inputs = opt.inputs

    # init model
    obj_detector = ObjectDetector(cfg=cfg)
    kept_detector = PoseDetector(cfg=cfg)

    # create csv
    if not os.path.exists(os.path.join(os.getcwd(), 'result')):
        os.makedirs(os.path.join(os.getcwd(), 'result'))
    csv_file = open(os.path.join(os.getcwd(), 'result', 'dv63.csv'), 'a+', newline='')
    cw = csv.writer(csv_file)

    # inputs analysis loop
    for input_video in inputs:
        media_loader = MediaLoader(input_video,
                                   logger=logger,
                                   realtime=False,
                                   fast=True)
        media_loader.start()

        while media_loader.is_frame_ready() is False:
            time.sleep(0.01)
            continue

        f_cnt = 0
        ts = [0., 0., 0.]
        while True:
            frame = media_loader.get_frame()

            if frame is None:
                logger.info("Frame is None -- Break main loop")
                break

            # Human Detection
            t0 = time.time()
            im = obj_detector.preprocess(frame)
            pred = obj_detector.detect(im)
            pred, det = obj_detector.postprocess(pred)
            t1 = time.time()

            # Pose Detection
            if len(det):
                inps, centers, scales = kept_detector.preprocess(frame, det)
                preds = kept_detector.detect(inps)
                rets, _ = kept_detector.postprocess(preds, centers, scales)
            else:
                rets = None
                heatmap = None
            t2 = time.time()

            # Show Processed Videos
            for d in det:
                x1, y1, x2, y2 = map(int, d[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (96, 96, 216), thickness=2, lineType=cv2.LINE_AA)

            if len(det) == len(rets):
                for i, ret in enumerate(rets):
                    _ret = [r for idx, r in enumerate(ret) if idx in [6, 7, 8, 9, 12, 13]]
                    x1, y1, x2, y2 = map(int, det[i][:4])
                    w, h = x2 - x1, y2 - y1
                    dv6_list = []
                    for r in _ret:
                        x, y = int(r[0]), int(r[1])
                        cv2.circle(frame, (x, y), 3, (16, 16, 240), -1)
                        ax, ay = x - x1, y-y1
                        rx, ry = format(ax / w, '.6f'), format(ay / h, '.6f')
                        dv6_list.append(rx)
                        dv6_list.append(ry)

                    dv6_list.append(opt.label)
                    cw.writerow(dv6_list)
                    dv6_list.clear()

            cv2.imshow('_', frame)
            t3 = time.time()

            ts[0] += (t1 - t0)
            ts[1] += (t2 - t1)
            ts[2] += (t3 - t2)

            if cv2.waitKey(1) == ord('q'):
                logger.info("-- CV2 Stop by Keyboard Input --")
                break

            f_cnt += 1
            if f_cnt % cfg.CONSOLE_LOG_INTERVAL == 0:
                logger.debug(
                    f"{f_cnt} Frame - obj_det: {ts[0] / f_cnt:.4f} / "
                    f"kept_det: {ts[1] / f_cnt:.4f} / "
                    f"vis: {ts[2] / f_cnt:.4f}")

        media_loader.stop()
    csv_file.close()
    logger.info("-- Stop program --")


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputs', required=True, nargs='+',
                        help='input video files')
    parser.add_argument('-l', '--label', default=0, type=int,
                        help='classification label')
    parser.add_argument('-c', '--config', default='./configs/dv6_annotate.yaml',
                        help="annotation configuration yaml file path")
    _args = parser.parse_args()
    return _args


if __name__ == "__main__":
    args = args_parse()

    update_config(cfg, args=args.config)
    init_logger(cfg)

    main(args)

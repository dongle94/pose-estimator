import os
import argparse
import copy
import numpy as np
import cv2
import time
from datetime import datetime
# from pathlib import Path

from utils.config import _C as cfg
from utils.config import update_config
from utils.medialoader import MediaLoader
from detectors.obj_detector import HumanDetector
from detectors.pose_detector import PoseDetector
from utils.logger import init_logger, get_logger
from utils.visualization import get_heatmaps, merge_heatmaps, vis_pose_result


from utils.source import check_sources, YoloLoadStreams


class PoseEstimator(object):
    def __init__(self, cfg_path, source):
        _cfg = copy.deepcopy(cfg)
        update_config(_cfg, cfg_path)

        init_logger(_cfg)
        self.logger = get_logger()

        self.data_loader = MediaLoader(source)

        self.obj_detector = HumanDetector(cfg=_cfg)
        self.pose_detector = PoseDetector(cfg=_cfg)

    def run(self, input_frame=None, heatmap=False):
        # get input
        if input_frame is None:
            input_frame = self.data_loader.get_frame()

        keys_preds = np.array([])
        raw_heatmaps = np.array([])

        # human detection inference
        _input = self.obj_detector.preprocess(input_frame)
        obj_preds = self.obj_detector.detect(_input)
        obj_preds, obj_dets = self.obj_detector.postprocess(obj_preds)

        # pose keypoints inference
        if len(obj_dets):
            inps, centers, scales = self.pose_detector.preprocess(input_frame, obj_dets)
            keys_rets = self.pose_detector.detect(inps)
            keys_preds, raw_heatmaps = self.pose_detector.postprocess(keys_rets, centers, scales)

        _heatmap = None
        if heatmap is True:
            heatmaps = get_heatmaps(raw_heatmaps, colormap=None, draw_index=None)
            _heatmap = merge_heatmaps(heatmaps, obj_dets, input_frame.shape)

        return obj_dets, keys_preds, _heatmap

    def visualize(self, box=False, keypoint=False, heatmap=False):
        im0 = self.data_loader.get_frame()
        boxes, keypoints, heatmaps = self.run(input_frame=im0, heatmap=True)

        if box is True:
            for b in boxes:
                x1, y1, x2, y2 = map(int, b[:4])
                cv2.rectangle(im0, (x1, y1), (x2, y2), (96, 96, 216), thickness=2, lineType=cv2.LINE_AA)
        if keypoint is True:
            if keypoints is not None:
                im0 = vis_pose_result(model=None, img=im0, result=keypoints)

        if heatmap is True:
            if heatmaps is not None:
                if len(heatmaps.shape) == 2:
                    heatmaps = np.uint8(255 * heatmaps)
                    heatmaps = cv2.applyColorMap(heatmaps, cv2.COLORMAP_HOT)
                im0 = cv2.add((0.4 * heatmaps).astype(np.uint8), im0)

        return im0


def main(args):
    pose_estimator = PoseEstimator(args.config, args.source)

    while pose_estimator.data_loader.img is None:
        time.sleep(0.001)
        continue

    video_writer = None
    if args.save:
        source_name = os.path.splitext(os.path.basename(args.source))[0]
        filename = f'./runs/{source_name}_{datetime.now().strftime("%H_%M_%S")}.mp4'
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        FPS = pose_estimator.data_loader.cap.get(cv2.CAP_PROP_FPS)
        W = int(pose_estimator.data_loader.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(pose_estimator.data_loader.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_writer = cv2.VideoWriter(filename,
                                       cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                       FPS,
                                       (W, H))

    while pose_estimator.data_loader.cap.isOpened():
        frame = pose_estimator.visualize(keypoint=True, heatmap=True)

        if args.save and video_writer is not None:
            video_writer.write(frame)

        cv2.imshow("result", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()



def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='configuration')
    parser.add_argument('-s', '--source', type=str, help='file/URL(RTSP)/0(webcam)')
    parser.add_argument("--save", action="store_true")
    _args = parser.parse_args()
    return _args


if __name__ == "__main__":
    args = args_parse()

    main(args)

import os
import argparse
import copy
import numpy as np
import cv2
import time
from datetime import datetime

from utils.config import _C as cfg
from utils.config import update_config
from utils.logger import init_logger, get_logger

from utils.medialoader import MediaLoader
from core.obj_detectors import ObjectDetector
from core.pose_estimator import PoseDetector
from utils.visualization import get_heatmaps, merge_heatmaps, vis_pose_result, get_rule_heatmaps
from utils.coordinates import get_angle


class PoseEstimator(object):
    def __init__(self, _cfg, source):
        self.logger = get_logger()

        self.data_loader = MediaLoader(source)

        self.obj_detector = ObjectDetector(cfg=_cfg)
        self.pose_detector = PoseDetector(cfg=_cfg)

        # debug log variable
        self.f_cnt = 0
        self.kpt_cnt = 0
        self.det_times = 0.
        self.kpt_times = 0.
        self.rule_times = 0.
        self.heatmap_times = 0.
        self.log_interval = _cfg.CONSOLE_LOG_INTERVAL

    def run(self, input_frame=None, heatmap=False, draw_index=[], color_map=None, rule=False):
        # get input
        if input_frame is None:
            input_frame = self.data_loader.get_frame()

        keys_preds = np.array([])
        raw_heatmaps = np.array([])

        self.f_cnt += 1

        # human detection inference
        st = time.time()

        _input = self.obj_detector.preprocess(input_frame)
        obj_preds = self.obj_detector.detect(_input)
        obj_preds, obj_dets = self.obj_detector.postprocess(obj_preds)

        et = time.time()
        self.det_times += et - st


        # pose keypoints inference
        if len(obj_dets):
            self.kpt_cnt += 1
            st = time.time()

            inps, centers, scales = self.pose_detector.preprocess(input_frame, obj_dets)
            keys_rets = self.pose_detector.detect(inps)
            keys_preds, raw_heatmaps = self.pose_detector.postprocess(keys_rets, centers, scales)

            et = time.time()
            self.kpt_times += et - st

        # Process Rule
        if rule is True and len(keys_preds):
            st = time.time()

            color_map = {}
            for batch in range(len(keys_preds)):
                color_map[batch] = {}
                for idx in draw_index:
                    color_map[batch][idx] = cv2.COLORMAP_OCEAN
            for batch, keys_pred in enumerate(keys_preds):
                # 5,7,9 - right side / 6,8,10 - left side
                for i in [7, 8]:
                    angle = get_angle(keys_pred[i-2][:2], keys_pred[i][:2], keys_pred[i+2][:2])
                    try:
                        if 0 < angle < 90:
                            color_map[batch][i] = cv2.COLORMAP_DEEPGREEN    #cv2.COLORMAP_OCEAN

                        else:
                            color_map[batch][i] = cv2.COLORMAP_HOT
                    except:
                        color_map[batch][i] = cv2.COLORMAP_OCEAN

            et = time.time()
            self.rule_times += et - st

        _heatmap = None
        if heatmap is True:
            st = time.time()

            if rule is True:
                heatmaps = get_rule_heatmaps(raw_heatmaps, colormap=color_map, draw_index=draw_index)
            else:
                heatmaps = get_heatmaps(raw_heatmaps, colormap=color_map, draw_index=draw_index)
            _heatmap = merge_heatmaps(heatmaps, obj_dets, input_frame.shape)

            et = time.time()
            self.heatmap_times += et - st

        # Logging
        if self.f_cnt % self.log_interval == 0:
            self.logger.debug(f"{self.f_cnt} Frames: det - {self.det_times / self.f_cnt:.4f} sec / "
                              f"kpt - {self.kpt_times / self.f_cnt:.4} sec / "
                              f"rule - {self.rule_times / self.f_cnt:.4f} sec / "
                              f"heatmap - {self.heatmap_times / self.f_cnt:.4f} sec")


        return obj_dets, keys_preds, _heatmap

    def visualize(self, box=False, keypoint=False, heatmap=False, draw_index=[], color_map=None, rule=False):
        im0 = self.data_loader.get_frame()
        boxes, keypoints, heatmaps = self.run(input_frame=im0, heatmap=heatmap, draw_index=draw_index,
                                              color_map=color_map, rule=rule)
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
                im0 = cv2.add((0.7 * heatmaps).astype(np.uint8), im0)

        return im0


def main(args):
    pose_estimator = PoseEstimator(_cfg=_cfg, source=args.source)

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
        frame = pose_estimator.visualize(box=True, keypoint=True, heatmap=True, draw_index=[7, 8],
                                         color_map=None, rule=True)
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

    _cfg = copy.deepcopy(cfg)
    update_config(_cfg, args.config)

    init_logger(_cfg)
    Logger = get_logger()

    main(args)

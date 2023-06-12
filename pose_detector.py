import argparse
import time
import os
import cv2
from pathlib import Path

from utils.config import _C as cfg
from utils.config import update_config
from utils.source import check_sources, YoloLoadStreams

def main():

    source = str(args.source)
    source_name = source
    is_file, is_url, is_webcam = check_sources(source)
    print(f"{source=} / {is_file=}, {is_url=}, {is_webcam=}")

    # dataloader
    save_result = not args.nosave
    if is_file:
        source_name = os.path.splitext(os.path.basename(source))[0]
    elif is_url:
        pass
    elif is_webcam:
        dataset = YoloLoadStreams(source)
    else:
        assert print("-- No Valid Sources --")

    # Video Writer
    if save_result:
        write_width = int(dataset.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        write_height = int(dataset.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if dataset.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) > 1080:
            write_width = int(write_width / 2)
            write_height = int(write_height / 2)
        save_path = './runs/result.mp4'
        vid_path, vid_writer = None, None

    # Run Inference
    for path, im, im0s, vid_cap, s in dataset:
        if save_result:
            # is image

            # is video or is stream
            if is_file or is_webcam:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()
                    if vid_cap:
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = write_width
                        h = write_height
                    else:
                        fps, w, h = 30, im0s.shape[1], im0s.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0s)


        cv2.imshow('-', im0s)


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='configuration')
    parser.add_argument('--source', type=str, help='file/URL(RTSP)/0(webcam)')
    parser.add_argument('--nosave')
    args =  parser.parse_args()
    return args


if __name__ == "__main__":
    args = args_parse()
    update_config(cfg, args.config)
    print(f"{cfg=}")

    main()

import os

from yacs.config import CfgNode as CN


_C = CN()

# Envrionments
_C.DEVICE = None

# Media
_C.MEDIA_SOURCE = "0"
_C.MEDIA_OPT_AUTO = True
_C.MEDIA_WIDTH = 1280
_C.MEDIA_HEIGHT = 720
_C.MEDIA_FPS = 30

# Object Detector
_C.IMG_SIZE = 640
_C.DET_MODEL_TYPE = ""
_C.DET_MODEL_PATH = ""
_C.HALF = False
_C.OBJ_CLASSES = None
_C.CONF_THRES = 0.5
_C.NMS_IOU = 0.45
_C.AGNOSTIC_NMS = True
_C.MAX_DET = 100


# Keypoint Detector
_C.KEPT_MODEL_TYPE = ""
_C.KEPT_MODEL_PATH = ""
_C.KEPT_MODEL_CNF = ""
_C.KEPT_HALF = False
_C.KEPT_IMG_SIZE = [288, 384]


# Logger
_C.LOG_LEVEL = 'DEBUG'
_C.CONSOLE_LOG = False
_C.CONSOLE_LOG_INTERVAL = 10
_C.LOGGER_NAME = ""
_C.FILE_LOG = False
_C.LOG_FILE_DIR = './log/'
_C.LOG_FILE_SIZE = 100
_C.LOG_FILE_COUNTER = 10
_C.LOG_FILE_ROTATE_TIME = "D"
_C.LOG_FILE_ROTATE_INTERVAL = 1




def update_config(cfg, args):
    if not args:
        print("-- No exist Config File --")
        return

    cfg.defrost()
    cfg.merge_from_file(args)
    # cfg.merge_from_list(args.opts)

    cfg.freeze()


if __name__ ==  "__main__":
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)
import os

from yacs.config import CfgNode as CN


_C = CN()

# Envrionments
_C.DEVICE = None

# Object Detector
_C.IMG_SIZE = 640
_C.DET_MODEL_TYPE = ""
_C.DET_MODEL_PATH = ""
_C.HALF = False


# Keypoint Detector
_C.KEPT_MODEL_TYPE = ""
_C.KEPT_MODEL_PATH = ""
_C.KEPT_HALF = False
_C.KEPT_IMG_SIZE = [288, 384]



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
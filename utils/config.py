import os

from yacs.config import CfgNode as CN


_C = CN()

_C.IMG_SIZE = 640


def update_config(cfg, args):
    if not args:
        print("-- No exist Config File --")
        return

    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.freeze()


if __name__ ==  "__main__":
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)
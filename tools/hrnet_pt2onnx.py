import os
import sys
import argparse
from pathlib import Path

import torch
import onnx

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
os.chdir(ROOT)

from core.hrnet_pose.hrnet_utils.torch_utils import select_device
from core.hrnet_pose.models.pose_hrnet import PoseHighResolutionNet


def get_cfg_file(img_size, channel, dataset):
    if img_size is None:
        img_size = [256, 192]
    cfg_file = f"{dataset}_w{channel}_{img_size[0]}x{img_size[1]}.yaml"
    cfg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../core/hrnet_pose/cfg', cfg_file))

    return cfg_path


def convert(opt):
    file = Path(opt.weight)

    # Device and Half Check
    device = select_device(device=opt.device, gpu_num=opt.gpu_num)
    fp16 = opt.fp16
    if fp16:
        assert device.type != 'cpu', '--fp16 only compatible with GPU export, i.e. use --device cuda'

    # Model load
    weight_cfg = get_cfg_file(opt.imgsz, opt.channel, opt.dataset)
    from core.hrnet_pose.cfg.default import _C as pose_cfg
    pose_cfg.defrost()
    pose_cfg.merge_from_file(weight_cfg)
    pose_cfg.freeze()
    model = PoseHighResolutionNet(cfg=pose_cfg)
    model.load_state_dict(torch.load(str(file), map_location=device), strict=False)

    # Input Check
    im = torch.zeros((1, 3, *opt.imgsz)).to(device)

    model.to(device)
    model.eval()

    y = None
    for _ in range(2):
        y = model(im)  # dry runs

    data_format = 'fp32'
    if fp16:
        print("Apply half: --fp16")
        im, model = im.half(), model.half()
        data_format = 'fp16'
    shape = tuple((y[0] if isinstance(y, tuple) else y).shape)
    print(f"Converting from {file} with output shape {shape}")

    f = str(file.with_suffix('.onnx'))
    f = os.path.splitext(f)[0] + f"-{data_format}" + os.path.splitext(f)[1]
    torch.onnx.export(
        model=model,
        args=im,
        f=f,
        verbose=False,
        opset_version=opt.opset,
        do_constant_folding=True,
        input_names=['images'],
        output_names=['outputs'],
        dynamic_axes=None
    )

    # Checks
    model_onnx = onnx.load(f)  # load onnx model
    onnx.checker.check_model(model_onnx)

    print(f"Converting and Saving success: {f}")

    # Simplify
    if opt.simplify:
        try:
            import onnxsim

            model_onnx, check = onnxsim.simplify(model_onnx)
            assert check, "Simplified ONNX model could not be validated"
            onnx.save(model_onnx, f)
            print(f"Simplifying and Saving success: {f}")
        except Exception as e:
            print(f"simplifier failure: {e}")


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weight', required=True, help=".pt or .pth file path")
    parser.add_argument('-d', '--device', default='cpu', help='cpu or cuda')
    parser.add_argument('--gpu_num', default=0, type=int, help='0, 1, 2,...')
    parser.add_argument("--imgsz", type=int, nargs="+", default=[256, 192], help="image (h, w)")
    parser.add_argument("--channel", type=int, default=32, help="hrnet width channels")
    parser.add_argument('--simplify', action='store_true', help='ONNX: simplify model')
    parser.add_argument('--opset', type=int, default=17, help='ONNX: opset version')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument("--dataset", type=str, default="coco", help="coco or mpii, etc..")
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parse()
    convert(args)

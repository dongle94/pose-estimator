import os
import sys
import argparse

import torch
import onnx

from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
os.chdir(ROOT)

from core.hrnet_pose.hrnet_utils.torch_utils import select_device
from core.hrnet_pose.hrnet_pose_pt import PoseHRNetTorch


def convert(opt):
    file = Path(opt.weight)

    # Device and Half Check
    device = select_device(device=opt.device, gpu_num=opt.gpu_num)
    fp16 = opt.fp16
    if fp16:
        assert device.type != 'cpu', '--half only compatible with GPU export, i.e. use --device cuda'
    hrnet = PoseHRNetTorch(
        weight=str(file),
        device=opt.device,
        channel=opt.channel,
        img_size=opt.imgsz,
        gpu_num=opt.gpu_num,
        fp16=opt.fp16
    )
    model = hrnet.model

    im = torch.zeros((1, 3, *opt.imgsz), dtype=torch.half if opt.fp16 else torch.float).to(device)

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
    print(f"Converting and Saving success: {f}")

    # Checks
    model_onnx = onnx.load(f)  # load onnx model
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
    parser.add_argument('--opset', type=int, default=18, help='ONNX: opset version')
    parser.add_argument('--fp16', action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parse()
    convert(args)
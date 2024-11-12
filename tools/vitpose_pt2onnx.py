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

from core.vitpose.models.model import ViTPose
from core.vitpose.vit_utils.util import dyn_model_import
from utils.torch_utils import select_device


def convert(opt):
    device = select_device(device=opt.device, gpu_num=opt.gpu_num)
    fp16 = opt.fp16
    model_cfg = dyn_model_import(opt.dataset, opt.model_name)

    C, H, W = (3, 256, 192)
    model = ViTPose(model_cfg)

    ckpt = torch.load(opt.weight, map_location='cpu')
    if 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()

    im = torch.randn(1, C, H, W).to(device)
    data_format = 'fp32'
    if fp16:
        im, model = im.half(), model.half()
        data_format = 'fp16'

    dynamic_axes = {'input_0': {0: 'batch_size'},
                    'output_0': {0: 'batch_size'}}

    out_name = os.path.splitext(opt.weight)
    out_name = out_name[0] + '-' + data_format + '.onnx'

    torch.onnx.export(
        model=model,
        args=im,
        f=out_name,
        export_params=True,
        verbose=False,
        input_names=["input_0"],
        output_names=["output_0"],
        dynamic_axes=dynamic_axes
    )

    print(f"Converting and Saving success: {out_name}")

    # Checks
    model_onnx = onnx.load(out_name)  # load onnx model
    # Simplify
    if opt.simplify:
        try:
            import onnxsim

            model_onnx, check = onnxsim.simplify(model_onnx)
            assert check, "Simplified ONNX model could not be validated"
            onnx.save(model_onnx, out_name)
            print(f"Simplifying and Saving success: {out_name}")
        except Exception as e:
            print(f"simplifier failure: {e}")


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weight', required=True, help=".pt or .pth file path")
    parser.add_argument('-n', '--model_name', required=True, choices=['s', 'b', 'l', 'h'],
                        help="ViTPose model name [s: ViT-S, b: ViT-B, l: ViT-L, h: ViT-H]")
    parser.add_argument('--dataset', required=True, choices=['coco'],
                        help="ViTPose model dataset format [coco]")
    parser.add_argument('-d', '--device', default='cpu', help='cpu or cuda')
    parser.add_argument('--gpu_num', default=0, type=int, help='0, 1, 2,...')
    parser.add_argument('--simplify', action='store_true', help='ONNX: simplify model')
    parser.add_argument('--opset', type=int, default=17, help='ONNX: opset version')
    parser.add_argument('--fp16', action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parse()
    convert(args)

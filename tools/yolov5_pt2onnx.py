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

from core.yolov5.models.experimental import attempt_load
from core.yolov5.yolov5_utils.torch_utils import select_device
from core.yolov5.yolov5_utils.general import check_img_size
from core.yolov5.models.yolo import Detect


def convert(opt):
    file = Path(opt.weight)

    # Device and Half Check
    device = select_device(device=opt.device, gpu_num=opt.gpu_num)
    fp16 = opt.fp16
    if fp16:
        assert device.type != 'cpu', '--half only compatible with GPU export, i.e. use --device cuda'
    model = attempt_load(opt.weight, device=device, inplace=True, fuse=True)  # load FP32 model

    # Input shape Check
    imgsz = opt.imgsz * 2 if len(opt.imgsz) == 1 else opt.imgsz
    gs = int(max(model.stride))  # grid size (max stride)
    imgsz = [check_img_size(x, gs) for x in imgsz]  # verify img_size are gs-multiples
    im = torch.zeros(1, 3, *imgsz).to(device)

    # Update model
    model.eval()
    for k, m in model.named_modules():
        if isinstance(m, Detect):
            m.inplace = False
            m.dynamic = False
            m.export = True

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
    output_names = ['output0']

    torch.onnx.export(
        model=model,
        args=im,
        f=f,
        verbose=False,
        opset_version=opt.opset,
        do_constant_folding=True,
        input_names=['images'],
        output_names=output_names,
        dynamic_axes=None
    )

    # check
    model_onnx = onnx.load(f)   # load onnx model
    onnx.checker.check_model(model_onnx)    # check onnx model

    # metadata
    d = {'stride': int(max(model.stride)), 'names': model.names}
    for k, v in d.items():
        meta = model_onnx.metadata_props.add()
        meta.key, meta.value = k, str(v)
    onnx.save(model_onnx, f)
    print(f"Converting and Saving success: {f}")

    # Simplify
    if opt.simplify:
        try:
            import onnxsim

            model_onnx, check = onnxsim.simplify(model_onnx)
            assert check, "assert check failed"
            onnx.save(model_onnx, f)
            print(f"Simplifying and Saving success: {f}")
        except Exception as e:
            print(f"simplifier failure: {e}")


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weight', required=True, help=".pt or .pth file path")
    parser.add_argument('-d', '--device', default='cpu', help='cpu or cuda')
    parser.add_argument('--gpu_num', default=0, type=int, help='0, 1, 2,...')
    parser.add_argument("--imgsz", type=int, nargs="+", default=[640, 640], help="image (h, w)",
                        choices=[640, 1280, [640, 640], [1280, 1280]])
    parser.add_argument('--simplify', action='store_true', help='ONNX: simplify model')
    parser.add_argument('--opset', type=int, default=18, help='ONNX: opset version')
    parser.add_argument('--fp16', action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parse()
    convert(args)

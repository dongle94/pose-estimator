import os
import sys
import argparse

import tensorrt as trt

from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
os.chdir(ROOT)


def convert(opt):
    onnx_file = Path(opt.weight)
    print(f'starting export ViTPose with TensorRT {trt.__version__}...')
    assert onnx_file.exists(), f'failed to export ONNX file: {onnx_file}'
    f = onnx_file.with_suffix('.engine')  # TensorRT engine file
    logger = trt.Logger(trt.Logger.INFO)
    if opt.verbose:
        logger.min_severity = trt.Logger.Severity.VERBOSE

    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    # config.max_workspace_size = self.args.workspace * 1 << 30
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, opt.workspace << 30)
    profile = builder.create_optimization_profile()

    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)

    if not parser.parse_from_file(str(onnx_file)):
        raise RuntimeError(f'failed to load ONNX file: {onnx_file}')

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    for inp in inputs:
        print(f'Input "{inp.name}" with shape{inp.shape} {inp.dtype}')
        n, c, h, w = inp.shape
        input_min, input_opt, input_max = (1, c, h, w), (1, c, h, w), (1, c, h, w)
        profile.set_shape(network.get_input(0).name, min=input_min, opt=input_opt, max=input_max)
    for out in outputs:
        print(f'Output "{out.name}" with shape{out.shape} {out.dtype}')

    if builder.platform_has_fast_fp16 and opt.fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    config.add_optimization_profile(profile)

    # with builder.build_serialized_network(network, config) as engine_bytes, open(f, 'wb') as t:
    #     t.write(engine_bytes)
    engine_bytes = builder.build_serialized_network(network, config)
    with open(f, 'wb') as t:
        t.write(engine_bytes)
    print(f"Converting and Saving success: {f}")


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weight', required=True, help=".pt or .pth file path")
    parser.add_argument('--gpu_num', default=0, type=int, help='0, 1, 2,...')
    parser.add_argument('--verbose', action='store_true', help='TensorRT: verbose log')
    parser.add_argument('--workspace', type=int, default=4, help='TensorRT: workspace size (GB)')
    parser.add_argument('--fp16', action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parse()
    convert(args)
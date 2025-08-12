import functools
import math
import os
import time
import platform
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from core.yolo import __version__
from core.yolo.util import (
    NUM_THREADS,
    PYTHON_VERSION,
)
from core.yolo.util.checks import check_version

# Version checks (all default to version>=min_version)
TORCH_1_9 = check_version(torch.__version__, "1.9.0")
TORCH_2_0 = check_version(torch.__version__, "2.0.0")


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """Ensure all processes in distributed training wait for the local master (rank 0) to complete a task first."""
    initialized = dist.is_available() and dist.is_initialized()
    use_ids = initialized and dist.get_backend() == "nccl"

    if initialized and local_rank not in {-1, 0}:
        dist.barrier(device_ids=[local_rank]) if use_ids else dist.barrier()
    yield
    if initialized and local_rank == 0:
        dist.barrier(device_ids=[local_rank]) if use_ids else dist.barrier()


def smart_inference_mode():
    """Apply torch.inference_mode() decorator if torch>=1.9.0 else torch.no_grad() decorator."""

    def decorate(fn):
        """Apply appropriate torch decorator for inference mode based on torch version."""
        if TORCH_1_9 and torch.is_inference_mode_enabled():
            return fn  # already in inference_mode, act as a pass-through
        else:
            return (torch.inference_mode if TORCH_1_9 else torch.no_grad)()(fn)

    return decorate


@functools.lru_cache
def get_cpu_info():
    """Return a string with system CPU information, i.e. 'Apple M2'."""
    from core.yolo.util import PERSISTENT_CACHE  # avoid circular import error

    if "cpu_info" not in PERSISTENT_CACHE:
        try:
            import cpuinfo  # pip install py-cpuinfo

            k = "brand_raw", "hardware_raw", "arch_string_raw"  # keys sorted by preference
            info = cpuinfo.get_cpu_info()  # info dict
            string = info.get(k[0] if k[0] in info else k[1] if k[1] in info else k[2], "unknown")
            PERSISTENT_CACHE["cpu_info"] = string.replace("(R)", "").replace("CPU ", "").replace("@ ", "")
        except Exception:
            pass
    return PERSISTENT_CACHE.get("cpu_info", "unknown")


@functools.lru_cache
def get_gpu_info(index):
    """Return a string with system GPU information, i.e. 'Tesla T4, 15102MiB'."""
    properties = torch.cuda.get_device_properties(index)
    return f"{properties.name}, {properties.total_memory / (1 << 20):.0f}MiB"


def select_device(device="", gpu_num=0, logger=None):
    """
    Selects the appropriate PyTorch device based on the provided arguments.

    The function takes a string specifying the device or a torch.device object and returns a torch.device object
    representing the selected device. The function also validates the number of available devices and raises an
    exception if the requested device(s) are not available.

    Args:
        device (str | torch.device, optional): Device string or torch.device object.
            Options are 'None', 'cpu', or 'cuda', or '0' or '0,1,2,3'. Defaults to an empty string, which auto-selects
            the first available GPU, or CPU if no GPU is available.
        gpu_num: (int, optional): Number of GPU devices to use. Defaults to 0
        logger (logging.Logger | None, optional): Logger to use. Defaults to None.

    Returns:
        (torch.device): Selected device.

    Raises:
        ValueError: If the specified device is not available or if the batch size is not a multiple of the number of
            devices when using multiple GPUs.

    Examples:
        >>> select_device('cuda', gpu_num=0)
        device(type='cuda', index=0)

        >>> select_device('cpu')
        device(type='cpu')

    Note:
        Sets the 'CUDA_VISIBLE_DEVICES' environment variable for specifying which GPUs to use.
    """

    if isinstance(device, torch.device) or str(device).startswith("tpu"):
        return device

    s = f"Ultralytics YOLOv{__version__} 🚀 Python-{platform.python_version()} torch-{torch.__version__} "
    device = str(device).lower()
    for remove in "cuda:", "none", "(", ")", "[", "]", "'", " ":
        device = device.replace(remove, "")  # to string, 'cuda:0' -> '0' and '(0, 1)' -> '0,1'
    cpu = device == "cpu"
    mps = device in {"mps", "mps:0"}  # Apple Metal Performance Shaders (MPS)
    if cpu or mps:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        # if device == "cuda":
        #     device = "0"
        # if "," in device:
        #     device = ",".join([x for x in device.split(",") if x])  # remove sequential commas, i.e. "0,,1" -> "0,1"
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        # os.environ["CUDA_VISIBLE_DEVICES"] = device  # set environment variable - must be before assert is_available()
        if not (torch.cuda.is_available() and torch.cuda.device_count() >= len(device.split(","))):
            print(s)
            install = (
                "See https://pytorch.org/get-started/locally/ for up-to-date torch install instructions if no "
                "CUDA devices are seen by torch.\n"
                if torch.cuda.device_count() == 0
                else ""
            )
            raise ValueError(
                f"Invalid CUDA 'device={device}' requested."
                f" Use 'device=cpu' or pass valid CUDA device(s) if available,"
                f" i.e. 'device=0' or 'device=0,1,2,3' for Multi-GPU.\n"
                f"\ntorch.cuda.is_available(): {torch.cuda.is_available()}"
                f"\ntorch.cuda.device_count(): {torch.cuda.device_count()}"
                f"\nos.environ['CUDA_VISIBLE_DEVICES']: {visible}\n"
                f"{install}"
            )

    if not cpu and not mps and torch.cuda.is_available():  # prefer GPU if available
        devices = device.split(",") if device else "0"  # i.e. "0,1" -> ["0", "1"]
        # n = len(devices)  # device count
        # if n > 1:  # multi-GPU
        #     if batch < 1:
        #         raise ValueError(
        #             "AutoBatch with batch<1 not supported for Multi-GPU training, "
        #             "please specify a valid batch size, i.e. batch=16."
        #         )
        #     if batch >= 0 and batch % n != 0:  # check batch_size is divisible by device_count
        #         raise ValueError(
        #             f"'batch={batch}' must be a multiple of GPU count {n}. Try 'batch={batch // n * n}' or "
        #             f"'batch={batch // n * n + n}', the nearest batch sizes evenly divisible by {n}."
        #         )
        space = " " * (len(s) + 1)
        for i, d in enumerate(devices):
            s += f"{'' if i == 0 else space}CUDA:{d} ({get_gpu_info(i)})\n"  # bytes to MB
        arg = f"cuda:{gpu_num}"
    elif mps and TORCH_2_0 and torch.backends.mps.is_available():
        # Prefer MPS if available
        arg = "mps"
    else:  # revert to CPU
        arg = "cpu"

    if logger is not None:
        logger.info(s.rstrip())

    return torch.device(arg)


def time_sync():
    """Return PyTorch-accurate time."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def fuse_conv_and_bn(conv, bn):
    """Fuse Conv2d() and BatchNorm2d() layers."""
    fusedconv = (
        nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(conv.weight.device)
    )

    # Prepare filters
    w_conv = conv.weight.view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # Prepare spatial bias
    b_conv = (
        torch.zeros(conv.weight.shape[0], dtype=conv.weight.dtype, device=conv.weight.device)
        if conv.bias is None
        else conv.bias
    )
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def fuse_deconv_and_bn(deconv, bn):
    """Fuse ConvTranspose2d() and BatchNorm2d() layers."""
    fuseddconv = (
        nn.ConvTranspose2d(
            deconv.in_channels,
            deconv.out_channels,
            kernel_size=deconv.kernel_size,
            stride=deconv.stride,
            padding=deconv.padding,
            output_padding=deconv.output_padding,
            dilation=deconv.dilation,
            groups=deconv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(deconv.weight.device)
    )

    # Prepare filters
    w_deconv = deconv.weight.view(deconv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fuseddconv.weight.copy_(torch.mm(w_bn, w_deconv).view(fuseddconv.weight.shape))

    # Prepare spatial bias
    b_conv = torch.zeros(deconv.weight.shape[1], device=deconv.weight.device) if deconv.bias is None else deconv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fuseddconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fuseddconv


def model_info(model, detailed=False, verbose=True, imgsz=640):
    """
    Print and return detailed model information layer by layer.

    Args:
        model (nn.Module): Model to analyze.
        detailed (bool, optional): Whether to print detailed layer information.
        verbose (bool, optional): Whether to print model information.
        imgsz (int | list, optional): Input image size.

    Returns:
        n_l (int): Number of layers.
        n_p (int): Number of parameters.
        n_g (int): Number of gradients.
        flops (float): GFLOPs.
    """
    if not verbose:
        return
    n_p = get_num_params(model)  # number of parameters
    n_g = get_num_gradients(model)  # number of gradients
    layers = __import__("collections").OrderedDict((n, m) for n, m in model.named_modules() if len(m._modules) == 0)
    n_l = len(layers)  # number of layers
    if detailed:
        h = f"{'layer':>5}{'name':>40}{'type':>20}{'gradient':>10}{'parameters':>12}{'shape':>20}{'mu':>10}{'sigma':>10}"
        print("Info: ", h)
        for i, (mn, m) in enumerate(layers.items()):
            mn = mn.replace("module_list.", "")
            mt = m.__class__.__name__
            if len(m._parameters):
                for pn, p in m.named_parameters():
                    print(
                        f"Info: {i:>5g}{f'{mn}.{pn}':>40}{mt:>20}{p.requires_grad!r:>10}{p.numel():>12g}{str(list(p.shape)):>20}{p.mean():>10.3g}{p.std():>10.3g}{str(p.dtype).replace('torch.', ''):>15}"
                    )
            else:  # layers with no learnable params
                print(f"Info: {i:>5g}{mn:>40}{mt:>20}{False!r:>10}{0:>12g}{str([]):>20}{'-':>10}{'-':>10}{'-':>15}")

    flops = get_flops(model, imgsz)  # imgsz may be int or list, i.e. imgsz=640 or imgsz=[640, 320]
    fused = " (fused)" if getattr(model, "is_fused", lambda: False)() else ""
    fs = f", {flops:.1f} GFLOPs" if flops else ""
    yaml_file = getattr(model, "yaml_file", "") or getattr(model, "yaml", {}).get("yaml_file", "")
    model_name = Path(yaml_file).stem.replace("yolo", "YOLO") or "Model"
    print(f"Info: {model_name} summary{fused}: {n_l:,} layers, {n_p:,} parameters, {n_g:,} gradients{fs}")
    return n_l, n_p, n_g, flops


def get_num_params(model):
    """Return the total number of parameters in a YOLO model."""
    return sum(x.numel() for x in model.parameters())


def get_num_gradients(model):
    """Return the total number of parameters with gradients in a YOLO model."""
    return sum(x.numel() for x in model.parameters() if x.requires_grad)


def get_flops(model, imgsz=640):
    """
    Calculate FLOPs (floating point operations) for a model in billions.

    Attempts two calculation methods: first with a stride-based tensor for efficiency,
    then falls back to full image size if needed (e.g., for RTDETR models). Returns 0.0
    if thop library is unavailable or calculation fails.

    Args:
        model (nn.Module): The model to calculate FLOPs for.
        imgsz (int | list, optional): Input image size.

    Returns:
        (float): The model FLOPs in billions.
    """
    try:
        import thop
    except ImportError:
        thop = None  # conda support without 'ultralytics-thop' installed

    if not thop:
        return 0.0  # if not installed return 0.0 GFLOPs

    try:
        model = de_parallel(model)
        p = next(model.parameters())
        if not isinstance(imgsz, list):
            imgsz = [imgsz, imgsz]  # expand if int/float
        try:
            # Method 1: Use stride-based input tensor
            stride = max(int(model.stride.max()), 32) if hasattr(model, "stride") else 32  # max stride
            im = torch.empty((1, p.shape[1], stride, stride), device=p.device)  # input image in BCHW format
            flops = thop.profile(deepcopy(model), inputs=[im], verbose=False)[0] / 1e9 * 2  # stride GFLOPs
            return flops * imgsz[0] / stride * imgsz[1] / stride  # imgsz GFLOPs
        except Exception:
            # Method 2: Use actual image size (required for RTDETR models)
            im = torch.empty((1, p.shape[1], *imgsz), device=p.device)  # input image in BCHW format
            return thop.profile(deepcopy(model), inputs=[im], verbose=False)[0] / 1e9 * 2  # imgsz GFLOPs
    except Exception:
        return 0.0


def initialize_weights(model):
    """Initialize model weights to random values."""
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in {nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU}:
            m.inplace = True


def scale_img(img, ratio=1.0, same_shape=False, gs=32):
    """
    Scale and pad an image tensor, optionally maintaining aspect ratio and padding to gs multiple.

    Args:
        img (torch.Tensor): Input image tensor.
        ratio (float, optional): Scaling ratio.
        same_shape (bool, optional): Whether to maintain the same shape.
        gs (int, optional): Grid size for padding.

    Returns:
        (torch.Tensor): Scaled and padded image tensor.
    """
    if ratio == 1.0:
        return img
    h, w = img.shape[2:]
    s = (int(h * ratio), int(w * ratio))  # new size
    img = F.interpolate(img, size=s, mode="bilinear", align_corners=False)  # resize
    if not same_shape:  # pad/crop img
        h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
    return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean


def intersect_dicts(da, db, exclude=()):
    """
    Return a dictionary of intersecting keys with matching shapes, excluding 'exclude' keys, using da values.

    Args:
        da (dict): First dictionary.
        db (dict): Second dictionary.
        exclude (tuple, optional): Keys to exclude.

    Returns:
        (dict): Dictionary of intersecting keys with matching shapes.
    """
    return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}


def is_parallel(model):
    """
    Return True if model is of type DP or DDP.

    Args:
        model (nn.Module): Model to check.

    Returns:
        (bool): True if model is DataParallel or DistributedDataParallel.
    """
    return isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel))


def de_parallel(model):
    """
    De-parallelize a model: return single-GPU model if model is of type DP or DDP.

    Args:
        model (nn.Module): Model to de-parallelize.

    Returns:
        (nn.Module): De-parallelized model.
    """
    return model.module if is_parallel(model) else model

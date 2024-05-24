import os
import platform
import torch
import torch.distributed as dist
from contextlib import contextmanager

from core.yolov8 import __version__
from core.yolov8.yolov8_utils.checks import check_version


TORCH_2_0 = check_version(torch.__version__, "2.0.0")


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """Decorator to make all processes in distributed training wait for each local_master to do something."""
    initialized = torch.distributed.is_available() and torch.distributed.is_initialized()
    if initialized and local_rank not in (-1, 0):
        dist.barrier(device_ids=[local_rank])
    yield
    if initialized and local_rank == 0:
        dist.barrier(device_ids=[0])


def select_device(device="", gpu_num=0):
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

    if isinstance(device, torch.device):
        return device

    s = f"Ultralytics YOLOv{__version__} 🚀 Python-{platform.python_version()} torch-{torch.__version__} "
    device = str(device).lower()
    for remove in "cuda:", "none", "(", ")", "[", "]", "'", " ":
        device = device.replace(remove, "")  # to string, 'cuda:0' -> '0' and '(0, 1)' -> '0,1'
    cpu = device == "cpu"
    mps = device in ("mps", "mps:0")  # Apple Metal Performance Shaders (MPS)
    if cpu or mps:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        # if device == "cuda":
        #     device = "0"
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)

        # os.environ["CUDA_VISIBLE_DEVICES"] = device  # set environment variable - must be before assert is_available()
        install = (
            "See https://pytorch.org/get-started/locally/ for up-to-date torch install instructions if no "
            "CUDA devices are seen by torch.\n"
            if torch.cuda.device_count() == 0
            else ""
        )
        assert torch.cuda.is_available() and torch.cuda.device_count() >= gpu_num + 1, \
            f"Invalid CUDA 'device={device}' requested. Use 'device=cpu' or pass valid CUDA device(s) if available," \
            + "i.e. 'device=0' or 'device=0,1,2,3' for Multi-GPU." \
            + f"torch.cuda.is_available(): {torch.cuda.is_available()}" \
            + f"\ntorch.cuda.device_count(): {torch.cuda.device_count()}" \
            + f"\nos.environ['CUDA_VISIBLE_DEVICES']: {visible}" \
            + f"\n{install}"

    if not cpu and not mps and torch.cuda.is_available():  # prefer GPU if available
        arg = f"cuda:{gpu_num}"
    elif mps and TORCH_2_0 and torch.backends.mps.is_available():
        # Prefer MPS if available
        arg = "mps"
    else:  # revert to CPU
        arg = "cpu"

    return torch.device(arg)

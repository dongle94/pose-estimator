import os
import torch


def select_device(device='', gpu_num=0):
    # device = str(device).strip().lower().replace('cuda', '').replace('none', '')
    device = str(device).strip().lower().replace('none', '')
    cpu = device == 'cpu'
    mps = device in ("mps", "mps:0")
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif device:  # non-cpu device requested
        # os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable - must be before assert is_available()
        assert torch.cuda.is_available() and torch.cuda.device_count() >= gpu_num + 1, \
            f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)" \
            + f"torch.cuda.is_available(): {torch.cuda.is_available()}" \
            + f"\ntorch.cuda.device_count(): {torch.cuda.device_count()}"

    if not cpu and not mps and torch.cuda.is_available():  # prefer GPU if available
        arg = f"cuda:{gpu_num}"
    elif mps and torch.backends.mps.is_available():
        arg = "mps"
    else:
        arg = "cpu"

    return torch.device(arg)

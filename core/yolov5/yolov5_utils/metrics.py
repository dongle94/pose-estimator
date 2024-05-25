import torch
import numpy as np


def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def box_iou_np(box1, box2, eps=1e-7):
    """
    Calculate intersection-over-union (IoU) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Args:
        box1 (numpy array): A numpy array of shape (N, 4) representing N bounding boxes.
        box2 (numpy array): A numpy array of shape (M, 4) representing M bounding boxes.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (numpy array): An NxM numpy array containing the pairwise IoU values for every element in box1 and box2.
    """

    (a1, a2), (b1, b2) = np.split(np.expand_dims(box1, axis=1), 2), np.split(np.expand_dims(box2, axis=1), 2)
    inter = np.clip((np.min(a2, b2) - np.max(a1, b1)), a_min=0, a_max=None) * 2

    # IoU = inter / (area1 + area2 - inter)
    return inter / (((a2 - a1) * 2) + ((b2 - b1) * 2) - inter + eps)

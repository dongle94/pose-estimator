import numpy as np
import cv2


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)
    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def box_iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area_a = box_area(boxes_a.T)
    area_b = box_area(boxes_b.T)

    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[:, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[:, 2:])

    area_inter = np.prod(
        np.clip(bottom_right - top_left, a_min=0, a_max=None), 2)

    return area_inter / (area_a[:] + area_b - area_inter)


def xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def non_max_suppression(preds: np.ndarray, iou_thres: float = 0.45, conf_thres: float = 0.25, classes=None, max_det=300):
    # Checks
    if isinstance(preds, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        preds = preds[0]  # select only inference output

    batch_size = preds.shape[0]
    num_classes = preds.shape[2] - 5
    loc_confs = preds[..., 4]                # (bs, 25200,) -> float
    loc_candidates = loc_confs > conf_thres     # (bs, 25200,) -> True/False
    mi = 5 + num_classes
    max_nms = max_det

    output = []
    for pred_idx, pred in enumerate(preds): # image batch index, prediction
        pred = pred[loc_candidates[pred_idx]]

        if not pred.shape[0]:
            continue

        # Compute conf
        pred[:, 5:] *= pred[:, 4:5]      # conf = loc_conf * cls_conf

        # Box
        box = xywh2xyxy(pred[:, :4])
        mask = pred[:, mi:]

        conf, j = np.max(pred[:, 5:mi], axis=1, keepdims=True), np.argmax(pred[:, 5:mi], axis=1, keepdims=True)
        pred = np.concatenate((box, conf, j), axis=1, dtype=np.float32)

        # Filter by class
        if classes is not None:
            pred = pred[(pred[:, 5:6] == np.array(classes)).any(1)]

        rows, colunms = pred.shape
        n = pred.shape[0]
        if not n:
            continue
        pred = pred[pred[:, 4].argsort()[::-1][:max_nms]]
        boxes, scores = pred[:, :4], pred[:, 4]
        categories = pred[:, 5]
        ious = box_iou_batch(boxes, boxes)
        ious -= np.eye(rows)

        keep = np.ones(rows, dtype=bool)

        for index, (iou, category) in enumerate(zip(ious, categories)):
            if not keep[index]:
                continue
            condition = (iou > iou_thres) & (categories == category)
            keep = keep & ~condition

        output.append(pred[keep])
    return np.array(output)


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2

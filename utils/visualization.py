import numpy as np
import cv2
import math


def vis_pose_result(img, pred_kepts, model='coco', radius=4, thickness=1, kpt_score_thr=0.3):
    """Visualize the detection results on the image.

    Args:
        img (str | np.ndarray): Image filename or loaded image.
        pred_kepts (list[dict]): The results to draw over `img`
                (bbox_result, pose_result).
        model (str): The dataset format
        radius (int): Radius of circles.
        thickness (int): Thickness of lines.
        kpt_score_thr (float): The threshold to visualize the keypoints.
    """
    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255],
                        [255, 0, 0], [255, 255, 255]])

    if model == 'coco':
        skeleton = [[0, 1], [0, 2], [1, 3], [2, 4],  # face
                    [8, 10], [6, 8], [5, 6], [5, 7], [7, 9],    # upper
                    [5, 11], [6, 12],
                    [11, 12], [14, 16], [12, 14], [13, 15], [11, 13]]     # lower
        # 0: blue, 9: yellow, 16: green
        kpt_color = palette[[
            17, 17, 17, 17, 17,
            17, 17, 17, 17, 17,
            17, 17, 17, 17, 17,
            17, 17
        ]]

        link_color = palette[[
            12, 12, 12, 12,
            12, 12, 12, 12, 12,
            12, 12,
            12, 12, 12, 12, 12
        ]]
    elif model == 'mpii':
        skeleton = [[8, 9], [7, 8],  # face
                    [10, 11], [11, 12], [7, 12], [7, 13], [13, 14], [14, 15], [6, 7],  # upper
                    [2, 6], [3, 6], [0, 1], [1, 2], [3, 4], [4, 5]]  # lower

        kpt_color = palette[[
            17, 17, 17, 17, 17,
            17, 17, 17, 17, 17,
            17, 17, 17, 17, 17,
            17
        ]]

        link_color = palette[[
            12, 12,
            12, 12, 12, 12, 12, 12, 12,
            12, 12, 12, 12, 12, 12
        ]]
    else:
        raise ValueError(f'model: {model} is not supported.')

    img = draw_keypoints(
        img=img,
        batch_kepts=pred_kepts,
        skeleton=skeleton,
        kpt_score_thr=kpt_score_thr,
        kpt_color=kpt_color,
        link_color=link_color,
        radius=radius,
        thickness=thickness
    )

    return img


def draw_keypoints(img, batch_kepts, skeleton=None, kpt_score_thr=0.3, kpt_color=None, link_color=None,
                   radius=4, thickness=1, show_keypoint_weight=False):
    img_h, img_w, _ = img.shape

    for kepts in batch_kepts:
        kepts = np.array(kepts, copy=False)

        # draw links
        assert len(link_color) == len(skeleton)
        for skid, sk in enumerate(skeleton):
            pos1 = (int(kepts[sk[0], 0]), int(kepts[sk[0], 1]))
            pos2 = (int(kepts[sk[1], 0]), int(kepts[sk[1], 1]))

            if (pos1[0] <= 0 or pos1[0] >= img_w
                or pos1[1] <=0 or pos1[1] >= img_h
                or pos2[0] <= 0 or pos2[0] >= img_w
                or pos2[1] <= 0 or pos2[1] >= img_h
                or kepts[sk[0], 2] < kpt_score_thr or kepts[sk[1], 2] < kpt_score_thr
                or link_color[skid] is None):
                continue
            color = tuple(int(c) for c in link_color[skid])
            cv2.line(img, pos1, pos2, color, thickness=thickness)

        # draw kepoints
        assert len(kpt_color) == len(kepts)
        for kid, kpt in enumerate(kepts):
            x_coord, y_coord, kpt_score = int(kpt[0]), int(kpt[1]), kpt[2]

            if kpt_score < kpt_score_thr or kpt_color[kid] is None:
                continue

            color = tuple(int(c) for c in kpt_color[kid])
            cv2.circle(img, (int(x_coord), int(y_coord)), radius, color, -1)

    return img


def get_heatmaps(batch_heatmaps, colormap=None, draw_index: list = None):
    heatmaps = []
    if type(draw_index) == list and len(draw_index) == 0:
        draw_index = None
    if len(batch_heatmaps.shape) == 3:
        batch_heatmaps = [batch_heatmaps]

    for _heatmaps in batch_heatmaps:
        new_heatmap = np.zeros((_heatmaps.shape[1], _heatmaps.shape[2]), dtype=np.float32)
        for idx, heatmap in enumerate(_heatmaps):
            if draw_index is not None and idx not in draw_index:
                continue
            new_heatmap = np.maximum(new_heatmap, heatmap)

        if colormap is not None:
            new_heatmap = new_heatmap * 255
            new_heatmap = cv2.applyColorMap(new_heatmap.astype(np.uint8), colormap)
        heatmaps.append(new_heatmap)
    return heatmaps


def merge_heatmaps(heatmaps, boxes, img_size):
    if heatmaps and len(heatmaps[0].shape) == 3:
        heatmap = np.zeros((img_size[0], img_size[1], img_size[2]), dtype=np.float32)
    else:
        heatmap = np.zeros((img_size[0], img_size[1]), dtype=np.float32)

    for h, b in zip(heatmaps, boxes):
        if len(heatmaps[0].shape) == 3:
            new_heatmap = np.zeros((img_size[0], img_size[1], img_size[2]), dtype=np.float32)
        else:
            new_heatmap = np.zeros((img_size[0], img_size[1]), dtype=np.float32)
        x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        box_w, box_h = x2 - x1, y2 - y1

        # h = cv2.resize(h, (img_size[0], img_size[1]))
        h0, w0 = h.shape[:2]
        h = h[int(h0 * 0.1):int(h0 * 0.9), int(w0 * 0.1):int(w0 * 0.9)]

        h0, w0 = h.shape[:2]
        box_ratio = box_w / box_h
        if box_ratio < img_size[0] / img_size[1]:
            nw = h0 * box_ratio
            k = int((w0 - nw) / 2)
            h = h[:, k: w0 - k + 1]
        else:
            nh = w0 / box_ratio
            k = int((h0 - nh) / 2)
            h = h[k: h0 - k +1, :]

        resize_h = cv2.resize(h, (box_w, box_h))

        new_heatmap[y1:y2, x1:x2] = resize_h
        heatmap = np.maximum(heatmap, new_heatmap)
    return heatmap


def get_rule_heatmaps(batch_heatmaps, colormap=None, draw_index: list = None):
    heatmaps = []
    if type(draw_index) == list and len(draw_index) == 0:
        draw_index = None
    if len(batch_heatmaps.shape) == 3:
        batch_heatmaps = [batch_heatmaps]

    for batch, _heatmaps in enumerate(batch_heatmaps):
        if colormap is not None:
            new_heatmap = np.zeros((_heatmaps.shape[1], _heatmaps.shape[2], 3), dtype=np.float32)
        else:
            new_heatmap = np.zeros((_heatmaps.shape[1], _heatmaps.shape[2]), dtype=np.float32)
        for idx, heatmap in enumerate(_heatmaps):
            if draw_index is not None and idx not in draw_index:
                continue

            if colormap is not None:
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)
                heatmap = heatmap * 255
                heatmap = cv2.applyColorMap(heatmap.astype(np.uint8), colormap[batch][idx])


            new_heatmap = np.maximum(new_heatmap, heatmap)
        heatmaps.append(new_heatmap)
    return heatmaps

import numpy as np
import cv2
import math


def vis_pose_result(model,
                    img,
                    result,
                    radius=4,
                    thickness=1,
                    kpt_score_thr=0.3):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str | np.ndarray): Image filename or loaded image.
        result (list[dict]): The results to draw over `img`
                (bbox_result, pose_result).
        radius (int): Radius of circles.
        thickness (int): Thickness of lines.
        kpt_score_thr (float): The threshold to visualize the keypoints.
        skeleton (list[tuple()]): Default None.
        show (bool):  Whether to show the image. Default True.
        out_file (str|None): The filename of the output visualization image.
    """

    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255],
                        [255, 0, 0], [255, 255, 255]])

    # show the results
    skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],
                [3, 5], [4, 6]]

    pose_link_color = palette[[
        16, 16, 0, 0, 9, 9, 9, 9, 16, 0, 16, 0, 9, 9, 9, 9, 9, 9, 9
    ]]
    pose_kpt_color = palette[[
        9, 9, 9, 9, 9, 16, 0, 16, 0, 16, 0, 16, 0, 16, 0, 16, 0
    ]]

    img = imshow_keypoints(
        img, result, skeleton, kpt_score_thr,
        pose_kpt_color, pose_link_color, radius,
        thickness
    )

    return img


def imshow_keypoints(img,
                     pose_result,
                     skeleton=None,
                     kpt_score_thr=0.3,
                     pose_kpt_color=None,
                     pose_link_color=None,
                     radius=4,
                     thickness=1,
                     show_keypoint_weight=False):
    """Draw keypoints and links on an image.

    Args:
            img (str or Tensor): The image to draw poses on. If an image array
                is given, id will be modified in-place.
            pose_result (list[kpts]): The poses to draw. Each element kpts is
                a set of K keypoints as an Kx3 numpy.ndarray, where each
                keypoint is represented as x, y, score.
            kpt_score_thr (float, optional): Minimum score of keypoints
                to be shown. Default: 0.3.
            pose_kpt_color (np.array[Nx3]`): Color of N keypoints. If None,
                the keypoint will not be drawn.
            pose_link_color (np.array[Mx3]): Color of M links. If None, the
                links will not be drawn.
            thickness (int): Thickness of lines.
    """

    #img = cv2.imread(img)
    img_h, img_w, _ = img.shape

    for kpts in pose_result:

        kpts = np.array(kpts, copy=False)

        # draw each point on image
        if pose_kpt_color is not None:
            assert len(pose_kpt_color) == len(kpts)

            for kid, kpt in enumerate(kpts):
                x_coord, y_coord, kpt_score = int(kpt[0]), int(kpt[1]), kpt[2]
                # x_coord, y_coord = int(kpt[0]), int(kpt[1])

                if kpt_score < kpt_score_thr or pose_kpt_color[kid] is None:
                    # skip the point that should not be drawn
                    continue

                color = tuple(int(c) for c in pose_kpt_color[kid])
                if show_keypoint_weight:
                    img_copy = img.copy()
                    cv2.circle(img_copy, (int(x_coord), int(y_coord)), radius,
                               color, -1)
                    transparency = max(0, min(1, kpt_score))
                    cv2.addWeighted(
                        img_copy,
                        transparency,
                        img,
                        1 - transparency,
                        0,
                        dst=img)
                else:
                    cv2.circle(img, (int(x_coord), int(y_coord)), radius,
                               color, -1)

        # draw links
        if skeleton is not None and pose_link_color is not None:
            assert len(pose_link_color) == len(skeleton)

            for sk_id, sk in enumerate(skeleton):
                pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
                pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))

                if (pos1[0] <= 0 or pos1[0] >= img_w or pos1[1] <= 0
                        or pos1[1] >= img_h or pos2[0] <= 0 or pos2[0] >= img_w
                        or pos2[1] <= 0 or pos2[1] >= img_h
                        or kpts[sk[0], 2] < kpt_score_thr
                        or kpts[sk[1], 2] < kpt_score_thr
                        or pose_link_color[sk_id] is None):
                    # skip the link that should not be drawn
                    continue
                color = tuple(int(c) for c in pose_link_color[sk_id])
                if show_keypoint_weight:
                    img_copy = img.copy()
                    X = (pos1[0], pos2[0])
                    Y = (pos1[1], pos2[1])
                    mX = np.mean(X)
                    mY = np.mean(Y)
                    length = ((Y[0] - Y[1])**2 + (X[0] - X[1])**2)**0.5
                    angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
                    stickwidth = 2
                    polygon = cv2.ellipse2Poly(
                        (int(mX), int(mY)), (int(length / 2), int(stickwidth)),
                        int(angle), 0, 360, 1)
                    cv2.fillConvexPoly(img_copy, polygon, color)
                    transparency = max(
                        0, min(1, 0.5 * (kpts[sk[0], 2] + kpts[sk[1], 2])))
                    cv2.addWeighted(
                        img_copy,
                        transparency,
                        img,
                        1 - transparency,
                        0,
                        dst=img)
                else:
                    cv2.line(img, pos1, pos2, color, thickness=thickness)

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

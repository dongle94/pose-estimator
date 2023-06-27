import os
import sys
import cv2
import numpy as np
import math
import torch
import torch.nn as nn
from torchvision import transforms

from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
os.chdir(ROOT)


class HRNet(nn.Module):
    def __init__(self, weight="pose_hrnet_w48_384x288.pth", device="",  img_size=(288, 384), fp16=False):
        super().__init__()

        self.device = self.select_device(device)
        model = torch.load(weight).to(self.device).eval()

        self.fp16 = fp16
        self.model = model.half() if fp16 else model.float()
        self.img_size = img_size

        self.pose_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def warmup(self, imgsz=(1, 3, 384, 288)):
        im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
        self.forward(im)
        print("-- HRNet warmup -- ")

    def preprocess(self, im, boxes):
        person_results = []
        for bbox in boxes[:, :5]:
            box = bbox.cpu().numpy()
            person_results.append(box)
        bboxes_xyxy = np.array(person_results)

        centers = []
        scales = []
        for box in bboxes_xyxy:
            center, scale = self.box_to_center_scale(box)
            centers.append(center)
            scales.append(scale)

        rotation = 0
        model_inputs = []
        for center, scale in zip(centers, scales):
            trans = self.get_affine_transform(center, scale, rotation, self.img_size)
            # Crop smaller image of people
            model_input = cv2.warpAffine(
                im,
                trans,
                (int(self.img_size[0]), int(self.img_size[1])),
                flags=cv2.INTER_LINEAR)
            # hwc -> 1chw
            model_input = self.pose_transform(model_input)  # .unsqueeze(0)
            model_inputs.append(model_input)

        # n * 1chw -> nchw
        model_inputs = torch.stack(model_inputs)
        inputs = model_inputs.to(self.device)
        if self.fp16:
            inputs = inputs.half()

        return inputs, centers, scales

    def forward(self, inputs):
        output = self.model(inputs)
        return output

    def postprocess(self, preds, center, scale):
        batch_heatmaps = preds.cpu().detach().numpy()

        # raw_heatmaps -> coordinates
        coords, maxvals = self.get_max_preds(batch_heatmaps)

        heatmap_height = batch_heatmaps.shape[2]
        heatmap_width = batch_heatmaps.shape[3]

        # post-processing
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                    diff = np.array(
                        [
                            hm[py][px + 1] - hm[py][px - 1],
                            hm[py + 1][px] - hm[py - 1][px]
                        ]
                    )
                    coords[n][p] += np.sign(diff) * .25
        preds = coords.copy()

        # Transform back
        for i in range(coords.shape[0]):
            preds[i] = self.transform_preds(
                coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
            )
        preds = np.concatenate((preds, maxvals), axis=2)
        return preds, batch_heatmaps


    @staticmethod
    def select_device(device=''):
        device = str(device).strip().lower().replace('cuda', '').replace('none', '')
        cpu = device == 'cpu'
        if cpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        elif device:  # non-cpu device requested
            os.environ[
                'CUDA_VISIBLE_DEVICES'] = device  # set environment variable - must be before assert is_available()

        if not cpu and torch.cuda.is_available():
            device = device if device else '0'
            arg = 'cuda:0'
        else:
            arg = 'cpu'

        return torch.device(arg)

    def box_to_center_scale(self, box):
        """convert a box to center,scale information required for pose transformation
        Parameters
        ----------
        box : list of tuple
            list of length 2 with two tuples of floats representing
            bottom left and top right corner of a box

        Returns
        -------
        (numpy array, numpy array)
            Two numpy arrays, coordinates for the center of the box and the scale of the box
        """
        center = np.zeros((2), dtype=np.float32)

        top_left_corner = [box[0], box[1]]
        bottom_right_corner = [box[2], box[3]]
        # bottom_left_corner = box[0]
        # top_right_corner = box[1]
        box_width = bottom_right_corner[0] - top_left_corner[0]
        box_height = bottom_right_corner[1] - top_left_corner[1]
        top_left_x = top_left_corner[0]
        top_left_y = top_left_corner[1]
        # bottom_left_x = bottom_left_corner[0]
        # bottom_left_y = bottom_left_corner[1]
        center[0] = top_left_x + box_width * 0.5
        center[1] = top_left_y + box_height * 0.5

        aspect_ratio = self.img_size[0] * 1.0 / self.img_size[1]
        pixel_std = 200

        if box_width > aspect_ratio * box_height:
            box_height = box_width * 1.0 / aspect_ratio
        elif box_width < aspect_ratio * box_height:
            box_width = box_height * aspect_ratio
        scale = np.array(
            [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

    @staticmethod
    def get_affine_transform(center, scale, rot, output_size,
                             shift=np.array([0, 0], dtype=np.float32), inv=0):
        def get_3rd_point(a, b):
            direct = a - b
            return b + np.array([-direct[1], direct[0]], dtype=np.float32)

        def get_dir(src_point, rot_rad):
            sn, cs = np.sin(rot_rad), np.cos(rot_rad)

            src_result = [0, 0]
            src_result[0] = src_point[0] * cs - src_point[1] * sn
            src_result[1] = src_point[0] * sn + src_point[1] * cs

            return src_result

        if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
            print(scale)
            scale = np.array([scale, scale])

        scale_tmp = scale * 200.0
        src_w = scale_tmp[0]
        dst_w = output_size[0]
        dst_h = output_size[1]

        _rot_rad = np.pi * rot / 180
        src_dir = get_dir([0, src_w * -0.5], _rot_rad)
        dst_dir = np.array([0, dst_w * -0.5], np.float32)

        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center + scale_tmp * shift
        src[1, :] = center + src_dir + scale_tmp * shift
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

        src[2:, :] = get_3rd_point(src[0, :], src[1, :])
        dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

        if inv:
            trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

        return trans

    def transform_preds(self, coords, center, scale, output_size):
        target_coords = np.zeros(coords.shape)

        trans = self.get_affine_transform(center, scale, 0, output_size, inv=1)
        for p in range(coords.shape[0]):
            target_coords[p, 0:2] = self.affine_transform(coords[p, 0:2], trans)
        return target_coords

    @staticmethod
    def affine_transform(pt, t):
        new_pt = np.array([pt[0], pt[1], 1.]).T
        new_pt = np.dot(t, new_pt)
        return new_pt[:2]

    @staticmethod
    def get_max_preds(batch_heatmaps):
        '''
        get predictions from score maps
        heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
        '''
        assert isinstance(batch_heatmaps, np.ndarray), \
            'batch_heatmaps should be numpy.ndarray'
        assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

        batch_size = batch_heatmaps.shape[0]
        num_joints = batch_heatmaps.shape[1]
        width = batch_heatmaps.shape[3]
        heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
        idx = np.argmax(heatmaps_reshaped, 2)       # 최대값 인덱스
        maxvals = np.amax(heatmaps_reshaped, 2)     # 최대값 value

        maxvals = maxvals.reshape((batch_size, num_joints, 1))
        idx = idx.reshape((batch_size, num_joints, 1))

        preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

        preds[:, :, 0] = (preds[:, :, 0]) % width
        preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

        pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
        pred_mask = pred_mask.astype(np.float32)

        preds *= pred_mask
        return preds, maxvals


if __name__ == "__main__":
    from detectors.yolov5_pt import YoloDetector
    from utils.visualization import vis_pose_result, get_heatmaps, merge_heatmaps
    detector = YoloDetector(weight='./weights/yolov5n.pt', device=0, img_size=640)
    detector.warmup()

    keypointer = HRNet(weight="./weights/hrnet_merge_w48_384x288.pth", device=0, fp16=True, img_size=(288, 384))
    keypointer.warmup()

    img = cv2.imread('./data/images/army.jpg')
    im, im0 = detector.preprocess(img)
    pred = detector.forward(im)
    pred, det = detector.postprocess(pred, im.shape, im0.shape)

    input_img = im0.copy()
    kept_inputs, centers, scales = keypointer.preprocess(input_img, det)
    kept_pred = keypointer.forward(kept_inputs)
    kept_pred, raw_heatmaps = keypointer.postprocess(kept_pred, np.asarray(centers), np.asarray(scales))

    # process heatmap
    heatmaps = get_heatmaps(raw_heatmaps, colormap=cv2.COLORMAP_JET)
    heatmap = merge_heatmaps(heatmaps, det, im0.shape)

    for d in det:
        x1, y1, x2, y2 = map(int, d[:4])
        cv2.rectangle(im0, (x1, y1), (x2, y2), (128, 128, 240), thickness=2, lineType=cv2.LINE_AA)
    im0 = vis_pose_result(model=None, img=im0, result=kept_pred)
    cv2.imshow('_', im0)

    # if heatmap's colormap is None, activate annotaion.
    #new_heatmap = np.uint8(255 * heatmap)
    #new_heatmap = cv2.applyColorMap(new_heatmap, cv2.COLORMAP_JET)
    new_heatmap = cv2.add((0.4 * heatmap).astype(np.uint8), im0)
    cv2.imshow("+", new_heatmap)
    cv2.waitKey(0)

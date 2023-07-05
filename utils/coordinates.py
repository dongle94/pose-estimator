import numpy as np


def get_angle(p1, p2, p3):
    # rp1, rp2, rp3 = keys_pred[4][:2], keys_pred[6][:2], keys_pred[8][:2]
    vec1, vec2 = (p1[0] - p2[0], p1[1] - p2[1]), (p3[0] - p2[0], p3[1] - p2[1])
    dir1, dir2 = get_dir(vec1), get_dir(vec2)
    th1 = np.arctan(vec1[1] / vec1[0]) * 180 / np.pi
    th2 = np.arctan(vec2[1] / vec2[0]) * 180 / np.pi

    if np.abs(dir1 - dir2) == 3:        # dir in [1, 4]
        return 180 - np.abs(th1) - np.abs(th2)
    elif np.abs(dir1 - dir2) == 0:      # dir1 == dir2
        return np.abs(th1 - th2)
    elif np.abs(dir1 - dir2) == 1:
        if dir1 in [1, 2] and dir2 in [1, 2]:
            return np.abs(th1) + np.abs(th2)
        elif dir1 in [2, 3] and dir2 in [2, 3]:
            return 180 - np.abs(th1) - np.abs(th2)
        elif dir1 in [3, 4] and dir2 in [3, 4]:
            return np.abs(th1) + np.abs(th2)
    elif np.abs(dir1 - dir2) == 2:
        return min(180 + (np.abs(th1) - np.abs(th2)), 180 + (np.abs(th2) - np.abs(th1)))


def get_dir(vector):
    # 우상단: 1, 우하단: 2, 좌하단: 3, 좌상단: 4, 원점: 5
    vx, vy = vector[0], vector[1]
    if np.sign(vx) == 1:
        if np.sign(vy) == -1:
            return 1    # 우상단
        elif np.sign(vy) == 1:
            return 2    # 우하단
        elif np.sign(vy) == 0:
            return 2
    elif np.sign(vx) == -1:
        if np.sign(vy) == -1:
            return 4    # 좌상단
        elif np.sign(vy) == 1:
            return 3    # 좌하단
        elif np.sign(vy) == 0:
            return 3    # 좌하단
    elif np.sign(vx) == 0:
        if np.sign(vy) == -1:
            return 1    # 우상단
        elif np.sign(vy) == 1:
            return 2    # 우하단
        elif np.sign(vy) == 0:
            return 5    # 원점

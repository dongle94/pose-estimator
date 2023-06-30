# Input으로 [1,17,3] 크기의 array가 들어옴

def count_rule(coordinates, arm_stretch_angle=150, arm_between_angle=80, 
                            leg_stretch_angle=150, leg_between_angle=60):
    
    import numpy as np
    import pandas as pd
    import cv2 
    
    IsArmStretch = 0 # 팔은 쭉 폈는가
    IsArmClose = 0 # 팔이 잘 벌려져 있는가

    IsLegStretch = 0 # 다리를 일자로 쭉 폈는가
    IsLegClose = 0 # 다리를 벌리지 않았는가

    IsKneeStop = 1 # 무릎을 굽히지 않았는가
    IsChinOver = 0 # 턱이 봉을 넘었는가

    # 1. 상체
    # 1.1. 어깨 - 팔꿈치 - 손목
    joint_idx = {'left_arm':[5, 7, 9], 'right_arm':[6,8,10]}
    for k, v in joint_idx.items():
        first = coordinates[v[0]]
        mid1 = coordinates[v[1]]
        end = coordinates[v[2]]

        angle_tmp1, mid1 = calculate_angle2D_3point(first ,mid1, end)
        if angle_tmp1 >= arm_stretch_angle:
            IsArmStretch = 1
        else:
            IsArmStretch = 0
            break

    # 1.2. 손목 - 목 - 손목
    # ArmClose
    joint_idx = {'arm_between':[9, 5, 6, 10]}
    for k, v in joint_idx.items():
        first = coordinates[v[0]]
        mid_1 = coordinates[v[1]]
        mid_2 = coordinates[v[2]]
        end = coordinates[v[3]]
        angle_tmp2, mid2 = calculate_angle2D_4point(first ,mid_1, mid_2, end)
        if angle_tmp2 <= arm_between_angle:
            IsArmClose = 1
        else:
            IsArmClose = 0
            break

    # 2. 하체 
    # 2.1. 고관절-무릎-발목
    joint_idx = {'left_leg':[11, 13, 15], 'right_leg':[12, 14, 16]}
    for k, v in joint_idx.items():
        first = coordinates[v[0]]
        mid3 = coordinates[v[1]]
        end = coordinates[v[2]]

        angle_tmp3, mid3 = calculate_angle2D_3point(first ,mid3, end)
        if angle_tmp3 >= leg_stretch_angle:
            IsLegStretch = 1
        else:
            IsLegStretch = 0
            break

    # 2.2. 왼쪽 무릎 - 허리중심 - 오른쪽 무릎
    joint_idx = {'leg_between':[13, 11, 12, 14]} 
    for k, v in joint_idx.items():
        first = coordinates[v[0]]
        mid_1 = coordinates[v[1]]
        mid_2 = coordinates[v[2]]
        end = coordinates[v[3]]
        angle_tmp4, mid4 = calculate_angle2D_4point(first ,mid_1, mid_2, end)
        if angle_tmp4 <= leg_between_angle:
            IsLegClose = 1
        else:
            IsLegClose = 0
            break
    
    # 2.3. 무릎 y 값 vs 발목 y 값
    joint_idx = {'left_leg':[13, 15], 'right_leg':[14, 16]}
    for k, v in joint_idx.items():
        knee = coordinates[v[0]][:2]
        ankle = coordinates[v[1]][:2]
        knee_y = knee[1]
        ankle_y = ankle[1]
        if ankle_y <= knee_y:
            IsKneeStop = 0
            break
    
    # 3. 턱을 넘었는가?
    # 3.1. 예측 봉 위치
    left_wrist = coordinates[9][:2][1]
    right_wrist = coordinates[10][:2][1]
    bar_location = min(left_wrist,right_wrist)
    
    # 3.2. 예측 턱 위치
    joint_idx = {'left_eye_nose':[1, 0], 'right_eye_nose':[2,0]} 
    expected_chin = []
    for k, v in joint_idx.items():
        eye = coordinates[v[0]]
        nose = coordinates[v[1]]
        expected_chin.append(calculate_chin(eye ,nose))
    chin_location = min(expected_chin)[1]
    
    if chin_location <= bar_location:
        IsChinOver = 1
    else:
        IsChinOver = 0
        
    result_tmp = pd.DataFrame({'IsArmStretch' : [IsArmStretch], 
                            'IsArmClose' : [IsArmClose],
                            'IsLegStretch' : [IsLegStretch],
                            'IsLegClose' : [IsLegClose],
                            'IsKneeStop' : [IsKneeStop],
                            'IsChinOver' : [IsChinOver]})
    
    angle_list = [angle_tmp1, angle_tmp2, angle_tmp3, angle_tmp4]
    mid_list = [map(int,mid1), map(int,mid2), map(int,mid3), map(int,mid4)]   

    return result_tmp, angle_list, mid_list


def calculate_angle2D_3point(a, b, c):

    import numpy as np

    a = np.array(a[:2]) #first
    b = np.array(b[:2]) #mid
    c = np.array(c[:2]) #end

    ba = a-b
    bc = c-b
    dot_result = np.dot(ba, bc)


    ba_size = np.linalg.norm(ba)
    bc_size = np.linalg.norm(bc)

    # 컴퓨팅 계산 오류 방지 clip
    radi_temp = np.clip(dot_result / (ba_size*bc_size), -1.0, 1.0)
    radi = np.arccos(radi_temp)
    angle = np.abs(radi*180.0/np.pi)

    
    return round(angle, 2), b

def calculate_angle2D_4point(a, b1, b2, c):

    import numpy as np
    b1 = np.array(b1[:2])
    b2 = np.array(b2[:2])
    
    b = [round(abs((b1[0]+b2[0])/2),2), round(abs((b1[1]+b2[1])/2),2)]

    a = np.array(a[:2]) #first
    b = np.array(b[:2]) #mid
    c = np.array(c[:2]) #end

    ba = a-b
    bc = c-b
    dot_result = np.dot(ba, bc)

    ba_size = np.linalg.norm(ba)
    bc_size = np.linalg.norm(bc)
    radi = np.arccos(dot_result / (ba_size*bc_size))
    angle = np.abs(radi*180.0/np.pi) # 60분법 변환
    
    return round(angle,2), b

def calculate_chin(a, b):
    import numpy as np
    chin_x = b[0]
    a = np.array(a[1]) # eye,  y 좌표
    b = np.array(b[1]) # nose, y 좌표
    
    eye_to_nose = abs(a-b)
    # x좌표는 nose, y좌표는 nose + 
    chin = [chin_x, b + round(eye_to_nose*1.618,2)]
    
    return chin










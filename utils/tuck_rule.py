# Input으로 [1,17,3] 크기의 array가 들어옴

def count_rule(coordinates, arm_stretch_angle=130, arm_between_angle=80, 
                            leg_stretch_angle=150, leg_between_angle=60):
    
    import numpy as np
    import pandas as pd
    
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
        mid = coordinates[v[1]]
        end = coordinates[v[2]]
        if 'right' in k:
            dimension = 1 # < 각도
        else:
            dimension = -1 # > 각도
        angle_tmp = calculate_angle2D_3point(first ,mid, end, dimension)
        if angle_tmp >= arm_stretch_angle:
            IsArmStretch = 1
        else:
            IsArmStretch = 0
            break

    # 1.2. 손목 - 목 - 손목
    joint_idx = {'arm_between':[9, 5, 6, 10]}
    for k, v in joint_idx.items():
        first = coordinates[v[0]]
        mid1 = coordinates[v[1]]
        mid2 = coordinates[v[2]]
        end = coordinates[v[3]]
        angle_tmp = calculate_angle2D_4point(first ,mid1, mid2, end, 1)
        if angle_tmp <= arm_between_angle:
            IsArmClose = 1
        else:
            IsArmClose = 0
            break

    # 2. 하체 
    # 2.1. 고관절-무릎-발목
    joint_idx = {'left_leg':[11, 13, 15], 'right_leg':[12, 14, 16]}
    for k, v in joint_idx.items():
        first = coordinates[v[0]]
        mid = coordinates[v[1]]
        end = coordinates[v[2]]
        if 'left' in k:
            dimension = 1 
        else:
            dimension = -1
        angle_tmp = calculate_angle2D_3point(first ,mid, end, dimension)
        if angle_tmp >= leg_stretch_angle:
            IsLegStretch = 1
        else:
            IsLegStretch = 0
            break

    # 2.2. 왼쪽 무릎 - 허리중심 - 오른쪽 무릎
    joint_idx = {'leg_between':[13, 11, 12, 14]} 
    for k, v in joint_idx.items():
        first = coordinates[v[0]]
        mid1 = coordinates[v[1]]
        mid2 = coordinates[v[2]]
        end = coordinates[v[3]]
        angle_tmp = calculate_angle2D_4point(first ,mid1, mid2, end, -1)
        if angle_tmp <= leg_between_angle:
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
    joint_idx = {'left_ear_nose':[3, 0], 'right_ear_nose':[4,0]} 
    expected_chin = []
    for k, v in joint_idx.items():
        ear = coordinates[v[0]]
        nose = coordinates[v[1]]
        expected_chin.append(calculate_chin(ear ,nose))
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
    
    return result_tmp
    

def calculate_angle2D_3point(a, b, c, direction=-1):
    """
    calculate_angle2D is divided by left and right side because this function uses external product
    input : a,b,c -> landmarks with shape [x,y,z,visibility]
          direction -> int -1 or 1
                      -1 means Video(photo) for a person's left side and 1 means Video(photo) for a person's right side
    output : angle between vector ba and bc with range 0~360
    """
    import numpy as np
    # external product's z value
    external_z = (b[0]-a[0])*(b[1]-c[1]) - (b[1]-a[1])*(b[0]-c[0])

    a = np.array(a[:2]) #first
    b = np.array(b[:2]) #mid
    c = np.array(c[:2]) #end

    ba = b-a
    bc = b-c
    dot_result = np.dot(ba, bc)


    ba_size = np.linalg.norm(ba)
    bc_size = np.linalg.norm(bc)

    radi_temp = np.clip(dot_result / (ba_size*bc_size), -1.0, 1.0)
    radi = np.arccos(radi_temp)
    angle = np.abs(radi*180.0/np.pi)

    
    if external_z * direction > 0:
        angle = 360 - angle
    
    return round(angle, 2)

def calculate_angle2D_4point(a, b1, b2, c, direction=-1):
    """
    calculate_angle2D is divided by left and right side because this function uses external product
    input : a,b,c -> landmarks with shape [x,y,z,visibility]
          direction -> int -1 or 1
                      -1 means Video(photo) for a person's left side and 1 means Video(photo) for a person's right side
    output : angle between vector ba and bc with range 0~360
    """
    import numpy as np
    b1 = np.array(b1[:2])
    b2 = np.array(b2[:2])
    
    b = [round(abs((b1[0]+b2[0])/2),2), round(abs((b1[1]+b2[1])/2),2)]
    
    # external product's z value
    external_z = (b[0]-a[0])*(b[1]-c[1]) - (b[1]-a[1])*(b[0]-c[0])

    a = np.array(a[:2]) #first
    b = np.array(b[:2]) #mid
    c = np.array(c[:2]) #end

    
    ba = b-a
    bc = b-c
    dot_result = np.dot(ba, bc)

    ba_size = np.linalg.norm(ba)
    bc_size = np.linalg.norm(bc)
    radi = np.arccos(dot_result / (ba_size*bc_size))
    angle = np.abs(radi*180.0/np.pi) # 60분법 변환

    
    if external_z * direction > 0:
        angle = 360 - angle
    
    return round(angle,2)

def calculate_chin(a, b):
    import numpy as np
    chin_x = b[0]
    a = np.array(a[1]) # ear,  y 좌표
    b = np.array(b[1]) # nose, y 좌표
    
    ear_to_nose = abs(a-b)
    # x좌표는 nose, y좌표는 nose + 
    chin = [chin_x, b+round(ear_to_nose*1.6,2)]
    
    return chin










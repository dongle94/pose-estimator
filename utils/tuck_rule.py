class tuck_rule():
    def __init__(self, arm_stretch_angle=150, arm_between_angle=80,
                      leg_stretch_angle=150, leg_between_angle=60):
        import pandas as pd
        
        self.IsArmStretch = 0 # 팔은 쭉 폈는가
        # self.IsArmClose = 0 # 팔이 잘 벌려져 있는가
        self.IsLegStretch = 0 # 다리를 일자로 쭉 폈는가
        self.IsLegClose = 0 # 다리를 벌리지 않았는가
        self.IsKneeStop = 1 # 무릎을 굽히지 않았는가
        self.IsChinOver = 0 # 턱이 봉을 넘었는가
        
        self.arm_stretch_angle = arm_stretch_angle
        self.arm_between_angle = arm_between_angle
        self.leg_stretch_angle = leg_stretch_angle
        self.leg_between_angle = leg_between_angle
        
        self.anno_df = pd.DataFrame({'Nose_x' : [],
                                     'Nose_y' : [], 
                                    'LeftEye_x' : [],
                                    'LeftEye_y' : [],
                                    'RightEye_x' : [],
                                    'RightEye_y' : [],
                                    'LeftEar_x' : [],
                                    'LeftEar_y' : [],
                                    'RightEar_x' : [],
                                    'RightEar_y' : [],
                                    'LeftShoulder_x' : [],
                                    'LeftShoulder_y' : [],
                                    'RightShoulder_x' : [],
                                    'RightShoulder_y' : [],
                                    'LeftElbow_x' : [],
                                    'LeftElbow_y' : [],
                                    'RightElbow_x' : [],
                                    'RightElbow_y' : [],
                                    'LeftWrist_x' : [],
                                    'LeftWrist_y' : [],
                                    'RightWrist_x' : [],
                                    'RightWrist_y' : [],
                                    'LeftHip_x' : [],
                                    'LeftHip_y' : [],
                                    'RightHip_x' : [],
                                    'RightHip_y' : [],
                                    'LeftKnee_x' : [],
                                    'LeftKnee_y' : [],
                                    'RightKnee_x' : [],
                                    'RightKnee_y' : [],
                                    'LeftAnkle_x' : [],
                                    'LeftAnkle_y' : [],
                                    'RightAnkle_x' : [],
                                    'RightAnkle_y' : []})
        
        self.result = pd.DataFrame({'IsArmStretch' : [], 
                                    'IsLegStretch' : [],
                                    'IsLegClose' : [],
                                    'IsKneeStop' : [],
                                    'IsChinOver' : [],
                                    'PullUpCount' : []})
        
        self.try_count = 0 # pd.DataFrame({'TryCount' : []})
        self.try_array = []
        
        self.upgoing = True # 올라가는 중 여부 판단
        
        self.pullup_counter = 0 # 풀업 횟수 카운팅
        self.pullup_ready = 1 # upper_check 여러번 하지 않도록
        self.upper_ok = 0 # upper 확인 완료
        
        self.start_frame = 0 # 풀업 시도한 시작 Frame 
        self.end_frame = -1
        
        self.frame_counter = -1 # Frame 순서 카운팅
        self.end_condition = 10000
        
    def rule_check(self, frame, coordinates, face_detector):
        import pandas as pd
        import cv2
        # 1. 상체
        # 1.1. 어깨 - 팔꿈치 - 손목
        joint_idx = {'left_arm':[5, 7, 9], 'right_arm':[6,8,10]}
        for _, v in joint_idx.items():
            first = coordinates[v[0]]
            mid1 = coordinates[v[1]]
            end = coordinates[v[2]]

            angle_tmp1, mid1 = self.calculate_angle2D_3point(first, mid1, end)
            if angle_tmp1 >= self.arm_stretch_angle:
                self.IsArmStretch = 1
            else:
                self.IsArmStretch = 0
                break

        
        '''
        # 1.2. 손목 - 목 - 손목
        # ArmClose
        joint_idx = {'arm_between':[9, 5, 6, 10]}
        for _, v in joint_idx.items():
            first = coordinates[v[0]]
            mid_1 = coordinates[v[1]]
            mid_2 = coordinates[v[2]]
            end = coordinates[v[3]]
            angle_tmp2, mid2 =self.calculate_angle2D_4point(first ,mid_1, mid_2, end)
            if angle_tmp2 <= self.arm_between_angle:
                self.IsArmClose = 1
            else:
                self.IsArmClose = 0
                break
        '''
        
        # 2. 하체 
        # 2.1. 고관절-무릎-발목
        joint_idx = {'left_leg':[11, 13, 15], 'right_leg':[12, 14, 16]}
        for _, v in joint_idx.items():
            first = coordinates[v[0]]
            mid3 = coordinates[v[1]]
            end = coordinates[v[2]]

            angle_tmp3, mid3 = self.calculate_angle2D_3point(first ,mid3, end)
            if angle_tmp3 >= self.leg_stretch_angle:
                self.IsLegStretch = 1
            else:
                self.IsLegStretch = 0
                break

        # 2.2. 왼쪽 무릎 - 허리중심 - 오른쪽 무릎
        joint_idx = {'leg_between':[13, 11, 12, 14]} 
        for _, v in joint_idx.items():
            first = coordinates[v[0]]
            mid_1 = coordinates[v[1]]
            mid_2 = coordinates[v[2]]
            end = coordinates[v[3]]
            angle_tmp4, mid4 = self.calculate_angle2D_4point(first ,mid_1, mid_2, end)
            if angle_tmp4 <= self.leg_between_angle:
                self.IsLegClose = 1
            else:
                self.IsLegClose = 0
                break
        
        # 2.3. 무릎 y 값 vs 발목 y 값
        joint_idx = {'left_leg':[13, 15], 'right_leg':[14, 16]}
        for _, v in joint_idx.items():
            knee = coordinates[v[0]][:2]
            ankle = coordinates[v[1]][:2]
            knee_y = knee[1]
            ankle_y = ankle[1]
            if ankle_y <= knee_y:
                self.IsKneeStop = 0
                break
        
        # 3. 턱을 넘었는가?
        # 3.1. 예측 봉 위치
        left_wrist = coordinates[9][:2][1]
        right_wrist = coordinates[10][:2][1]
        bar_location = min(left_wrist,right_wrist)
        
        # 3.2. 예측 턱 위치
        # face detection
        try:
            faces = face_detector(frame, cv=True)
            face_box, _, _ = faces[0]
            chin_location = face_box[3]
            cv2.rectangle(frame, (int(face_box[0]), int(face_box[1])), (int(face_box[2]), int(face_box[3])), (255, 0,0), 3 )
        except:
            chin_location = bar_location+1
            pass
                
        '''
        joint_idx = {'left_eye_nose':[1, 0], 'right_eye_nose':[2,0]} 
        expected_chin = []
        for _, v in joint_idx.items():
            eye = coordinates[v[0]]
            nose = coordinates[v[1]]
            expected_chin.append(self.calculate_chin(eye ,nose))
        chin_location = min(expected_chin)[1]
        '''
        
        if chin_location <= bar_location:
            self.IsChinOver = 1
        else:
            self.IsChinOver = 0
        
        result_tmp = pd.DataFrame({'IsArmStretch' : [self.IsArmStretch],
                                'IsLegStretch' : [self.IsLegStretch],
                                'IsLegClose' : [self.IsLegClose],
                                'IsKneeStop' : [self.IsKneeStop],
                                'IsChinOver' : [self.IsChinOver],
                                'PullUpCount' : [self.pullup_counter]})
        
        anno_tmp = pd.DataFrame({'Nose_x' : [coordinates[0][0]],
                                'Nose_y' : [coordinates[0][1]], 
                                'LeftEye_x' : [coordinates[1][0]],
                                'LeftEye_y' : [coordinates[1][1]],
                                'RightEye_x' : [coordinates[2][0]],
                                'RightEye_y' : [coordinates[2][1]],
                                'LeftEar_x' : [coordinates[3][0]],
                                'LeftEar_y' : [coordinates[3][1]],
                                'RightEar_x' : [coordinates[4][0]],
                                'RightEar_y' : [coordinates[4][1]],
                                'LeftShoulder_x' : [coordinates[5][0]],
                                'LeftShoulder_y' : [coordinates[5][1]],
                                'RightShoulder_x' : [coordinates[6][0]],
                                'RightShoulder_y' : [coordinates[6][1]],
                                'LeftElbow_x' : [coordinates[7][0]],
                                'LeftElbow_y' : [coordinates[7][1]],
                                'RightElbow_x' : [coordinates[8][0]],
                                'RightElbow_y' : [coordinates[8][1]],
                                'LeftWrist_x' : [coordinates[9][0]],
                                'LeftWrist_y' : [coordinates[9][1]],
                                'RightWrist_x' : [coordinates[10][0]],
                                'RightWrist_y' : [coordinates[10][1]],
                                'LeftHip_x' : [coordinates[11][0]],
                                'LeftHip_y' : [coordinates[11][1]],
                                'RightHip_x' : [coordinates[12][0]],
                                'RightHip_y' : [coordinates[12][1]],
                                'LeftKnee_x' : [coordinates[13][0]],
                                'LeftKnee_y' : [coordinates[13][1]],
                                'RightKnee_x' : [coordinates[14][0]],
                                'RightKnee_y' : [coordinates[14][1]],
                                'LeftAnkle_x' : [coordinates[15][0]],
                                'LeftAnkle_y' : [coordinates[15][1]],
                                'RightAnkle_x' : [coordinates[16][0]],
                                'RightAnkle_y' : [coordinates[16][1]]})
        

        self.angle_list = [angle_tmp1, angle_tmp3, angle_tmp4]
        self.mid_list = [list(map(int,mid1)), list(map(int,mid3)), list(map(int,mid4))]   
        
        self.RightWrist_y = coordinates[10][1]
        
        self.result = pd.concat([self.result, result_tmp])
        self.anno_df = pd.concat([self.anno_df, anno_tmp])
        
        # 시각화
        self.visualize_angle(frame, self.angle_list, self.mid_list)
        
        # 종료조건
        if self.frame_counter == 0:
            self.end_condition = coordinates[8][1] # 오른 팔꿈치
        
        # return result_tmp, self.angle_list, self.mid_list, self.left_wrist_y
    
    def visualize_angle(self, frame, angle_list, mid_list):
        import cv2
        
        for angle, mid in zip(angle_list, mid_list):
            cv2.putText(frame, str(int(angle)),tuple(mid),cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255),4,cv2.LINE_AA)
    
    '''       
    # 어깨, 고관절, 무릎이 모두 올라가고 있으면 Up 
    def upgoing_check(self):
        from utils.logger import get_logger
        logger = get_logger()
        
        if self.frame_counter >= 2:
            temp_anno = self.anno_df[['RightHip_y', 'RightKnee_y']].tail(3)
            # 3개의 Frame을 비교해서 Up, Up
            if list(temp_anno.iloc[0] >= temp_anno.iloc[1]) == [True, True] and list(temp_anno.iloc[1] >= temp_anno.iloc[2]) == [True, True]:
                self.upgoing = True
            # Down, Up
            elif list(temp_anno.iloc[0] >= temp_anno.iloc[1]) == [False, False] and list(temp_anno.iloc[1] >= temp_anno.iloc[2]) == [True, True]:
                self.end_frame = self.frame_counter -2
                self.try_array.extend([self.try_count for _ in range(self.end_frame-self.start_frame+1)])            
                self.start_frame = self.frame_counter - 1 # reset
                # logger.info(f'Reset at {self.start_frame}')
                self.try_count += 1
            # (Down, Down) or (Up, Down)
            else: 
                self.upgoing = False
    ''' 
        
    def upper_check(self):
        from utils.logger import get_logger
        logger = get_logger()
        
        if self.result['IsArmStretch'].tail(5).sum() == 5 and self.pullup_ready ==1:
            self.upper_ok = 1
            self.pullup_ready = 0
            
            # 상체 조건 만족 시 리셋
            self.end_frame = self.frame_counter -2
            self.try_array.extend([self.try_count for _ in range(self.end_frame-self.start_frame+1)])
            self.start_frame = self.frame_counter - 1
            
            logger.info(f'Reset at {self.start_frame}')
            
            self.try_count += 1

    # 턱이 봉을 넘지 않으면 Pullup ready가 1인게 리셋이 안돼서 Try 횟수가 리셋이 안됨
    # 팔각도가 줄어들 때 강제 리셋?
    
    
    def lower_check(self):
        from utils.logger import get_logger
        logger = get_logger()
        
        if self.result['IsChinOver'].tail(5).sum() == 5 and self.upper_ok == 1:
            if self.result['IsLegStretch'][self.start_frame:].mean() >= 0.8 and self.result['IsLegClose'][self.start_frame:].mean() >= 0.8 and \
               self.result['IsKneeStop'][self.start_frame:].mean() >= 0.8:
            
                self.pullup_counter += 1
                self.upper_ok = 0
                self.pullup_ready = 1
                logger.info(f'Frame {self.frame_counter}에 풀업 횟수 : {self.pullup_counter}')
                
            # 넘었는데 다리가 안붙어 있었다면 초기화
            else:
                self.upper_ok = 0
                self.pullup_ready = 1
                logger.info(f'Frame {self.start_frame} ~ {self.frame_counter}에 다리가 안붙어있음')
        
        # 넘었는데 upper_ok 가 0일 경우    
        # elif self.result['IsChinOver'].tail(5).sum() == 5 and self.upper_ok == 0:
            # logger.info(f'Frame {self.start_frame} ~ {self.frame_counter}에 팔을 쭉 안폄')
        
        # 안넘고 팔도 쭉 안편경우는?
        
            
    def frame_count(self):
        self.frame_counter += 1
    
    def get_frame(self):
        return self.frame_counter
    
    def break_check(self):
        if self.RightWrist_y >= self.end_condition:
            return True
        else:
            return False

    def save_result(self):
        import pandas as pd
        
        self.try_array.extend([self.try_array[-1] for _ in range(self.anno_df.shape[0] - len(self.try_array))])
        try_df = pd.DataFrame({'TryCount' : self.try_array})
        
        # print(self.anno_df.shape, self.result.shape, try_df.shape)
        final_result = pd.concat([self.anno_df, self.result], axis=1).reset_index(drop=True)
        final_result = pd.concat([final_result, try_df], axis=1)
        
        final_result.to_csv('result.csv', index=False)

    def calculate_angle2D_3point(self, a, b, c):

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

    def calculate_angle2D_4point(self, a, b1, b2, c):

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

    def calculate_chin(self, a, b):
        import numpy as np
        chin_x = b[0]
        a = np.array(a[1]) # eye,  y 좌표
        b = np.array(b[1]) # nose, y 좌표
        
        eye_to_nose = abs(a-b)
        # x좌표는 nose, y좌표는 nose + 
        chin = [chin_x, b + round(eye_to_nose*1.618,2)]
        
        return chin
    





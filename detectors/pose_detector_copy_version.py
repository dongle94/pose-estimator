import os
import sys
import numpy as np

from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
os.chdir(ROOT)

from detectors.hrnet import HRNet

class PoseDetector(object):
    def __init__(self, cfg=None):
        # Keypoint Detector model configuration
        if os.path.abspath(cfg.KEPT_MODEL_PATH) != cfg.KEPT_MODEL_PATH:
            weight = os.path.abspath(os.path.join(ROOT, cfg.KEPT_MODEL_PATH))
        else:
            weight = os.path.abspath(cfg.KEPT_MODEL_PATH)
        device = cfg.DEVICE
        fp16 = cfg.KEPT_HALF
        img_size = cfg.KEPT_IMG_SIZE

        # Model load with weight
        self.detector = HRNet(weight=weight, device=device, img_size=img_size, fp16=fp16)

        # warm up
        self.detector.warmup(imgsz=(1, 3, img_size[0], img_size[1]))

    def preprocess(self, img, boxes):
        # boxes coords are ltrb
        inp, centers, scales =  self.detector.preprocess(img, boxes)
        return inp, centers, scales

    def detect(self, inputs):
        preds = self.detector.forward(inputs)

        return preds

    def postprocess(self, preds, centers, scales):
        preds = self.detector.postprocess(preds, centers, scales)

        return preds

def test():
    import time
    import cv2
    import pandas as pd
    import numpy as np
    from detectors.obj_detector import HumanDetector
    from utils.config import _C as cfg
    from utils.config import update_config
    from utils.medialoader import MediaLoader
    from utils.visualization import vis_pose_result
    from utils.tuck_rule import frame_check, visualize_angle
    from batch_face import RetinaFace

    
    update_config(cfg, args='./configs/config.yaml')
    print(cfg)

    result = pd.DataFrame({'IsArmStretch' : [], 
                            'IsArmClose' : [],
                            'IsLegStretch' : [],
                            'IsLegClose' : [],
                            'IsKneeStop' : [],
                            'IsChinOver' : []})
    frame_counter = -1
    pullup_counter = 0
    pullup_ready = 1
    start = 0
    start_frame = 0
    
    obj_detector = HumanDetector(cfg=cfg)
    kept_detector = PoseDetector(cfg=cfg)
    face_detector = RetinaFace(gpu_id=0)

    
    s = sys.argv[1]
    media_loader = MediaLoader(s)

    while media_loader.img is None:
        time.sleep(0.001)
        continue

    while media_loader.cap.isOpened():
        
        frame = media_loader.get_frame()
        width = int(media_loader.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(media_loader.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        end_con = height
            
        if frame is None:
            logger.info("Frame is None -- Break main loop")
            break
        else:
            frame_counter += 1

        im = obj_detector.preprocess(frame)
        pred = obj_detector.detect(im)
        pred, det = obj_detector.postprocess(pred)
        
        if det.size()[0]:
            
            inps, centers, scales = kept_detector.preprocess(frame, det)
            # inps : [1,3,384,288] 이미지 데이터
            preds = kept_detector.detect(inps)
            # preds : [1,17,96,72] 예측 관절좌표 (관절당 히트맵)
            rets = kept_detector.postprocess(preds, centers, scales)
            # rets : [1,17,3] (관절당 히트맵 중)
            
            # Frame마다 체크 
            result_tmp, angle_list, mid_list, left_wrist_y = frame_check(rets[0])
            # 결과 합침
            result = pd.concat([result, result_tmp])
            # 시각화           
            visualize_angle(frame, angle_list, mid_list)

            # 상체 펴짐 연속 5 Frame 지속 
            if result['IsArmStretch'].tail(5).sum() == 5 and result['IsArmClose'].tail(5).sum() == 5 and pullup_ready == 1:
                start = 1
                start_frame = frame_counter
                pullup_ready = 0
                end_con = mid_list[0][1]
                # 종료조건을 주기 위해 첫 풀업을 하고 난 뒤의 팔꿈치 위치를 저장
                #if pullup_counter==1 :
                    #end_con = mid_list[0][1]

            # 종료 조건
            if left_wrist_y >= end_con:
                logger.info("-- Hand Off from Bar --")
                break
                                          
            # 상체 확인 후 턱이 봉을 넘는다면
            if result['IsChinOver'].tail(5).sum() == 5 and start == 1:
                # 올라올 때 다리가 80% 이상 붙어있는지 
                if result['IsLegStretch'][start_frame:].mean() >= 0.8 and result['IsLegClose'][start_frame:].mean() >= 0.8 and result['IsKneeStop'][start_frame:].mean() >= 0.8:
                    pullup_counter += 1
                    start = 0
                    pullup_ready = 1
                    logger.info(f'Frame {frame_counter}에 풀업 횟수 : {pullup_counter}')
                # 넘었는데 다리가 안붙어 있었다면 초기화
                else:
                    start = 0
                    pullup_ready = 1
                    logger.info(f'Frame {start_frame} ~ {frame_counter}에 다리가 안붙어있음')
                            
            # print('풀업 횟수 : ', pullup_counter)
            
        else:
            rets = None


        # face detection
        try:
            faces = face_detector(frame, cv=True)
            face_box, landmarks, score = faces[0]
            print(face_box, landmarks, score)
            cv2.rectangle(frame, (int(face_box[0]), int(face_box[1])), (int(face_box[2]), int(face_box[3])), (255, 0,0), 3 )
        except:
            pass
        
        
        for d in det:
            x1, y1, x2, y2 = map(int, d[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (96, 96, 216), thickness=2, lineType=cv2.LINE_AA)
        
        # face
        #cv2.rectangle(frame, (int(face_box[0]), int(face_box[1])), (int(face_box[2]), int(face_box[3])), (255, 0,0), 3 )

        if rets is not None:
            frame = vis_pose_result(model=None, img=frame, result=rets)

        cv2.imshow('_', frame)

        if cv2.waitKey(1) == ord('q'):
            logger.info("-- CV2 Stop by Keyboard Input --")
            break
        
        time.sleep(0.001)
    # 모든 Frame 결과 저장
    result.to_csv('result.csv', index=False)
    media_loader.stop()
    logger.info("-- Stop program --")



if __name__ == "__main__":
    
    from utils.config import _C as _cfg
    from utils.config import update_config
    from utils.logger import init_logger, get_logger
    
    # get config
    update_config(_cfg, args='./configs/config.yaml')
    print(_cfg)

    # get logger
    init_logger(cfg=_cfg)
    logger = get_logger()
    
    test()

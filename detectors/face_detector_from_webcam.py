import dlib
import cv2

detector = dlib.get_frontal_face_detector()
webcam = cv2.VideoCapture(0)

import time

while (webcam.isOpened()):
    ret, img = webcam.read()
    print(ret, img)
    if ret == True:
        width = int(img.shape[1])
        height = int(img.shape[0])
        
        face_frame = cv2.resize(img, (width*0.5, height*0.5))
        rgb_image = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
        #dets = detector(rgb_image)
        t =time.time()
        
        
        dets, scores, subdetectors = detector.run(rgb_image, 1, 0)
        print(f"{time.time() - t:.4f}sec")
        # 기존 0.09 sec
        
        
        #for det in dets:
        #    cv2.rectangle(img, (det.left(), det.top()), (det.right(), det.bottom()), (255,0,0), 3)
        for i, det in enumerate(dets):
            cv2.rectangle(img, (det.left(), det.top()), (det.right(), det.bottom()), (255, 0,0), 3 )
            print("Detection {}, score: {}, face_type: {}".format(det, scores[i], subdetectors[i]))

        cv2.imshow("WEBCAM", img)

        if cv2.waitKey(1) == 27:
            break

webcam.release()
cv2.destroyAllWindows()
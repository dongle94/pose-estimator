import cv2

video_name = input('파일명을 입력하시오 : ')
print(video_name)
video = cv2.VideoCapture(video_name)
# Bunny.mp4, sq2.webm, squat.webm

if video.isOpened():
    
    fps = int(video.get(cv2.CAP_PROP_FPS)) # 영상의 프레임 받기
    # f_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    f_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    f_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    while True:
        ret, frame = video.read()
        if ret:
            re_frame = cv2.resize(frame, (round(f_width/2), round(f_height/2)))
            cv2.imshow('Video Sample', re_frame) # show
            # cv2.imshow('Video Sample', frame)
            key = cv2.waitKey(1000//fps) # 지연시간은 1000/숫자 로 30프레임으로 가정한 값
            
            if key == ord('q'): # ESC : 27
                break
        else:
            break
else:
    print("Can't Open video")

video.release() # 메모리 해제
cv2.destroyAllWindows() # 모든 창 닫기
    
# QObject::moveToThread: Current thread (0x145b500) is not the object's thread (0x163c110) 오류 해결하려면
# pip install --no-binary opencv-python opencv-python
    

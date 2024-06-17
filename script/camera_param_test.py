import os
import cv2
import time


SOURCE = 2
FOURCC = 'MJPG'
WIDTH = 1280
HEIGHT = 720
FPS = 30

SAVE_PATH = './data/CAM/'


cap = cv2.VideoCapture(SOURCE)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FOURCC))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FPS, FPS)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"-- Load Stream: {w}*{h}, FPS: {fps} --")

cap.set(cv2.CAP_PROP_BRIGHTNESS, 255)
# cap.set(cv2.CAP_PROP_GAIN, 0)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
cap.set(cv2.CAP_PROP_EXPOSURE, 36)
# cap.set(cv2.CAP_PROP_ISO_SPEED, 10)

brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
auto_exposure = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
iso_speed = cap.get(cv2.CAP_PROP_ISO_SPEED)
print(f"-- Brightness: {brightness} / Exposure: {exposure} / Auto_Expouser: {auto_exposure} / ISO_Speed: {iso_speed} --")

os.makedirs(SAVE_PATH, exist_ok=True)


frame_idx = 0
while cap.isOpened():
    st = time.time()
    frame_idx += 1

    cap.grab()
    ret, frame = cap.retrieve()

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    cv2.imwrite(os.path.join(SAVE_PATH, f"{frame_idx:04d}" + ".jpg"), frame)

    et = time.time()
    if frame_idx % 30 == 0:
        print(f"-- {et-st:.4f} seconds --")

cap.release()
cv2.destroyAllWindows()

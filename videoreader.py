import cv2
import os

path = '859931305-1-80.flv'
save_dir = 'frames'
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(path)

start = 300
i = -start
while (cap.isOpened() or i >= 1200):
    ret, frame = cap.read()
    if ret:
        if i >=0 :
            cv2.imwrite(os.path.join(save_dir, f'{i:06}.png'), frame)
        i += 1
    else:
        break
cap.release()
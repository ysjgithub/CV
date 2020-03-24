import numpy as np
from matplotlib import pyplot as plt


import os
import cv2 as cv
path = "./vedio/UCF101/"
filelist = os.listdir(path)

i = 0
filelist = [path+vedio for vedio in filelist][1:]

for vedio in filelist:
    cap = cv.VideoCapture(vedio)
    fps = cap.get(cv.CAP_PROP_FPS)
    frames = cap.get(cv.CAP_PROP_FRAME_COUNT)
    ret, frame = cap.read()
    cur_frame = 0
    print(fps)
    name = vedio.split('.')[0]
    while ret:
        if cur_frame % 5 == 0:
            outputPath = './image/UCF101/{}{}.jpg'.format(name,cur_frame//5)
            cv.imwrite(outputPath, frame)
        cur_frame += 1
        ret, frame = cap.read()
    cap.release()
cv.destroyAllWindows()





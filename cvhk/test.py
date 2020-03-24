import numpy as np
from matplotlib import pyplot as plt

import cv2 as cv

cv.namedWindow('frame')
a,b = -1,-1
def statics(event,x,y,flags,param):
    global a,b
    if event == cv.EVENT_LBUTTONUP:
        print(x,y,flags,param)
        a,b = x,y


cap = cv.VideoCapture('test.mp4')
cap.set(cv.CAP_PROP_FRAME_HEIGHT,60)
cap.set(cv.CAP_PROP_FRAME_WIDTH,60)

ps= []

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        # 写入已经翻转好的帧
        if a!=-1:
            ps.append( frame[a][b])
            print(len(ps))
        cv.imshow('frame',frame)

        cv.setMouseCallback('frame', statics)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
# 释放已经完成的工作
cap.release()
cv.destroyAllWindows()
print(ps)

ps = np.array(ps).reshape(len(ps),1,3)

print(ps.shape)
print(ps)

color = ('b','g','r')

for i,col in enumerate(color):
    histr = cv.calcHist([ps],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()





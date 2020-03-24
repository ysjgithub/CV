import numpy as np
import os
import cv2 as cv
path = "./image/UCF101/"
filelist = os.listdir(path)

i = 0
imgs = [path+vedio for vedio in filelist]
calcimgs = []
for i in range(1,len(imgs)):
    img = cv.imread(imgs[i], 0)
    calcimgs.append(img)


calcimgs = np.array(calcimgs)
s = calcimgs.mean(axis=0)
print(s)
# calcimgs = calcimgs- s
print(calcimgs)
# print(calcimgs)
calcimgs = [ calcimgs[i] - calcimgs[i-1] for i in range(1,len(calcimgs))]
# calcimgs = calcimgs.tolist()
for i in range(len(calcimgs)):
    cv.namedWindow('frame{}'.format(i))
    cv.imshow('frame{}'.format(i),calcimgs[i])

cv.waitKey(0)
cv.destroyAllWindows()



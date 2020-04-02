import numpy as np
import cv2 as cv
import time
t0 = time.time()

def get_similarity(hist1,hist2):
    similar = []
    for i in range(4096):
        if hist2[i] != 0:
            temp = hist1[i]/hist2[i]
            simi = np.sqrt(temp)

            similar.append(simi)
        else :
            similar.append(0)
    return similar

def calcWeights(w,h):
    # 计算距离带来的影响
    weights= np.zeros((w,h))
    H = (w**2+h**2)/4
    for i in range(w):
        for j in range(h):
            dis = (i-w/2)**2+(j-h/2)**2
            weights[i,j] = 1-dis/H
    return weights

def get_hist(rect,w,h):
    weights = calcWeights(h,w)
    print(np.sum(weights))
    C = 1/np.sum(weights)
    print(C)
    hist = np.zeros(4096)
    # 属于哪个q_u,哪个q_u就加上权重
    for col in range(h):
        for row in range(w):
            pixel = rect[col][row]/16
            qr,qg,qb = int(pixel[0]),int(pixel[1]),int(pixel[2])
            q_temp = 239*qr+16*qg+qb
            hist[q_temp]+=weights[col][row]
    return hist*C


def meanshiftstep(mywindow,hist0,frame):
    x0,y0,w,h = mywindow
    ite =0
    while ite<50:
        rect = frame[y0:y0 + h, x0:x0 + w]
        ite+=1
        print("得带次数{}".format(ite))
        shiftx, shifty = 0, 0
        hist1 = get_hist(rect,w,h)
        simlarity = get_similarity(hist0, hist1)
        for col in range(h):
            for row in range(w):
                pixel = rect[col][row]/16
                qr, qg, qb = int(pixel[0]), int(pixel[1]), int(pixel[2])
                q_temp = 239 * qr + 16 * qg + qb
                shiftx += simlarity[q_temp]*(row-w/2)
                shifty += simlarity[q_temp]*(col-h/2)
        if sum(simlarity)!=0:
            shiftx = shiftx/np.sum(simlarity)
            shifty = shifty/np.sum(simlarity)
        print(shiftx,shifty)
        x0+=shiftx
        y0+=shifty
        x0,y0 = int(x0),int(y0)
        # maxh,maxw = frame.shape[0],frame[1]
        # if shiftx**2+shifty**2<5:
        #     breakq
    return x0,y0,w,h

cap = cv.VideoCapture('video2.mp4')
start = None
fourcc = cv.VideoWriter_fourcc(*'DIVX')
out = cv.VideoWriter('output.avi', fourcc, 10.0, (428, 360))

while(cap.isOpened()):
    ret, frame = cap.read()
    weights = None
    if ret==True:
        # 写入已经翻转好的帧
        if start:
            x0,y0,w,h = meanshiftstep((x0,y0,w,h),hist0,frame)
            cv.rectangle(frame, (x0, y0), (x0+w, y0+h), 255, 3)
            out.write(frame)

        cv.imshow('frame',frame)
        if cv.waitKey(1) & 0xFF == ord('a'):
            mywindow = cv.selectROI('frame', frame, fromCenter=False)
            x0, y0, w, h = mywindow
            hist0 = get_hist(frame[y0:y0+h,x0:x0+w],w,h)
            cv.waitKey(0)
            start = True
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

out.release()

cap.release()

cv.destroyAllWindows()
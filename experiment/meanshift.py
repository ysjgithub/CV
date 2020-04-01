import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import cv2

import numpy as np
import cv2

# divide 256 gray scale into 64 bin from 0-63

def gray_64bin():
    bin = []
    count = 0

    for i in range(0,64):
        while(count<4):
            bin.append(i)

            count = count+1
        count = 0

    #print(bin)
    return bin
def gray_32bin():
    bin = []
    count = 0
    for i in range(0,32):
        while(count<8):
            bin.append(i)
            count = count+1
        count = 0
    return bin
def gray_16bin():
    bin = []
    count = 0
    for i in range(0,16):
        while(count<16):
            bin.append(i)
            count = count+1
        count = 0
    return bin
#print(gray_64bin())

def gray_bin():
    bin = []
    count = 0
    bin_num = 0
    for i in range(0,256):

        bin.append(i)
        #count = count+1
        #count = 0
        bin_num+=1
    #print(bin)
    return bin

def RGB_empty_hist():
    hist = []
    for i in range(0,4096):
        hist.append(0)
    #print(len(hist))
    return hist

#use rgb to get hist
def color_hist(img):
    h = img.shape[0]
    w = img.shape[1]
    hist = RGB_empty_hist()
    band = pow(1, 2) + pow(1, 2)
    wei_c = []
    for i in range(0,h):
        for j in range(0,w):
            qr = img[i][j][0]/16
            qg = img[i][j][1]/16
            qb = img[i][j][2]/16
            q_temp = qr*239+qg*16+qb
            #print(q_temp)
            q_temp = np.around(q_temp).astype(int)
            #print(i)

            dist = pow(i - 1, 2) + pow(j - 1, 2)
            wei = 1 - dist / band
            wei_c.append(wei)
            hist[q_temp]=hist[q_temp]+wei
    C = sum(wei_c)
    if C == 0:
        C = 1
    hist = [c_bin / C for c_bin in hist]
    return hist
#print(gray_bin())

def empty_hist():
    hist = []
    for i in range(0,16):
        hist.append(0)
    #print(len(hist))
    return hist

def get_hist(img):
    h = img.shape[0]
    w = img.shape[1]
    #print("wh in hist",w,h)

    bin = gray_16bin();
    hist = empty_hist()
    c_x = w/2
    c_y = h/2
    wei_c = []
    band = pow(c_x,2)+ pow(c_y,2)

    for col in range(0,h):
        for row in range(0,w):
            color = img[col][row]
            #print(color)
            color_bin = bin[color]
            #print(color_bin)
            dist = pow(col-c_y,2)+pow(row-c_x,2)
            wei = 1-dist/band
            wei_c.append(wei)
            hist[color_bin] = hist[color_bin] + wei
    C = sum(wei_c)
    #normalize hist
    hist=[c_bin / C for c_bin in hist]
    #print(len(hist))
    return hist

def get_similarity(hist1,hist2):
    similar = []
    for i in (range(0,4096)):
        if hist2[i] != 0:
            temp = hist1[i]/hist2[i]
            simi = np.sqrt(temp)

            similar.append(simi)
        else :
            similar.append(0)
    #print(similar)
    return similar

def meanshift_step(roi,roi_window,hist1,img):
    box_cx, box_cy, box_w, box_h = roi_window
    len = box_h*box_w
    num = 0
    sim = []
    # caculate 2 simularity
    # caculate new center
    while (num < 50):
        #print(num)
        x_shift = 0
        y_shift = 0
        sum_w = 0
        hist2 = color_hist(roi)
        similarity = get_similarity(hist1, hist2)
        s_mean=np.mean(similarity)
        sim.append(s_mean)
        print("simi", s_mean)
        num = num+1
        countt = 0
        for col in range(0, box_h):
            for row in range(0,box_w):

                #color meanshift
                qr = img[col][row][0] / 16
                qg = img[col][row][1] / 16
                qb = img[col][row][2] / 16
                #for each pixel find which bin it belongs to
                q_temp = qr * 239 + qg * 16 + qb
                q_temp = np.around(q_temp).astype(int)
                sum_w = sum_w + similarity[q_temp]
                # version 2

                #x_shift = row*similarity[color_bin]+x_shift
                #print("loop of x_shift",x_shift)
                #y_shift = col*similarity[color_bin]+y_shift

                #version 1
                #gray_shift
                # y_shift = y_shift + similarity[color_bin]*(col-box_h/2)
                # x_shift = x_shift + similarity[color_bin]*(row-box_w/2)
                y_shift = y_shift + similarity[q_temp] * (col - box_h / 2)
                x_shift = x_shift + similarity[q_temp] * (row - box_w / 2)

        if sum_w == 0:
            sum_w = 1
        y_shift = y_shift/sum_w
        x_shift = x_shift/sum_w

        #new center version 1

        box_cx = box_cx + x_shift
        box_cy = box_cy + y_shift

        box_cx = np.around(box_cx)
        box_cx = box_cx.astype(int)
        box_cy = np.around(box_cy)
        box_cy = box_cy.astype(int)


        roi = img[box_cy:box_cy + box_h, box_cx:box_cx + box_w]

    return box_cx,box_cy


cv.namedWindow('frame')
x0,y0,x1,y1 = -1,-1,-1,-1
w,h = -1,-1
cap = cv.VideoCapture('video2.mp4')
hist0 = None
start =False
h0,H0 = None,None
debug=False
mywindow = None

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        # 写入已经翻转好的帧
        if start:
            x0,y0 = meanshift_step(frame[y0:y1,x0:x1],mywindow,hist0,frame)
            x1=x0+w
            y1=y0+h
            print(x1,y1)
            cv.rectangle(frame, (x0, y0), (x1, y1), 255, 3)

        cv.imshow('frame',frame)

        if debug:
            cv.waitKey(0)
        if cv.waitKey(1) & 0xFF == ord('L'):
            debug=True
        if cv.waitKey(1) & 0xFF == ord('a'):
            mywindow = cv.selectROI('frame', frame, fromCenter=False)
            x0, y0, w, h = mywindow
            x1= x0+w
            y1= y0+h
            hist0 = color_hist(frame[y0:y1,x0:x1])
            cv.waitKey(0)

            start = True
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break
cap.release()

cv.destroyAllWindows()
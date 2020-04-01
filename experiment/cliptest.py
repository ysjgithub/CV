import torch
import numpy as np
from torch.utils.data import dataloader, dataset
import cv2 as cv
import torchvision.transforms as transforms
import re
import random
import os

def readfile(file_list):
    r = []
    for  path in file_list:
        with open(path) as f:
            s = f.readline()
            while s:
                r.append(re.split(r"[' ','\n']", s)[:-1])
                s = f.readline()
            f.close()
    return r

def clipframe(path,transform):
    pathv = "../vedio/UCF101/{}".format(path)
    pathp = "../vedio/UCF101pt/{}".format(path.split('.')[-2])
    cap = cv.VideoCapture(pathv)
    ret, frame = cap.read()
    cur_frame = 0
    if not os.path.exists(pathp):
        os.makedirs(pathp)
    while ret:
        # print(type(frame))
        if transform:
            frame = transform(frame)
        # print(type(frame))

        cv.imwrite("{}/{}.jpg".format(pathp,cur_frame),np.array(frame))
        cur_frame += 1
        ret, frame = cap.read()
    cap.release()

transform = transforms.Compose([
    transforms.ToPILImage(),
        transforms.RandomCrop((240,240)),
])

test_file = ['./file/testlist01.txt','./file/testlist02.txt','./file/testlist03.txt']

filelist = readfile(test_file)
print(len(filelist))
for i in range(len(filelist)):
    clipframe(filelist[i][0],transform)
    print(i)

# Training dataset


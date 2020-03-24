import torch.utils.data
import torch
import numpy as np
import os
import os.path
import cv2 as cv
import re
from torchvision import transforms


def readfile():
    r = []
    with open('../ucfTrainTestlist/trainlist01.txt') as f:
        i = 0
        s = f.readline()
        while s:
            r.append(re.split(r"[' ','\n']", s)[:-1])
            s = f.readline()
            i += 1
        f.close()
        return r
import random

def clipframe(path,transform):
    cap = cv.VideoCapture("../vedio/UCF101/{}".format(path))
    imgs = []
    ret, frame = cap.read()
    cur_frame = 0
    while ret:
        if cur_frame % 5 == 0:
            if transform:
                frame = transform(frame)
            imgs.append(frame)
        cur_frame += 1
        ret, frame = cap.read()
    cap.release()
    # print(len(imgs))
    imgs = random.sample(imgs,7)
    # print(s)
    return np.array(imgs)


class MyTrainData(torch.utils.data.Dataset):

    def __init__(self,transform=None):
        self.filelist = readfile()
        self.transform = transform
        # print(self.filelist)

    def __getitem__(self, idx):
        # print(self.filelist[idx])
        imgs = clipframe(self.filelist[idx][0],self.transform)
        # print(imgs.shape)
        imgs = torch.from_numpy(imgs).float()
        gt = torch.from_numpy(np.array([int(self.filelist[idx][1])])).float()
        return imgs, gt

    def __len__(self):
        return len(self.filelist)

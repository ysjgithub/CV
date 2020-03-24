import  torch.utils.data
import torch
import numpy as np
import os
import os.path
import  cv2 as cv
import re

def readfile():
    r = []
    with open('../ucfTrainTestlist/trainlist01.txt') as f:
        i = 0
        s = f.readline()
        while s:
            r.append(re.split(r'[" ","\n"]',s)[:-1])
            s = f.readline()
            i+=1
        f.close()
        return r

def clipframe(path):
    cap = cv.VideoCapture(path)
    imgs = []
    ret, frame = cap.read()
    cur_frame = 0
    while ret:
        if cur_frame % 5 == 0:
            imgs.append(frame)
        cur_frame += 1
        ret, frame = cap.read()
    cap.release()
    return np.array(imgs)

r  = readfile()
img = clipframe('../vedio/UCF101/v_ApplyEyeMakeup_g01_c01.avi')
print(torch.from_numpy(img).float().shape)


class MyTrainData(torch.utils.data.dataset):

    def __init__(self,root,transform=None,train=True):
        self.root = root
        self.train = train
        self.filelist = readfile()

    def __getitem__(self, idx):

        imgs = clipframe(self.dirs[idx])
        imgs = torch.from_numpy(imgs).float()
        gt = torch.from_numpy(self.dirs[idx].float())
        return imgs,gt

    def __len__(self):

        return len(self.filelist)

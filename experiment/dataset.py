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
    with open('../file/trainlist01.txt') as f:
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
                frame = transform(frame).numpy()
            imgs.append(frame)
        cur_frame += 1
        ret, frame = cap.read()
    cap.release()
    imgs = random.sample(imgs,7)
    return np.array(imgs).reshape(3,7,240,240)


class MyTrainData(torch.utils.data.Dataset):

    def __init__(self,transform=None):

        self.filelist = readfile()[0:10]
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

transform = transforms.Compose([
    transforms.ToPILImage(),
        transforms.RandomCrop(240),
        transforms.ToTensor(),
])
train_data = torch.utils.data.DataLoader(MyTrainData(transform),shuffle=True,num_workers=4,batch_size=10)
for x,y in train_data:
    print(x.shape,y.shape)
from torch import nn
def panel(input,output):
    return nn.Sequential(
        nn.Conv3d(input, output, 3, 1, 1),
        nn.BatchNorm3d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
    )

from torchsummary import summary
model = nn.Sequential(
    panel(3,64),
    panel(64,64),
    # panel(64,64)
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
summary(model,(3,7,240,240))
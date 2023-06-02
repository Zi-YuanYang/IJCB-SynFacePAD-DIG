from typing import Callable
import os
import os.path
from os.path import exists
import pandas as pd
import numpy as np
import cv2
from collections import defaultdict
import random

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.distributed as dist
import torch.utils.data
import torchvision

import albumentations
from albumentations.pytorch import ToTensorV2

import matplotlib.pyplot as plt


PRE__MEAN = [0.5, 0.5, 0.5]
PRE__STD = [0.5, 0.5, 0.5]

def ApplyWeightedRandomSampler(dataset_csv):
    dataframe = pd.read_csv(dataset_csv) # head: image_path, label
    # print(dataframe)
    class_counts = dataframe.label.value_counts()
    sample_weights = [1/class_counts[i] for i in dataframe.label.values]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(dataframe), replacement=True)
    return sampler

# map_size is for PixBis
class TrainDataset_withaug(Dataset):

    def __init__(self, csv_file, input_shape=(224, 224), map_size=14):
        print(csv_file)
        self.dataframe = pd.read_csv(csv_file)
        self.composed_transformations = albumentations.Compose([
            albumentations.Resize(height=input_shape[0], width=input_shape[1]),
            albumentations.HorizontalFlip(),
            albumentations.RandomGamma(gamma_limit=(80, 180)), # 0.5, 1.5
            albumentations.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20),
            albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0.1, p=0.5),
            albumentations.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            albumentations.Normalize(PRE__MEAN, PRE__STD, always_apply=True),
            ToTensorV2(),
        ])
        self.map_size = map_size

    def __len__(self):
        return len(self.dataframe)

    def get_labels(self):
        return self.dataframe.iloc[:, 1]

    def __getitem__(self, idx):

        img_path = self.dataframe.iloc[idx, 0]
        label_str = self.dataframe.iloc[idx, 1]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = 1 if label_str == 'bonafide' else 0
        map_x = torch.ones((self.map_size,self.map_size)) if label == 1 else torch.zeros((self.map_size, self.map_size))

        if label == 0 and random.randint(1,100) % 2 == 0:
        # if True:
            x_size, y_size = image.shape[:2]
            # print(image.shape[2])
            x_ = random.randint(1,x_size)
            y_ = random.randint(1,y_size)
            ann = [[x_,y_]]
            image = makeTarget(image,ann,x_size,y_size)
        # print('123')
        # save_plts(1,1,"asdf.jpg",image)
        image = self.composed_transformations(image = image)['image']

        # image.max() 
       
        return {
            "images": image,
            "labels": torch.tensor(label, dtype = torch.float),
            "map": map_x
        }


class TrainDataset_woaug(Dataset):

    def __init__(self, csv_file, input_shape=(224, 224), map_size=14):
        print(csv_file)
        self.dataframe = pd.read_csv(csv_file)
        self.composed_transformations = albumentations.Compose([
            albumentations.Resize(height=input_shape[0], width=input_shape[1]),
            albumentations.HorizontalFlip(),
            albumentations.RandomGamma(gamma_limit=(80, 180)),  # 0.5, 1.5
            albumentations.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20),
            albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0.1, p=0.5),
            albumentations.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            albumentations.Normalize(PRE__MEAN, PRE__STD, always_apply=True),
            ToTensorV2(),
        ])
        self.map_size = map_size

    def __len__(self):
        return len(self.dataframe)

    def get_labels(self):
        return self.dataframe.iloc[:, 1]

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        label_str = self.dataframe.iloc[idx, 1]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = 1 if label_str == 'bonafide' else 0
        map_x = torch.ones((self.map_size, self.map_size)) if label == 1 else torch.zeros(
            (self.map_size, self.map_size))

        # if label == 0 and random.randint(1, 100) % 2 == 0:
        #     # if True:
        #     x_size, y_size = image.shape[:2]
        #     # print(image.shape[2])
        #     x_ = random.randint(1, x_size)
        #     y_ = random.randint(1, y_size)
        #     ann = [[x_, y_]]
        #     image = makeTarget(image, ann, x_size, y_size)
        # print('123')
        # save_plts(1,1,"asdf.jpg",image)
        image = self.composed_transformations(image=image)['image']

        # image.max()

        return {
            "images": image,
            "labels": torch.tensor(label, dtype=torch.float),
            "map": map_x
        }


class TestDataset(Dataset):

    def __init__(self, csv_file, input_shape=(224, 224)):
        self.dataframe = pd.read_csv(csv_file)
        self.composed_transformations = albumentations.Compose([
            albumentations.Resize(height=input_shape[0], width=input_shape[1]),
            albumentations.Normalize(PRE__MEAN, PRE__STD, always_apply=True),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        label_str = self.dataframe.iloc[idx, 1]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = 1 if label_str == 'bonafide' else 0
        
        # if label == 0:
        # print(np.max(np.max(image)))            
        
        image = self.composed_transformations(image=image)['image']

        return {
            "images": image,
            "labels": torch.tensor(label, dtype = torch.float),
            "img_path": img_path
        }

# def makeTarget(inp,ann,x_size,y_size):
#         Target = []
#         t = np.zeros((1,x_size,y_size),dtype=float)
#         Target.append(t)
#         hsv=cv2.cvtColor(inp,cv2.COLOR_RGB2HSV)     #BGR模式转换为HSV模式
#         h,s,v=cv2.split(hsv)       #通道拆分
#         # v[:,:]=255                 #V通道置为255
#         # newHSV=cv2.merge([h,s,v])  #通道合并为新的HSV图片
#         for x_center,y_center in ann:
#             t = getGaussian(x_center,y_center,x_size=x_size, y_size=y_size)
#             # target = makeGaussian2(x_center, y_center, sigma_x = 10, sigma_y= 10,x_size=img_shape[1], y_size=img_shape[0])
#             t = np.expand_dims(t, 0)
#             Target.append(t)
            
#         Target = np.asarray(Target)
#         Target = np.sum(Target, 0)
#         Target = np.transpose(Target, (1,2,0))
#         newv = v+255*0.1*Target[:,:,0]
#         newv = newv.astype("uint8")
#         # newv[newv>=255]=255
#         new = cv2.merge([h,s,newv])
        
#         inp1 = cv2.cvtColor(new,cv2.COLOR_HSV2RGB) 
#         # for i in range(inp.shape[2]):
#         #     inp[:,:,i] = inp[:,:,i] + 255*Target[:,:,0] #* (1+Target[:,:,0]) 

#         # for x in range(x_size):
#         #     for y in range(y_size):
#         #         if inp[x,y,0]>=255 or inp[x,y,1]>=255 or inp[x,y,2] >= 255:
#         #             inp[x,y,0]=255
#         #             inp[x,y,1]=255
#         #             inp[x,y,2]=255
#             # inp[:,:,i][inp[:,:,i]>=255]=255
#             # inp[:,:,i][inp[:,:,i]<=0]=0
#         return inp1

def makeTarget(inp,ann,x_size,y_size):
        Target = []
        t = np.zeros((1,x_size,y_size),dtype=float)
        Target.append(t)
        for x_center,y_center in ann:
            t = getGaussian(x_center,y_center,x_size=x_size, y_size=y_size)
            # target = makeGaussian2(x_center, y_center, sigma_x = 10, sigma_y= 10,x_size=img_shape[1], y_size=img_shape[0])
            t = np.expand_dims(t, 0)
            Target.append(t)
        Target = np.asarray(Target)
        Target = np.sum(Target, 0)
        Target = np.transpose(Target, (1,2,0))
        inp = inp.astype("uint32")
        for i in range(inp.shape[2]):
            inp[:,:,i] = inp[:,:,i] + 255*Target[:,:,0]/2 #* (1+Target[:,:,0]) 
        
        # for x in range(x_size):
        #     for y in range(y_size):
        #         if inp[x,y,0]>=255 or inp[x,y,1]>=255 or inp[x,y,2] >= 255:
        #             inp[x,y,0]=255
        #             inp[x,y,1]=255
        #             inp[x,y,2]=255
            inp[:,:,i][inp[:,:,i]>=255]=255
            # inp[:,:,i][inp[:,:,i]<=0]=0
        return inp.astype("uint8")

def getGaussian(x_center=0, y_center=0,x_size=640, y_size=480):

    valid_range = random.randint(100,200)

    x_center = int(x_center)
    y_center = int(y_center)
    data = np.zeros((x_size+valid_range*2,y_size+valid_range*2),dtype=float)
    # print(x_center,y_center,data.shape,data[x_center:x_center+valid_range*2,y_center:y_center+valid_range*2].shape)
    # data[x_center:x_center+valid_range*2,y_center:y_center+valid_range*2] = calculated_Gaussian
    data[x_center:x_center+valid_range*2,y_center:y_center+valid_range*2] = makeGaussian2(x_center= valid_range, y_center= valid_range, sigma_x = random.randint(20,40), sigma_y= random.randint(20,40),x_size=valid_range*2, y_size=valid_range*2)
    # data[x_center:x_center+valid_range*2,y_center:y_center+valid_range*2] = np.ones((valid_range*2,valid_range*2))
    return data[valid_range:x_size+valid_range,valid_range:y_size+valid_range]

def makeGaussian2(x_center=0, y_center=0, theta=0, sigma_x = 100, sigma_y=100, x_size=640, y_size=480):

    # theta = 2*np.pi*theta/360
    x = np.arange(0,x_size, 1, float)
    y = np.arange(0,y_size, 1, float)
    y = y[:,np.newaxis]
    sx = sigma_x
    sy = sigma_y
    x0 = x_center
    y0 = y_center


    return np.exp(-(((x-x0)**2)/(2*(sx**2)) + ((y-y0)**2) /(2*(sy**2))))

def save_plts(row,col,fileName,*args,gray=False,**kwargs):
    plt.figure(dpi=300,figsize=(12,8))
    for i,item in enumerate(args):
        plt.subplot(row,col,i+1)
        if gray == True:
            plt.imshow(item,'gray')
            continue
        plt.imshow(item)
    plt.savefig(fileName) 
    plt.close('all')

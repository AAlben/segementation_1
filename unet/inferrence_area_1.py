import warnings
warnings.filterwarnings('ignore')

import os
import gc
import sys
import cv2
import json
import glob
import time
import numba
import random
import pathlib
import functools
import numpy as np
import pandas as pd
from tqdm import tqdm
from labelme import utils

from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.utils.data as D
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision
from torchvision import transforms as T
from torchvision.models.resnet import ResNet
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import Bottleneck

import albumentations as A
import rasterio
from rasterio.windows import Window
from pretrainedmodels.models.torchvision_models import pretrained_settings


import segmentation_models_pytorch as smp


label_dic_1 = {"_background_": 0,
               'br': 1,
               'na': 2,
               'ne': 3,
               'sh': 4,
               'ru': 5,
               'hi': 6,
               're': 7,
               'rf': 8,
               'lf': 9,
               'rb': 10,
               'lb': 11,
               'mn': 12,
               'ey': 13}
label_dic_2 = {"_background_": 0,
               "cow": 1}
use_labels = ['_background_', 'br', 'na', 'foot', 'mn', 'ey', 'hi', 're']


def set_seeds(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


set_seeds()


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class SelfTransform(object):

    def __init__(self):
        kernel = (5, 5)
        self.kernel = kernel

    def __call__(self, img):
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, self.kernel)
        return opening


class CowDataset(D.Dataset):

    def __init__(self):
        self.imgs, self.masks = [], []
        self.class_num = len(label_dic_1.keys()) - 1
        self.load()

        self.len = len(self.imgs)
        self.as_tensor = T.Compose([
            SelfTransform(),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def load(self):
        PATH = '/root/code/test_pytorch/takeoff_maskrcnn/train_bmp'
        for file in os.listdir(PATH):
            if '.bmp' not in file:
                continue

            json_path = os.path.splitext(file)[0] + '.json'
            if os.path.exists(json_path):
                continue

            img_path = os.path.join(PATH, file)
            img = cv2.imread(img_path)
            self.imgs.append(img)

            with open(json_path, 'rb') as f:
                mask_json = json.load(f)
            shapes = []
            for temp in mask_json['shapes']:
                if temp['label'] == 'whole':
                    continue
                shapes.append(temp)
            lbl, _ = utils.shapes_to_label(img.shape,
                                           shapes,
                                           label_dic_1)
            LBL = np.zeros((self.class_num, img.shape[0], img.shape[1]), dtype=np.float32)
            for k, v in label_dic_1.items():
                if v == 0:
                    continue
                where_index = np.where(lbl == v)
                LBL[v - 1, where_index[0], where_index[1]] = 1
            self.masks.append(LBL)

        return None

        PATH = '/root/code/test_pytorch/takeoff_maskrcnn/train_jpg'
        for file in os.listdir(PATH):
            if '.json' not in file:
                continue

            json_path = os.path.join(PATH, file)
            img_path = os.path.join(PATH, os.path.splitext(file)[0] + '.jpg')
            img = cv2.imread(img_path)
            self.imgs.append(img)

            with open(json_path, 'rb') as f:
                mask_json = json.load(f)
            shapes = []
            for temp in mask_json['shapes']:
                if temp['label'] == 'cow':
                    shapes.append(temp)
                    break
            lbl, _ = utils.shapes_to_label(img.shape,
                                           shapes,
                                           label_name_to_value_2)
            # LBL = np.zeros((self.class_num, img.shape[0], img.shape[1]))
            # for i in range(self.class_num):
            #     where_index = np.where(lbl == i)
            #     LBL[i, where_index[0], where_index[1]] = 1
            self.masks.append(lbl.astype(np.float32))

    def __getitem__(self, index):
        img, mask = self.imgs[index], self.masks[index]
        return self.as_tensor(img), mask

    def __len__(self):
        return self.len


def train(model, train_loader, loss_fn, optimizer):
    loss_ = nn.BCELoss()
    m = nn.Sigmoid()
    losses = []
    for i, (image, target) in enumerate(train_loader):
        image, target = image.to(DEVICE), target.float().to(DEVICE)
        optimizer.zero_grad()
        output = model(image)
        loss = loss_(m(output), target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.array(losses).mean()


def np_dice_score(probability, mask):
    p = probability.reshape(-1)
    t = mask.reshape(-1)

    p = p > 0.5
    t = t > 0.5
    uion = p.sum() + t.sum()

    overlap = (p * t).sum()
    dice = 2 * overlap / (uion + 0.001)
    return dice


def validation(model, val_loader, criterion):
    threshold = 0.6

    loss_ = nn.BCELoss()
    m = nn.Sigmoid()
    loss_l = []

    val_probability, val_mask = [], []
    model.eval()
    with torch.no_grad():
        for image, target in tqdm(val_loader):
            image, target = image.to(DEVICE), target.float().to(DEVICE)
            output = model(image)
            loss = loss_(m(output), target)
            loss_l.append(loss.item())

    #         output_ny = output.sigmoid().data.cpu().numpy()
    #         target_np = target.data.cpu().numpy()

    #         val_probability.append(output_ny)
    #         val_mask.append(target_np)

    # val_probability = np.concatenate(val_probability)
    # val_mask = np.concatenate(val_mask)

    # return np_dice_score(val_probability, val_mask)
    return np.mean(loss_l)


class SoftDiceLoss(nn.Module):

    def __init__(self, smooth=1., dims=(-2, -1)):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
        self.dims = dims

    def forward(self, x, y):
        tp = (x * y).sum(self.dims)
        fp = (x * (1 - y)).sum(self.dims)
        fn = ((1 - x) * y).sum(self.dims)
        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        dc = dc.mean()

        return 1 - dc


bce_fn = nn.BCEWithLogitsLoss()
# bce_fn = nn.BCELoss()
dice_fn = SoftDiceLoss()


def loss_fn(y_pred, y_true, ratio=0.8, hard=False):
    bce = bce_fn(y_pred, y_true)
    if hard:
        dice = dice_fn((y_pred.sigmoid()).float() > 0.5, y_true)
    else:
        dice = dice_fn(y_pred.sigmoid(), y_true)
    return ratio * bce + (1 - ratio) * dice


EPOCHES = 40
BATCH_SIZE = 8
NUM_WORKERS = 4

WINDOW = 1024
MIN_OVERLAP = 40
NEW_SIZE = 256


class SelfTransform(object):

    def __init__(self):
        kernel = (5, 5)
        self.kernel = kernel

    def __call__(self, img):
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, self.kernel)
        return opening

transform = T.Compose([
    SelfTransform(),
    T.ToTensor(),
    # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


model = smp.Unet(
    encoder_name="efficientnet-b1",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pretreined weights for encoder initialization
    in_channels=3,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
    classes=len(use_labels),                      # model output channels (number of classes in your dataset)
)
model_path = '/root/code/unet_cow_area/unet_best.pth'
model.load_state_dict(torch.load(model_path))
model.to(DEVICE)
model.eval()

threshold = 0.6
m = nn.Sigmoid()
with torch.no_grad():
    PATH = '/root/code/test_pytorch/takeoff_maskrcnn/train_bmp'
    for file in tqdm(os.listdir(PATH)):
        if '.bmp' not in file:
            continue

        json_path = os.path.splitext(file)[0] + '.json'
        if os.path.exists(json_path):
            continue

        img = cv2.imread(os.path.join(PATH, file))
        image = transform(img)
        image = image.to(DEVICE)[None]
        output = model(image)[0]
        output = m(output).cpu().numpy()

        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                mask[y, x] = np.argmax(output[:, y, x])

        with open(os.path.join(PATH, os.path.splitext(file)[0] + '.npy'), 'wb') as f:
            np.save(f, mask)

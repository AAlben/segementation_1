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


from losses import LovaszLossSoftmax
from losses import LovaszLossHinge
from losses import dice_coeff


label_dic_1 = {"_background_": 0,
               'br': 1,  # 乳房
               'na': 2,  # 腹壁
               'ne': 3,  # 脖子 - 颈部
               'sh': 4,  # 肩膀
               'ru': 5,  # 中间大圈
               'hi': 6,  # 腰
               're': 7,  # 臀
               'rf': 8,  # 右前
               'lf': 9,  # 左前
               'rb': 10,  # 右后
               'lb': 11,  # 左后
               'mn': 12,  # 口鼻
               'ey': 13}  # 眼睛

label_dic_2 = {"_background_": 0,
               "cow": 1}

label_dic_4 = {"_background_": 0,
               'sh': 1,
               'ru': 2}

use_labels = ['_background_', 'sh', 'ru']


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
        self.use_labels = use_labels
        self.class_num = len(label_dic_4.keys())
        self.load()

        self.len = len(self.imgs)
        self.as_tensor = T.Compose([
            SelfTransform(),
            T.ToTensor(),
            # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def load(self):

        PATH = '/root/code/model_data/train_bmp'
        for file in os.listdir(PATH):
            if '.json' not in file:
                continue

            json_path = os.path.join(PATH, file)
            img_path = os.path.join(PATH, os.path.splitext(file)[0] + '.bmp')
            img = cv2.imread(img_path)
            self.imgs.append(img)

            with open(json_path, 'rb') as f:
                mask_json = json.load(f)
            shapes = []
            for temp in mask_json['shapes']:
                if temp['label'] not in self.use_labels:
                    continue
                shapes.append(temp)
            lbl, _ = utils.shapes_to_label(img.shape,
                                           shapes,
                                           label_dic_4)
            # LBL = np.zeros((self.class_num, img.shape[0], img.shape[1]), dtype=np.float32)
            # for i, label in enumerate(self.use_labels):
            #     v = label_dic_3[label]
            #     where_index = np.where(lbl == v)
            #     LBL[i, where_index[0], where_index[1]] = 1
            self.masks.append(lbl)

        return None

        PATH = '/root/code/model_data/train_jpg'
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
            self.masks.append(lbl.astype(np.long))

    def __getitem__(self, index):
        img, mask = self.imgs[index], self.masks[index]
        return self.as_tensor(img), mask

    def __len__(self):
        return self.len


def train(model, train_loader, loss_fn, optimizer):
    losses = []
    for i, (image, target) in enumerate(train_loader):
        image, target = image.to(DEVICE), target.long().to(DEVICE)
        optimizer.zero_grad()
        output = model(image)
        loss = loss_fn(output, target)
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


def validation(model, val_loader, loss_fn):
    loss_l = []
    val_probability, val_mask = [], []
    model.eval()
    with torch.no_grad():
        for image, target in val_loader:
            image, target = image.to(DEVICE), target.long().to(DEVICE)
            output = model(image)
            loss = loss_fn(output, target)
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
cross_loss = nn.CrossEntropyLoss()
loss_f = LovaszLossSoftmax()


def loss_fn(y_pred, y_true, ratio=0.8, hard=False):
    bce = bce_fn(y_pred, y_true)
    if hard:
        dice = dice_fn((y_pred.sigmoid()).float() > 0.5, y_true)
    else:
        dice = dice_fn(y_pred.sigmoid(), y_true)
    return ratio * bce + (1 - ratio) * dice


EPOCHES = 70
BATCH_SIZE = 8
NUM_WORKERS = 4

WINDOW = 1024
MIN_OVERLAP = 40
NEW_SIZE = 256


ds = CowDataset()
ids = range(len(ds))
val_ids = random.sample(ids, int(len(ds) * 0.8))
train_ids = list(set(ids) - set(val_ids))
train_ds = D.Subset(ds, train_ids)
train_loader = D.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
valid_ds = D.Subset(ds, val_ids)
valid_loader = D.DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)


model = smp.Unet(
    encoder_name="efficientnet-b1",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pretreined weights for encoder initialization
    in_channels=3,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
    classes=len(label_dic_4.keys()),                      # model output channels (number of classes in your dataset)
)
model.to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
lr_step = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

header = r'''
        Train | Valid
Epoch |  Loss |  Dice (Best) | Time
'''

#          Epoch         metrics            time
raw_line = '{:6d}' + '\u2502{:7.4f}' * 3 + '\u2502{:6.2f}'

best_loss = 10
for epoch in tqdm(range(EPOCHES)):
    start_time = time.time()
    model.train()
    train_loss = train(model, train_loader, loss_f, optimizer)
    val_loss = validation(model, valid_loader, loss_f)
    lr_step.step(val_loss)

    print('epoch = %d; train_loss = %f; val_loss = %f' % (epoch, train_loss, val_loss))

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), '/root/code/model_state/unet_area2_best_0122.pth')

    # logging.info(raw_line.format(epoch, train_loss, val_dice, best_dice, (time.time() - start_time) / 60**1))

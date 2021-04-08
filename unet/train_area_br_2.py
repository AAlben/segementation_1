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
import albumentations as A
import segmentation_models_pytorch as smp
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.utils.data as D
from torch.utils.data import ConcatDataset
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision
from torchvision import transforms as T
from torchvision.models.resnet import ResNet
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import Bottleneck

import rasterio
from rasterio.windows import Window
from pretrainedmodels.models.torchvision_models import pretrained_settings

from losses import LovaszLossSoftmax
from losses import LovaszLossHinge
from losses import dice_coeff


label_dic = {"_background_": 0,
             'br': 1}

use_labels = ['_background_', 'br']


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


class Farm31Dataset(D.Dataset):

    def __init__(self):
        self.imgs, self.masks = [], []
        self.use_labels = use_labels
        self.class_num = len(label_dic.keys())
        self.load()

        self.len = len(self.imgs)
        self.as_tensor = T.Compose([
            SelfTransform(),
            T.ToTensor(),
        ])

    def load(self):
        PATH = '/data/data/train_bmp'
        for file in os.listdir(PATH):
            if '.json' not in file:
                continue

            json_path = os.path.join(PATH, file)
            img_path = os.path.join(PATH, os.path.splitext(file)[0] + '.bmp')
            img = cv2.imread(img_path)

            with open(json_path, 'rb') as f:
                mask_json = json.load(f)
            shapes = []
            for temp in mask_json['shapes']:
                if temp['label'] not in self.use_labels:
                    continue
                shapes.append(temp)
            if not shapes:
                continue
            lbl, _ = utils.shapes_to_label(img.shape,
                                           shapes,
                                           label_dic)
            self.masks.append(lbl)
            self.imgs.append(img)

    def __getitem__(self, index):
        img, mask = self.imgs[index], self.masks[index]
        return self.as_tensor(img), mask

    def __len__(self):
        return self.len


class Farm24Dataset(D.Dataset):

    def __init__(self):
        self.imgs, self.masks = [], []
        self.use_labels = use_labels
        self.class_num = len(label_dic.keys())
        self.resize_transform = A.Compose([
            A.LongestMaxSize(max_size=384)
        ])
        self.load()

        self.len = len(self.imgs)
        self.as_tensor = T.Compose([
            SelfTransform(),
            T.ToTensor(),
        ])

    def load(self):
        PATH = '/data/data/train_bmp'
        for file in os.listdir(PATH):
            if '.json' not in file:
                continue

            json_path = os.path.join(PATH, file)
            img_path = os.path.join(PATH, os.path.splitext(file)[0] + '.bmp')
            img = cv2.imread(img_path)

            with open(json_path, 'rb') as f:
                mask_json = json.load(f)
            shapes = []
            for temp in mask_json['shapes']:
                if temp['label'] not in self.use_labels:
                    continue
                shapes.append(temp)
            if not shapes:
                continue
            lbl, _ = utils.shapes_to_label(img.shape,
                                           shapes,
                                           label_dic)

            img_resize = self.resize_transform(image=img)
            img = img_resize['image']
            lbl = lbl.astype(np.uint8)
            lbl_resize = self.resize_transform(image=lbl)
            lbl = lbl_resize['image']

            self.imgs.append(img)
            self.masks.append(lbl)

        PATH = '/data/data/farm_24'
        for file in os.listdir(PATH):
            if '.json' not in file:
                continue

            json_path = os.path.join(PATH, file)
            img_path = os.path.join(PATH, os.path.splitext(file)[0] + '.bmp')
            img = cv2.imread(img_path)

            with open(json_path, 'rb') as f:
                mask_json = json.load(f)
            shapes = []
            for temp in mask_json['shapes']:
                if temp['label'] not in self.use_labels:
                    continue
                shapes.append(temp)
            if not shapes:
                continue
            lbl, _ = utils.shapes_to_label(img.shape,
                                           shapes,
                                           label_dic)
            self.masks.append(lbl)
            self.imgs.append(img)

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
    return np.mean(loss_l)


loss_f = LovaszLossSoftmax()

EPOCHES = 100
BATCH_SIZE = 2
NUM_WORKERS = 0

ds_1 = Farm31Dataset()
ds = ds_1
ids = range(len(ds))
val_ids = random.sample(ids, int(len(ds) * 0.8))
train_ids = list(set(ids) - set(val_ids))
train_ds = D.Subset(ds, train_ids)
train_loader = D.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
valid_ds = D.Subset(ds, val_ids)
valid_loader = D.DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

model = smp.Unet(
    encoder_name="efficientnet-b7",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pretreined weights for encoder initialization
    in_channels=3,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
    classes=len(label_dic.keys()),                      # model output channels (number of classes in your dataset)
)
model.to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
lr_step = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

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
        torch.save(model.state_dict(), '/data/model_state/unet_br_0311_1.pth')


ds_2 = Farm24Dataset()
ds = ds_2
ids = range(len(ds))
val_ids = random.sample(ids, int(len(ds) * 0.8))
train_ids = list(set(ids) - set(val_ids))
train_ds = D.Subset(ds, train_ids)
train_loader = D.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
valid_ds = D.Subset(ds, val_ids)
valid_loader = D.DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

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
        torch.save(model.state_dict(), '/data/model_state/unet_br_0311_2.pth')

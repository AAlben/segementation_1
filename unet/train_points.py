import warnings
warnings.filterwarnings('ignore')
from loguru import logger
logger.add('mask_rcnn.log', rotation='100 MB')

import os
import cv2
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from labelme import utils

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
import segmentation_models_pytorch as smp

from losses import dice_coeff
from losses import LovaszLossHinge
from losses import LovaszLossSoftmax


use_labels = ['_background_', '13', '38', '39', '40', '41', '44', '45', '1', '58', '48', '52', '50']
label_d = {"_background_": 0,
           '1': 1,
           '13': 2,
           '38': 3,
           '39': 4,
           '40': 5,
           '41': 6,
           '44': 7,
           '45': 8,
           '48': 9,
           '50': 10,
           '52': 11,
           '58': 12}


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


class FarmDataset(D.Dataset):

    def __init__(self):
        self.imgs, self.masks = [], []
        self.use_labels = use_labels
        self.class_num = len(label_d.keys())
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
            foot, shapes = [], []
            for temp in mask_json['shapes']:
                if temp['label'] not in use_labels:
                    continue
                shapes.append(temp)
            lbl, _ = utils.shapes_to_label(img.shape,
                                           shapes,
                                           label_d)
            if not shapes:
                continue
            self.imgs.append(img)
            self.masks.append(lbl)

    def __getitem__(self, index):
        img, mask = self.imgs[index], self.masks[index]
        return self.as_tensor(img), mask

    def __len__(self):
        return self.len


def train(model, train_loader, loss_fn, optimizer):
    model.train()
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
BATCH_SIZE = 5
NUM_WORKERS = 0

ds = FarmDataset()
ids = range(len(ds))
val_ids = random.sample(ids, int(len(ds) * 0.8))
train_ids = list(set(ids) - set(val_ids))
train_ds = D.Subset(ds, train_ids)
train_loader = D.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
valid_ds = D.Subset(ds, val_ids)
valid_loader = D.DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

model = smp.Unet(
    encoder_name="efficientnet-b5",
    encoder_weights="imagenet",
    in_channels=3,
    classes=len(label_d.keys()),
)
model.to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
lr_step = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

for epoch in tqdm(range(EPOCHES)):
    train_loss = train(model, train_loader, loss_f, optimizer)
    val_loss = validation(model, valid_loader, loss_f)
    lr_step.step(train_loss)
    content = f"epoch: {epoch:3d}; train_loss = {train_loss}; val_loss = {val_loss}"
    logger.info(content)
    torch.save(model.state_dict(), '/root/code/model_state/unet/unet_point_0415_%d.pth' % epoch)

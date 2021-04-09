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
import torch.utils.data as D
import torchvision
from torchvision import transforms as T
import albumentations as A
import segmentation_models_pytorch as smp

from losses import dice_coeff
from losses import LovaszLossHinge
from losses import LovaszLossSoftmax


label_dic = {"_background_": 0,
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

label_dic_4 = {"_background_": 0,
               'br': 1,  # 乳房
               'na': 2,  # 腹壁
               'foot': 3,
               'mn': 4,  # 口鼻
               'ey': 5,  # 眼睛
               'hi': 6,  # 腰
               're': 7}  # 臀

use_labels = ['_background_', 'na', 'lf', 'rf', 'rb', 'lb', 'mn', 'ey', 'hi', 're']


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
        self.resize_transform = A.Compose([
            A.LongestMaxSize(max_size=640)
        ])
        self.imgs, self.masks = [], []
        self.use_labels = use_labels
        self.class_num = len(label_dic_4.keys())
        self.load()

        self.len = len(self.imgs)
        self.as_tensor = T.Compose([
            SelfTransform(),
            T.ToTensor(),
        ])

    def load(self):
        PATH = '/root/code/model_data/train_bmp'
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
                if temp['label'] in ['lf', 'rf', 'lb', 'rb']:
                    temp['label'] = 'foot'
                    shapes.append(temp)
                    continue
                shapes.append(temp)
            lbl, _ = utils.shapes_to_label(img.shape,
                                           shapes,
                                           label_dic_4)
            if not shapes:
                continue

            self.imgs.append(img)
            self.masks.append(lbl)

        PATH = '/root/code/model_data/farm_24'
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
                if temp['label'] in ['lf', 'rf', 'lb', 'rb']:
                    temp['label'] = 'foot'
                    shapes.append(temp)
                    continue
                shapes.append(temp)
            lbl, _ = utils.shapes_to_label(img.shape,
                                           shapes,
                                           label_dic_4)
            if not shapes:
                continue

            img_resize = self.resize_transform(image=img)
            img = img_resize['image']
            lbl = lbl.astype(np.uint8)
            lbl_resize = self.resize_transform(image=lbl)
            lbl = lbl_resize['image']

            self.imgs.append(img)
            self.masks.append(lbl)

    def __getitem__(self, index):
        img, mask = self.imgs[index], self.masks[index]
        return self.as_tensor(img), mask

    def __len__(self):
        return self.len


class Farm24Dataset(D.Dataset):

    def __init__(self):
        self.imgs, self.masks = [], []
        self.use_labels = use_labels
        self.class_num = len(label_dic_4.keys())
        self.resize_transform = A.Compose([
            A.SmallestMaxSize(max_size=384)
        ])
        self.load()
        self.len = len(self.imgs)
        self.as_tensor = T.Compose([
            SelfTransform(),
            T.ToTensor(),
        ])

    def load(self):
        PATH = '/root/code/model_data/train_bmp'
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
                if temp['label'] in ['lf', 'rf', 'lb', 'rb']:
                    temp['label'] = 'foot'
                    shapes.append(temp)
                    continue
                shapes.append(temp)
            lbl, _ = utils.shapes_to_label(img.shape,
                                           shapes,
                                           label_dic_4)
            if not shapes:
                continue
            img_resize = self.resize_transform(image=img)
            img = img_resize['image']
            lbl = lbl.astype(np.uint8)
            lbl_resize = self.resize_transform(image=lbl)
            lbl = lbl_resize['image']
            self.imgs.append(img)
            self.masks.append(lbl)

        PATH = '/root/code/model_data/farm_24'
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
                if temp['label'] in ['lf', 'rf', 'lb', 'rb']:
                    temp['label'] = 'foot'
                    shapes.append(temp)
                    continue
                shapes.append(temp)
            lbl, _ = utils.shapes_to_label(img.shape,
                                           shapes,
                                           label_dic_4)
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
    losses = []
    model.train()
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
NUM_WORKERS = 4

ds_1 = Farm31Dataset()
# ds_2 = Farm24Dataset()
# ds = ConcatDataset([ds_1, ds_2])
ds = ds_1
ids = range(len(ds))
val_ids = random.sample(ids, int(len(ds) * 0.8))
train_ids = list(set(ids) - set(val_ids))
train_ds = D.Subset(ds, train_ids)
train_loader = D.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
valid_ds = D.Subset(ds, val_ids)
valid_loader = D.DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)


model = smp.Unet(
    encoder_name="efficientnet-b5",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pretreined weights for encoder initialization
    in_channels=3,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
    classes=len(label_dic_4.keys()),                      # model output channels (number of classes in your dataset)
)
model.to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
lr_step = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

for epoch in tqdm(range(EPOCHES)):
    train_loss = train(model, train_loader, loss_f, optimizer)
    val_loss = validation(model, valid_loader, loss_f)
    lr_step.step(val_loss)

    content = f'epoch = {epoch}; train_loss = {train_loss}; val_loss = {val_loss}'
    print(content)
    logger.info(content)

    torch.save(model.state_dict(), '/root/code/model_state/unet/unet_area_1_0409_%d.pth' % epoch)

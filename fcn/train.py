import os
import cv2
import json
import random
import numpy as np
from tqdm import tqdm
from labelme import utils

import warnings
warnings.filterwarnings('ignore')

import numba

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D
import torchvision
from torchvision import transforms as T


label_name_to_value_1 = {"_background_": 0,
                         "whole": 1}
label_name_to_value_2 = {"_background_": 0,
                         "cow": 1}


def set_seeds(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seeds()

EPOCHES = 40
NUM_WORKERS = 4
BATCH_SIZE = 10

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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
        self.class_num = 1
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
                if temp['label'] == 'whole':
                    shapes.append(temp)
                    break
            lbl, _ = utils.shapes_to_label(img.shape,
                                           shapes,
                                           label_name_to_value_1)
            # LBL = np.zeros((self.class_num, img.shape[0], img.shape[1]))
            # for i in range(self.class_num):
            #     where_index = np.where(lbl == i)
            #     LBL[i, where_index[0], where_index[1]] = 1
            self.masks.append(lbl.astype(np.float32))

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
        return self.as_tensor(img), mask[None]

    def __len__(self):
        return self.len


ds = CowDataset()
ids = range(len(ds))
val_ids = random.sample(ids, int(len(ds) * 0.8))
train_ids = list(set(ids) - set(val_ids))
train_ds = D.Subset(ds, train_ids)
train_loader = D.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
valid_ds = D.Subset(ds, val_ids)
valid_loader = D.DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

model = torchvision.models.segmentation.fcn_resnet50(pretrained=True,
                                                     progress=True,
                                                     num_classes=21,
                                                     aux_loss=None)
model.classifier[4] = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
model.to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(),
                              lr=1e-4, weight_decay=1e-3)
loss_fn = nn.BCEWithLogitsLoss()


@torch.no_grad()
def validation(model, loader, loss_fn):
    losses = []
    model.eval()
    for image, target in tqdm(loader):
        image, target = image.to(DEVICE), target.float().to(DEVICE)
        output = model(image)['out']
        loss = loss_fn(output, target)
        losses.append(loss.item())
    return np.array(losses).mean()


best_loss = 10

for epoch in tqdm(range(1, EPOCHES + 1)):
    losses = []
    model.train()
    for image, target in tqdm(train_loader):
        image, target = image.to(DEVICE), target.float().to(DEVICE)
        optimizer.zero_grad()
        output = model(image)['out']
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print('')
        print('-' * 100)
        print('train loss = %f' % loss)
        print('-' * 100)
    vloss = validation(model, valid_loader, loss_fn)
    print('')
    print('-' * 100)
    print('valid loss = %f' % vloss)
    print('-' * 100)
    if vloss < best_loss:
        best_loss = vloss
        torch.save(model.state_dict(), 'model_best_0114_2.pth')

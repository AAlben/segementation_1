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


from canny_1 import execute as canny_execute

NUM_WORKERS = 4
BATCH_SIZE = 5

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


model = torchvision.models.segmentation.fcn_resnet50(pretrained=True,
                                                     progress=True,
                                                     num_classes=21,
                                                     aux_loss=None)
model.classifier[4] = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
model.to(DEVICE)
model.load_state_dict(torch.load("./model_best_0114_2.pth"))
model.eval()


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
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

i = 0

with torch.no_grad():
    PATH = '/root/code/test_pytorch/takeoff_maskrcnn/train_bmp'
    PATH = '/root/code/test_pytorch/takeoff_maskrcnn/train_jpg'
    for file in tqdm(os.listdir(PATH)):
        if '.bmp' not in file and '.jpg' not in file:
            continue

        json_path = os.path.splitext(file)[0] + '.json'
        if os.path.exists(json_path):
            continue

        img_file = os.path.join(PATH, file)
        interval_points = canny_execute(img_file)
        if not interval_points:
            continue

        # print(interval_points)
        y_start, y_end, x_start, x_end = interval_points
        img = cv2.imread(img_file)[y_start:y_end, x_start:x_end]
        image = transform(img)
        iamge = image.to(DEVICE)[None]
        predict = model(iamge)['out'][0]
        predict = predict.sigmoid().cpu().numpy().astype(np.float32)[0]
        mask = predict

        # mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
        # for y in range(mask.shape[0]):
        #     for x in range(mask.shape[1]):
        #         mask[y, x] = np.argmax(predict[:, y, x])
        with open(os.path.join(PATH, os.path.splitext(file)[0] + '.npy'), 'wb') as f:
            np.save(f, mask)

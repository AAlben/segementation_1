import os
import cv2
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from labelme import utils as labelme_utils

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import warnings
warnings.filterwarnings('ignore')


class SelfTransform(object):

    def __init__(self):
        kernel = (5, 5)
        self.kernel = kernel

    def __call__(self, img):
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, self.kernel)
        return opening


transform_ = T.Compose([
    SelfTransform(),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

model_path = '/root/code/model_state/faster_rcnn_kaggle_0218.pth'
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 2  # 1 class (person) + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load(model_path))
model = model.to(device)
model.eval()

with torch.no_grad():
    image = transform_(img)
    image = image.to(device)[None]
    output = model(image)[0]
    print(output)

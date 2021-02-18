import os
import cv2
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from labelme import utils as labelme_utils

import torch
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


transforms = T.Compose([
    SelfTransform(),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 2  # 1 class (person) + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load('/root/code/model_state/faster_rcnn_kaggle.pth'))
model.eval()
model = model.to(device)


results = []
with torch.no_grad():
    PATH = '/root/code/model_data/train_bmp'
    for file in tqdm(os.listdir(PATH)[:100]):
        if '.bmp' not in file:
            continue

        json_path = os.path.splitext(file)[0] + '.json'
        if os.path.exists(os.path.join(PATH, json_path)):
            continue

        # print(os.path.join(PATH, file))
        img = cv2.imread(os.path.join(PATH, file))
        image = transforms(img)
        image = image.to(device)[None]
        output = model(image)[0]
        boxes = output['boxes'].cpu().numpy()
        results.append([os.path.join(PATH, file), boxes[0]])
print(results)

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


class CowDataset(torch.utils.data.Dataset):

    def __init__(self, transforms=None):
        self.transforms = T.Compose([
            SelfTransform(),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.masks = []
        self.imgs = []
        self.load()

    def load(self):
        PATHS = ['/root/code/model_data/train_bmp',
                 '/root/code/model_data/train_jpg']

        for PATH in PATHS:
            for file in os.listdir(PATH):
                if '.json' not in file:
                    continue
                json_path = os.path.join(PATH, file)
                has_tag = False
                with open(json_path, 'rb') as f:
                    mask_json = json.load(f)
                    for mask in mask_json['shapes']:
                        if mask['label'] in ['whole', 'cow']:
                            has_tag = True
                            break
                if not has_tag:
                    continue

                img_file = os.path.splitext(file)[0] + '.bmp'
                img_path = os.path.join(PATH, img_file)
                if not os.path.exists(img_path):
                    img_file = os.path.splitext(file)[0] + '.jpg'
                    img_path = os.path.join(PATH, img_file)

                self.imgs.append(img_path)
                self.masks.append(json_path)

    def __getitem__(self, idx):
        img_path, mask_path = self.imgs[idx], self.masks[idx]
        img = cv2.imread(img_path)
        with open(mask_path, 'rb') as f:
            mask_json = json.load(f)

        mask_shapes = []
        if '.jpg' in img_path:
            mask_shapes.append([mask_json["shapes"][0]])
            label_d = {"_background_": 0, 'cow': 1}
        elif '.bmp' in img_path:
            for mask in mask_json['shapes']:
                if mask['label'] != 'whole':
                    continue
                mask_shapes.append([mask])
                break
            label_d = {"_background_": 0, 'whole': 1}

        boxes = []
        lbl, _ = labelme_utils.shapes_to_label(img.shape,
                                               mask_shapes[0],
                                               label_d)
        nonzero_idx = np.nonzero(lbl)
        xmin = np.min(nonzero_idx[1])
        xmax = np.max(nonzero_idx[1])
        ymin = np.min(nonzero_idx[0])
        ymax = np.max(nonzero_idx[0])
        boxes.append([xmin, ymin, xmax, ymax])
        boxes = np.asarray(boxes)
        labels = np.ones(len(boxes))

        targets = {}
        targets['boxes'] = torch.from_numpy(boxes)
        targets['labels'] = torch.from_numpy(labels).type(torch.int64)
        img = self.transforms(img)
        return img, targets

    def __len__(self):
        return len(self.imgs)


class IoULoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        p_xmin, p_ymin, p_xmax, p_ymax = inputs
        xmin, ymin, xmax, ymax = targets
        u_xmin, u_ymin, u_xmax, u_ymax = max(p_xmin, xmin),  max(p_ymin, ymin), min(p_xmax, xmax), min(p_ymax, ymax)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (u_xmax - u_xmin) * (u_ymax - u_ymin)
        total = (p_xmax - p_xmin) * (p_ymax - p_ymin) + (xmax - xmin) * (ymax - ymin)
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return 1 - IoU


EPOCHS = 100
NUM_WORKERS = 4
BATCH_SIZE = 10

torch.manual_seed(1)
dataset = CowDataset()
indices = torch.randperm(len(dataset)).tolist()
split_index = int(len(dataset) * 0.8)
dataset_train = torch.utils.data.Subset(dataset, indices[:split_index])
dataset_test = torch.utils.data.Subset(dataset, indices[split_index:])
data_loader_train = torch.utils.data.DataLoader(dataset_train,
                                                num_workers=NUM_WORKERS,
                                                batch_size=BATCH_SIZE,
                                                shuffle=True,
                                                collate_fn=lambda x: list(zip(*x)))
data_loader_test = torch.utils.data.DataLoader(dataset_test,
                                               num_workers=NUM_WORKERS,
                                               batch_size=BATCH_SIZE,
                                               shuffle=False,
                                               collate_fn=lambda x: list(zip(*x)))


loss_fn = IoULoss()
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 2  # 1 class (person) + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model = model.to(device)


params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005)


now_loss = 10
for epoch in tqdm(range(EPOCHS)):
    model.train()
    for images, targets in data_loader_train:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    model.eval()
    loss_ = []
    with torch.no_grad():
        for images, target in data_loader_test:
            for i, image in enumerate(images):
                image = image.to(device)
                output = model([image])[0]
                Y = target[i]['boxes'][0].to(device)
                loss_.append(loss_fn(output['boxes'][0], Y).cpu())

    loss__ = abs(np.mean(loss_))
    if loss__ < now_loss:
        now_loss = loss__
        torch.save(model.state_dict(), '/root/code/model_state/faster_rcnn_kaggle_0219.pth')
    print("Loss = {:.4f} ".format(loss__))

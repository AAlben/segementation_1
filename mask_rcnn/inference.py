import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from labelme import utils as labelme_utils

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import utils
import transforms as T
from torchvision.transforms import functional as F
from engine import train_one_epoch, evaluate


IMGS = []


class CowDataset(object):

    def __init__(self, transforms):
        self.transforms = transforms
        self.imgs, self.masks = [], []
        self.load()

    def load(self):
        for file in os.listdir('train'):
            if '.jpg' not in file:
                continue

            json_path = os.path.splitext(file)[0] + '.json'
            if os.path.exists(json_path):
                continue

            img_path = os.path.join('train', file)
            self.imgs.append(img_path)
            IMGS.append(img_path)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = cv2.imread(img_path)
        target = {}
        img, target = F.to_tensor(img), target

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and person
num_classes = 2
model = get_model_instance_segmentation(num_classes)
model.load_state_dict(torch.load("./mask_rcnn_model_10.pth"))
model.eval()
model.to(device)


class SelfTransform(object):

    def __init__(self):
        kernel = (5, 5)
        self.kernel = kernel

    def __call__(self, img):
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, self.kernel)
        return opening


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


dataset_all = CowDataset(get_transform(train=False))


with torch.no_grad():
    for i, (img, _) in tqdm(enumerate(dataset_all)):
        prediction = model([img.to(device)])[0]
        mask = prediction['masks'][0]
        mask = mask.mul(255).byte().cpu().numpy()
        file = os.path.splitext(IMGS[i])[0] + '_mask.npy'
        print(file)
        with open(file, 'wb') as f:
            np.save(f, mask)

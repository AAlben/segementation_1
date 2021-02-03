import os
import cv2
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from labelme import utils as labelme_utils

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import utils
import transforms as T
from engine import train_one_epoch, evaluate


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# 1 class (person) + background
num_classes = 2
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load("/root/code/model_state/faster_rcnn_5.pth"))
model.eval()
model.to(device)


class CowDataset(object):

    def __init__(self, transforms):
        self.transforms = transforms
        self.imgs, self.masks = [], []
        self.load()

    def load(self):
        PATH = '/root/code/model_data/train_bmp'
        for file in os.listdir(PATH):
            if '.json' not in file:
                continue

            json_path = os.path.join(PATH, file)
            img_file = os.path.splitext(file)[0] + '.bmp'
            img_path = os.path.join(PATH, img_file)

            self.imgs.append(img_path)
            self.masks.append(json_path)
        return None

        PATH = '/root/code/model_data/train_jpg'
        for file in os.listdir(PATH):
            if '.json' not in file:
                continue

            json_path = os.path.join(PATH, file)
            img_file = os.path.splitext(file)[0] + '.jpg'
            img_path = os.path.join(PATH, img_file)

            self.imgs.append(img_path)
            self.masks.append(json_path)

    def __getitem__(self, idx):
        img_path, mask_path = self.imgs[idx], self.masks[idx]
        img = cv2.imread(img_path)

        with open(mask_path, 'rb') as f:
            mask_json = json.load(f)

        if '.jpg' in img_path:
            mask_shapes = []
            mask_shapes.append([mask_json["shapes"][0]])
            label_name_to_value = {"_background_": 0, 'cow': 1}
        elif '.bmp' in img_path:
            mask_shapes = []
            for mask in mask_json['shapes']:
                if mask['label'] != 'whole':
                    continue
                mask_shapes.append([mask])
                break
            label_name_to_value = {"_background_": 0, 'whole': 1}

        num_objs = 1
        boxes = []
        masks = []
        for mask in mask_shapes:
            lbl, _ = labelme_utils.shapes_to_label(img.shape,
                                                   mask,
                                                   label_name_to_value)
            nonzero_idx = np.nonzero(lbl)
            xmin = np.min(nonzero_idx[1])
            xmax = np.max(nonzero_idx[1])
            ymin = np.min(nonzero_idx[0])
            ymax = np.max(nonzero_idx[0])
            boxes.append([xmin, ymin, xmax, ymax])
            masks.append(lbl)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        # print('-' * 100)
        # print(img.shape)
        # print(target)
        # print('-' * 100)

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_transform():
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

results = []
transform_ = get_transform()
with torch.no_grad():
    PATH = '/root/code/model_data/train_bmp'
    for file in tqdm(os.listdir(PATH)[:20]):
        if '.bmp' not in file and '.jpg' not in file:
            continue

        json_path = os.path.splitext(file)[0] + '.json'
        print(json_path)
        if not os.path.exists(os.path.join(PATH, json_path)):
            continue

        img = cv2.imread(os.path.join(PATH, file))
        image, _ = transform_(img, {})
        image = image.to(device)[None]
        output = model(image)[0]
        boxes = output['boxes'].cpu().numpy()
        results.append([os.path.join(PATH, file), boxes[0]])
print(results)

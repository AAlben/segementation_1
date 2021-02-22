import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from labelme import utils as labelme_utils

import torch
import torchvision
from torchvision import transforms as TT
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import utils
import transforms as T
from engine import train_one_epoch, evaluate


class CowDataset(object):

    def __init__(self, transforms):
        self.transforms = transforms
        self.imgs, self.masks = [], []
        self.load()

    def load(self):
        PATHS = ['/root/code/model_data/train_bmp',
                 '/root/code/model_data/train_jpg']

        for PATH in PATHS:
            for file in os.listdir(PATH):
                if '.json' not in file:
                    continue
                json_path = os.path.join(PATH, file)
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
            label_name_to_value = {"_background_": 0, 'cow': 1}
        elif '.bmp' in img_path:
            for mask in mask_json['shapes']:
                if mask['label'] != 'whole':
                    continue
                mask_shapes.append([mask])
                break
            label_name_to_value = {"_background_": 0, 'whole': 1}

        label_d = label_name_to_value
        num_objs = 1
        boxes, masks = [], []
        lbl, _ = labelme_utils.shapes_to_label(img.shape,
                                               mask_shapes[0],
                                               label_d)
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
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_transform():
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 2

dataset_all = CowDataset(get_transform())
indices = torch.randperm(len(dataset_all)).tolist()
split_index = int(len(dataset_all) * 0.8)
dataset = torch.utils.data.Subset(dataset_all, indices[:split_index])
dataset_test = torch.utils.data.Subset(dataset_all, indices[split_index:])
data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=3,
                                          shuffle=True,
                                          num_workers=4,
                                          collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(dataset_test,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=4,
                                               collate_fn=utils.collate_fn)

model = get_model_instance_segmentation(num_classes)
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params,
                            lr=0.001,
                            momentum=0.9,
                            weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)


num_epochs = 100
for epoch in range(num_epochs):
    # qrain for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=5)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)
torch.save(model.state_dict(), '/root/code/model_state/mask_rcnn_0222.pth')

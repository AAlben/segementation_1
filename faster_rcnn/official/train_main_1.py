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


class PennFudanDataset(object):

    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

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


# pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# 1 class (person) + background
num_classes = 2
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


def get_transform():
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)


'''
TEST forward & predict
'''
# For Training
# images, targets = next(iter(data_loader))
# images = list(image for image in images)
# targets = [{k: v for k, v in t.items()} for t in targets]
# output = model(images, targets)   # Returns losses and detections


def main(model):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations

    dataset = CowDataset(get_transform())
    indices = torch.randperm(len(dataset)).tolist()
    split_index = int(len(dataset) * 0.8)
    dataset_train = torch.utils.data.Subset(dataset, indices[:split_index])
    dataset_test = torch.utils.data.Subset(dataset, indices[split_index:])

    data_loader = torch.utils.data.DataLoader(dataset_train,
                                              batch_size=8,
                                              shuffle=True,
                                              num_workers=4,
                                              collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(dataset_test,
                                                   batch_size=1,
                                                   shuffle=False,
                                                   num_workers=4,
                                                   collate_fn=utils.collate_fn)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params,
                                lr=0.005,
                                momentum=0.9,
                                weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 20

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
        torch.save(model.state_dict(), '/root/code/model_state/faster_rcnn_%d.pth' % epoch)

    print("That's it!")


main(model)

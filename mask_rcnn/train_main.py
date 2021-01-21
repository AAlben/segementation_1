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
        for file in os.listdir('train'):
            if '.json' not in file:
                continue

            json_path = os.path.join('train', file)
            img_file = os.path.splitext(file)[0] + '.jpg'
            img_path = os.path.join('train', img_file)

            self.imgs.append(img_path)
            self.masks.append(json_path)

        for file in os.listdir('train_2'):
            if '.json' not in file:
                continue

            json_path = os.path.join('train_2', file)
            img_file = os.path.splitext(file)[0] + '.bmp'
            img_path = os.path.join('train_2', img_file)

            self.imgs.append(img_path)
            self.masks.append(json_path)

    def __getitem__(self, idx):
        img_path, mask_path = self.imgs[idx], self.masks[idx]
        img = cv2.imread(img_path)
        img = cv2.Canny(img, 100, 200)

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
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

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
    return T.Compose(transforms)


def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2

    dataset_all = CowDataset(get_transform(train=True))
    # split the dataset in train and test set
    indices = torch.randperm(len(dataset_all)).tolist()
    split_index = int(len(dataset_all) * 0.8)

    dataset = torch.utils.data.Subset(dataset_all, indices[:split_index])
    dataset_test = torch.utils.data.Subset(dataset_all, indices[split_index:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=6, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 20
    for epoch in range(num_epochs):
        # qrain for one epoch, printing every 10 iterations
        metric_logger = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=5)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        coco_evaluator = evaluate(model, data_loader_test, device=device)
        torch.save(model.state_dict(), 'mask_rcnn_model_%d.pth' % epoch)
    print("That's it!")


if __name__ == '__main__':
    main()

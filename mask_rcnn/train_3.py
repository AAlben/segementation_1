import os
import cv2
import json
import numpy as np
from PIL import Image
from random import shuffle
from collections import OrderedDict
from labelme import utils as labelme_utils

import torch
import torch.utils.data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor, MaskRCNN
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool

import utils
import transforms as T
from engine import train_one_epoch, evaluate


torch.manual_seed(1)


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


class backbone_body(torch.nn.ModuleDict):

    def __init__(self, layers, return_layers):
        super().__init__(layers)
        self.return_layers = return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class BackboneFPN(torch.nn.Sequential):

    def __init__(self, body, fpn, out_channels):
        d = OrderedDict([("body", body),
                         ("fpn", fpn)])
        super(BackboneFPN, self).__init__(d)
        self.out_channels = out_channels


def maskrcnn_resnet18_fpn(num_classes):
    src_backbone = torchvision.models.resnet18(pretrained=True)
    # 去掉后面的全连接层
    return_layers = {'layer1': 0,
                     'layer2': 1,
                     'layer3': 2,
                     'layer4': 3}
    names = [name for name, _ in src_backbone.named_children()]
    # just 验证，失败则报异常
    if not set(return_layers).issubset(names):
        raise ValueError("return_layers are not present in model")

    orig_return_layers = return_layers
    # 复制一份到 layers
    return_layers = {k: v for k, v in return_layers.items()}
    layers = OrderedDict()
    for name, module in src_backbone.named_children():
        layers[name] = module
        if name in return_layers:
            del return_layers[name]
        if not return_layers:
            break

    # 得到去掉池化、全连接层的模型
    backbone_module = backbone_body(layers, orig_return_layers)

    # FPN层，resnet18 layer4 chanels为 512，fpn顶层512/8
    in_channels_stage2 = 64
    in_channels_list = [
        in_channels_stage2,
        in_channels_stage2 * 2,
        in_channels_stage2 * 4,
        in_channels_stage2 * 8,
    ]
    out_channels = 64

    fpn = FeaturePyramidNetwork(
        in_channels_list=in_channels_list,
        out_channels=out_channels,
        extra_blocks=LastLevelMaxPool(),
    )
    backbone_fpn = BackboneFPN(backbone_module,
                               fpn,
                               out_channels)
    model = MaskRCNN(backbone_fpn, num_classes)
    return model


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


model = maskrcnn_resnet18_fpn(num_classes)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params,
                            lr=0.001,
                            momentum=0.9,
                            weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=30,
                                               gamma=0.5)

num_epochs = 100
try:
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
except KeyboardInterrupt:
    torch.save(model.state_dict(), '/root/code/model_state/mask_rcnn_0222_2.pth')
except Exception as e:
    raise
torch.save(model.state_dict(), '/root/code/model_state/mask_rcnn_0222_2.pth')

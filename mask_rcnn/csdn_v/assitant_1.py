import os
import cv2
import numpy as np
from PIL import Image
from random import shuffle

import torch
import torch.utils.data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import utils
import transforms as T
from engine import train_one_epoch, evaluate


class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

colors = [[np.random.randint(0, 255),
           np.random.randint(0, 255),
           np.random.randint(0, 255)]for i in range(100)]

# 为了最终实例分割显示明显,定义常见类别为深色
colors[1] = [255, 0, 0]  # person
colors[2] = [0, 255, 0]  # bicycle
colors[3] = [0, 0, 255]  # car
colors[4] = [255, 255, 0]  # motorcycle


def demo():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    img_dir = '/home/zyk/dataset/PennFudanPed/PNGImages'
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.to(device)
    # put the model in evaluation mode
    model.eval()
    imgs = os.listdir(img_dir)
    shuffle(imgs)
    for i in range(50):
        imgsrc = cv2.imread(os.path.join(img_dir, imgs[i]))
        all_cls_mask_color = np.zeros_like(imgsrc)
        all_cls_mask_index = np.zeros_like(imgsrc)
        img = imgsrc / 255.
        img = np.transpose(img, (2, 0, 1))
        img = torch.tensor(img, dtype=torch.float)
        img = img.to(device)

        with torch.no_grad():
            prediction = model([img])[0]
            scores = prediction['scores']
            for idx, score in enumerate(scores):
                if score > 0.5:
                    mask = prediction['masks'][idx][0].cpu().numpy()
                    mask = mask > 0.5
                    cls_id = prediction['labels'][idx].item()
                    all_cls_mask_color[mask] = colors[cls_id]
                    all_cls_mask_index[mask] = 1

        img_weight = cv2.addWeighted(imgsrc, 0.4, all_cls_mask_color, 0.6, 0)  # 线性混合
        all_mask = all_cls_mask_index == 1
        result = np.copy(imgsrc)
        # 只取mask的混合部分
        result[all_mask] = img_weight[all_mask]
        union = np.concatenate((imgsrc, result), axis=1)
        cv2.imshow('', union)
        cv2.waitKey(0)


def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    # 把模型打印出来,按照名字,输入输出更换backbone,或者改变输出
    print(model)
    # get the number of input features for the classifier
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

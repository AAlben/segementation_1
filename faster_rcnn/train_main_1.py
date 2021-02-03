import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from labelme import utils as labelme_utils

import torch
import torchvision
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

        for path in PATHS:
            for file in os.listdir(path):
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
        labels = np.ones(len(boxes))

        targets = {}
        targets['boxes'] = torch.from_numpy(boxes).double()
        targets['labels'] = torch.from_numpy(labels).type(torch.int64)
        img = self.transforms(img)
        return img.double(), targets

    def __len__(self):
        return len(self.imgs)


torch.manual_seed(1)
dataset = CowDataset()
indices = torch.randperm(len(dataset)).tolist()
split_index = int(len(dataset) * 0.8)
dataset_train = torch.utils.data.Subset(dataset, indices[:split_index])
dataset_test = torch.utils.data.Subset(dataset, indices[split_index:])
data_loader_train = torch.utils.data.DataLoader(dataset_train,
                                                batch_size=4,
                                                shuffle=True,
                                                collate_fn=lambda x: list(zip(*x)))
data_loader_test = torch.utils.data.DataLoader(dataset_test,
                                               batch_size=2,
                                               shuffle=False,
                                               collate_fn=lambda x: list(zip(*x)))


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 2  # 1 class (person) + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model = model.to(device)


params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.01)

EPOCHS = 20
now_loss = 10
model.train()
for epoch in tqdm(range(EPOCHS)):
    for images, targets in tqdm(data_loader_train):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        model = model.double()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.zero_grad()
        optimizer.step()

    if losses.item() < now_loss:
        now_loss = losses.item()
        torch.save(model.state_dict(), '/root/code/model_state/faster_rcnn_kaggle.pth')
    print("Loss = {:.4f} ".format(losses.item()))

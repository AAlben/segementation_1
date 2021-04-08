import os
import itertools
import collections
import numpy as np
import pandas as pd
from numba import jit
from tqdm import tqdm
from PIL import Image, ImageFile

import torch
import torch.utils.data
from torchvision import transforms

import utils
from engine import train_one_epoch, evaluate
import transforms as T

from model import get_instance_segmentation_model


ImageFile.LOAD_TRUNCATED_IMAGES = True


def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated: [start0] [length0] [start1] [length1]... in 1d array
    shape: (height,width) of array to return
    Returns numpy array according to the shape, 1 - mask, 0 - background
    '''
    shape = (shape[1], shape[0])
    s = mask_rle.split()
    # gets starts & lengths 1d arrays
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    # gets ends 1d array
    ends = starts + lengths
    # creates blank mask image 1d array
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    # sets mark pixles
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    # reshape as a 2d mask image
    return img.reshape(shape).T  # Needed to align to RLE direction


class FashionDataset(torch.utils.data.Dataset):

    def __init__(self, image_dir, df_path, height, width, transforms=None):
        self.transforms = transforms
        self.image_dir = image_dir
        self.df = pd.read_csv(df_path, nrows=10000)
        self.height = height
        self.width = width
        self.image_info = collections.defaultdict(dict)
        self.df['CategoryId'] = self.df.ClassId.apply(lambda x: str(x).split("_")[0])
        temp_df = self.df.groupby('ImageId')['EncodedPixels', 'CategoryId'].agg(lambda x: list(x)).reset_index()
        size_df = self.df.groupby('ImageId')['Height', 'Width'].mean().reset_index()
        temp_df = temp_df.merge(size_df, on='ImageId', how='left')
        for index, row in tqdm(temp_df.iterrows(), total=len(temp_df)):
            image_id = row['ImageId']
            image_path = os.path.join(self.image_dir, image_id)
            self.image_info[index]["image_id"] = image_id
            self.image_info[index]["image_path"] = image_path
            self.image_info[index]["width"] = self.width
            self.image_info[index]["height"] = self.height
            self.image_info[index]["labels"] = row["CategoryId"]
            self.image_info[index]["orig_height"] = row["Height"]
            self.image_info[index]["orig_width"] = row["Width"]
            self.image_info[index]["annotations"] = row["EncodedPixels"]

    def __getitem__(self, idx):
        # load images ad masks
        img_path = self.image_info[idx]["image_path"]
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.width, self.height), resample=Image.BILINEAR)

        info = self.image_info[idx]
        mask = np.zeros((len(info['annotations']), self.width, self.height), dtype=np.uint8)
        labels = []
        for m, (annotation, label) in enumerate(zip(info['annotations'], info['labels'])):
            sub_mask = rle_decode(annotation, (info['orig_height'], info['orig_width']))
            sub_mask = Image.fromarray(sub_mask)
            sub_mask = sub_mask.resize((self.width, self.height), resample=Image.BILINEAR)
            mask[m, :, :] = sub_mask
            labels.append(int(label) + 1)

        num_objs = len(labels)
        boxes = []
        new_labels = []
        new_masks = []

        for i in range(num_objs):
            try:
                pos = np.where(mask[i, :, :])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                if abs(xmax - xmin) >= 20 and abs(ymax - ymin) >= 20:
                    boxes.append([xmin, ymin, xmax, ymax])
                    new_labels.append(labels[i])
                    new_masks.append(mask[i, :, :])
            except ValueError:
                continue

        if len(new_labels) == 0:
            boxes.append([0, 0, 20, 20])
            new_labels.append(0)
            new_masks.append(mask[0, :, :])

        nmx = np.zeros((len(new_masks), self.width, self.height), dtype=np.uint8)
        for i, n in enumerate(new_masks):
            nmx[i, :, :] = n

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(new_labels, dtype=torch.int64)
        masks = torch.as_tensor(nmx, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
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
        return len(self.image_info)


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


num_classes = 46 + 1
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

dataset_train = FashionDataset("../input/imaterialist-fashion-2019-FGVC6/train/",
                               "../input/imaterialist-fashion-2019-FGVC6/train.csv",
                               256,
                               256,
                               transforms=get_transform(train=True))


model_ft = get_instance_segmentation_model(num_classes)
model_ft.to(device)

data_loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=4, shuffle=True, num_workers=8,
    collate_fn=lambda x: tuple(zip(*x)))

params = [p for p in model_ft.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001,
                            momentum=0.9, weight_decay=0.0005)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=5,
                                               gamma=0.1)
num_epochs = 10

for epoch in range(num_epochs):
    train_one_epoch(model_ft, optimizer, data_loader, device, epoch, print_freq=10)
    lr_scheduler.step()

torch.save(model_ft.state_dict(), "model.bin")


def refine_masks(masks, labels):
    # Compute the areas of each mask
    areas = np.sum(masks.reshape(-1, masks.shape[-1]), axis=0)
    # Masks are ordered from smallest to largest
    mask_index = np.argsort(areas)
    # One reference mask is created to be incrementally populated
    union_mask = {k: np.zeros(masks.shape[:-1], dtype=bool) for k in np.unique(labels)}
    # Iterate from the smallest, so smallest ones are preserved
    for m in mask_index:
        label = labels[m]
        masks[:, :, m] = np.logical_and(masks[:, :, m], np.logical_not(union_mask[label]))
        union_mask[label] = np.logical_or(masks[:, :, m], union_mask[label])
    # Reorder masks
    refined = list()
    for m in range(masks.shape[-1]):
        mask = masks[:, :, m].ravel(order='F')
        rle = to_rle(mask)
        label = labels[m] - 1
        refined.append([masks[:, :, m], rle, label])
    return refined


num_classes = 46 + 1

dataset_test = FashionDataset("../input/imaterialist-fashion-2019-FGVC6/test/",
                              "../input/imaterialist-fashion-2019-FGVC6/sample_submission.csv", 512, 512,
                              transforms=None)

sample_df = pd.read_csv("../input/imaterialist-fashion-2019-FGVC6/sample_submission.csv")

model_ft = get_instance_segmentation_model(num_classes)
model_ft.load_state_dict(torch.load("model.bin"))
model_ft = model_ft.to(device)

for param in model_ft.parameters():
    param.requires_grad = False

model_ft.eval()


sub_list = []
missing_count = 0
submission = []
ctr = 0

tk0 = tqdm(range(3200))
tt = transforms.ToTensor()
for i in tk0:
    img = dataset_test[i]
    img = tt(img)
    result = model_ft([img.to(device)])[0]
    masks = np.zeros((512, 512, len(result["masks"])))
    for j, m in enumerate(result["masks"]):
        res = transforms.ToPILImage()(result["masks"][j].permute(1, 2, 0).cpu().numpy())
        res = np.asarray(res.resize((512, 512), resample=Image.BILINEAR))
        masks[:, :, j] = (res[:, :] * 255. > 127).astype(np.uint8)

    lbls = result['labels'].cpu().numpy()
    scores = result['scores'].cpu().numpy()

    best_idx = 0
    for scr in scores:
        if scr > 0.8:
            best_idx += 1

    if best_idx == 0:
        sub_list.append([sample_df.loc[i, 'ImageId'], '1 1', 23])
        missing_count += 1
        continue

    if masks.shape[-1] > 0:
            #lll = mask_to_rle(masks[:, :, :4], scores[:4], lbls[:4])
        masks = refine_masks(masks[:, :, :best_idx], lbls[:best_idx])
        for m, rle, label in masks:
            sub_list.append([sample_df.loc[i, 'ImageId'], ' '.join(list(map(str, list(rle)))), label])
    else:
        sub_list.append([sample_df.loc[i, 'ImageId'], '1 1', 23])
        missing_count += 1

submission_df = pd.DataFrame(sub_list, columns=sample_df.columns.values)
print("Total image results: ", submission_df['ImageId'].nunique())
print("Missing Images: ", missing_count)
submission_df = submission_df[submission_df.EncodedPixels.notnull()]
for row in range(len(submission_df)):
    line = submission_df.iloc[row, :]
    submission_df.iloc[row, 1] = line['EncodedPixels'].replace('.0', '')
submission_df.head()
submission_df.to_csv("submission.csv", index=False)

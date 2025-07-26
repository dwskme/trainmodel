import os

import cv2
import torch
from torch.utils.data import Dataset

from transforms import apply_filters


class LaneDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transforms=None):
        self.transforms = transforms
        self.images = []
        self.masks = []

        # only keep pairs that both exist
        for fn in sorted(os.listdir(image_dir)):
            base, ext = os.path.splitext(fn)
            img_path = os.path.join(image_dir, fn)
            mask_path = os.path.join(mask_dir, base + ".png")
            if os.path.isfile(mask_path):
                self.images.append(img_path)
                self.masks.append(mask_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = cv2.imread(self.images[idx])
        img = apply_filters(img)
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype("float32")

        if self.transforms:
            aug = self.transforms(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"]
            mask = mask.unsqueeze(0)
        return img, mask


def dice_loss(pred, target, smooth=1.0):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=(2, 3))
    return (
        1
        - (
            (2.0 * intersection + smooth)
            / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth)
        ).mean()
    )

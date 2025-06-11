import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from random import random

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, target_size=(256, 256)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_size = target_size
        self.images = os.listdir(image_dir)
        self.masks = os.listdir(mask_dir)

        self.augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        mask_name = self.masks[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = self.preprocess_mask(mask, img_name)

        image = self.resize_image(image)
        mask = self.resize_image(mask)

        if self.transform:
            image, mask = self.augment(image, mask)

        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        mask = torch.from_numpy(mask).long()

        return image, mask

    def preprocess_mask(self, mask, img_name):
        h, w = mask.shape
        class_mask = np.zeros((h, w), dtype=np.uint8)
        if 'cat' in img_name.lower():
            class_mask[mask == 255] = 1
        elif 'dog' in img_name.lower():
            class_mask[mask == 255] = 2
        return class_mask

    def augment(self, image, mask):
        image_pil = Image.fromarray(image)
        mask_pil = Image.fromarray(mask)
        seed = np.random.randint(0, 2**32)
        torch.manual_seed(seed)
        image_pil = self.augmentation(image_pil)
        mask_pil = self.augmentation(mask_pil)
        return np.array(image_pil), np.array(mask_pil)

    def resize_image(self, image):
        return cv2.resize(image, (self.target_size[1], self.target_size[0]))

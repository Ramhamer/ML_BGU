import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random
from torchvision import transforms
import torch
import matplotlib.pyplot as plt

def check_name(img_path):
    if "Non-COVID" in img_path:
        covidclass = float(0)
    else:
        covidclass = float(1)

    return covidclass

def grayscale_augmentation(image):
    """ Apply augmentations specifically for grayscale images. """
    augmentation_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ElasticTransform(alpha=50.0)  # Elastic transform with alpha=50.0
    ])
    
    return augmentation_transforms(image)

class Covid19Dataset(Dataset): #preprocessing for the data
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_names = os.listdir(image_dir)
        
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_names[idx])
        image = Image.open(img_path).convert("L")
        covidclass = check_name(img_path)
        
        # Apply transformations
        if self.transform:                          #every picture has the option to got an augmentation - the class will remain the same
            image = grayscale_augmentation(image)
        
        transform = transforms.ToTensor()
        image = transform(image)
        covidclass = np.array(covidclass)

        return image, covidclass

import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random
import torchvision.transforms as T
import torch
import matplotlib.pyplot as plt
import yaml

def check_dir(img_path):
    return

class Covid19Dataset(Dataset): #preprocessing for the data
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_names = os.listdir(image_dir)
        
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_names[idx])
        image = Image.open(img_path).convert("RGB")
        covidclass = check_dir(img_path)

        # Apply class reduction and channel class if needed
        if self.number_class != 7:
            mask = class_reduction(mask, self.number_class)
        
        # Apply transformations
        image,transforms_list = transform_image(image,self.transform)
        mask = transform_mask(mask,transforms_list)
        mask = channel_class(mask, self.number_class)
        return image, mask


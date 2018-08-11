# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 11:03:34 2018

@author: prodi
"""
from __future__ import print_function, division
import os
import torch
import pandas as pd
#from skimage import io, transform
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class HandBagsDataset(Dataset):
    """Handbags dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        self.files = os.listdir(root_dir)
        self.files = [os.path.join(root_dir, f) for f in self.files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        
        img_name = self.files[idx]
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image
    
def fetch_data(data_dir, batch_size, normalize=False):
    """
    Fetches the DataLoader object for each type in types from data_dir.
    Args:
    types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters
    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    
    if normalize==True:
    
        transformer = transforms.Compose([
                transforms.Resize((64,64)),        # resize the image to 64x64 (remove if images are already 64x64)
                transforms.RandomHorizontalFlip(),   # randomly flip image horizontally
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # transform it into a torch tensor
    else:
        
        transformer = transforms.Compose([
                transforms.Resize((64,64)),        # resize the image to 64x64 (remove if images are already 64x64)
                transforms.RandomHorizontalFlip(),   # randomly flip image horizontally
                transforms.ToTensor()])

    
    dataloader = DataLoader(HandBagsDataset(data_dir, transformer), batch_size=batch_size, shuffle=True)

    return dataloader
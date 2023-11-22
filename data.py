import math
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial

from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms, utils

import numpy as np
from tqdm import tqdm
import glob
import os
from PIL import Image
from tifffile import imsave, imread
import matplotlib.pyplot as plt

class Dataset(data.Dataset):
    def __init__(self, folder, image_size, exts = ['jpg', 'jpeg', 'png', 'tif', 'tiff', 'bmp']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        self.transform = transforms.Compose(
            [transforms.RandomCrop(image_size)])

    def __len__(self):
        return len(self.paths) # returns the number of images in the dataset
    
    def __getitem__(self, index):

        path = self.paths[index]
        img = imread(path)
        img = torch.from_numpy(img.copy()) 
        random = np.random.randint(1, 7)
        img = img[(0, random),...] #only take the first (target) and 7th channel (input)
        return img        
    

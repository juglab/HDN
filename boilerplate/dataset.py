import os
import torch
import numpy as np
from pathlib import Path
from tifffile import imread
from torch.utils.data import Dataset, DataLoader

import lib.utils as utils


def _make_datamanager(train_images, val_images, test_images, batch_size, test_batch_size):
    
    """Create data loaders for training, validation and test sets during training.
    The test set will simply be used for plotting and comparing generated images 
    from the learned denoised posterior during training phase. 
    No evaluation will be done on the test set during training. 
    Args:
        train_images (np array): A 3d array
        val_images (np array): A 3d array
        test_images (np array): A 3d array
        batch_size (int): The batch size for training and validation steps
        test_batch_size (int): The batch size for test steps
    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        data_mean: mean of train data and validation data combined
        data_std: std of train data and validation data combined
    """
    
    np.random.shuffle(train_images)
    train_images = train_images
    np.random.shuffle(val_images)
    val_images = val_images
    
    combined_data = np.concatenate((train_images, val_images), axis=0)
    data_mean = np.mean(combined_data)
    data_std = np.std(combined_data)
    train_images = (train_images-data_mean)/data_std
    train_images = torch.from_numpy(train_images)
    train_labels = torch.zeros(len(train_images),).fill_(float('nan'))
    train_set = TensorDataset(train_images, train_labels)

    # TODO add normal dataloader 
    
    val_images = (val_images-data_mean)/data_std
    val_images = torch.from_numpy(val_images)
    val_labels = torch.zeros(len(val_images),).fill_(float('nan'))
    val_set = TensorDataset(val_images, val_labels)
    
    np.random.shuffle(test_images)
    test_images = torch.from_numpy(test_images)
    test_images = (test_images-data_mean)/data_std
    test_labels = torch.zeros(len(test_images),).fill_(float('nan'))
    test_set = TensorDataset(test_images, test_labels)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=True)
    
    return train_loader, val_loader, test_loader, data_mean, data_std


class HDNDataset(Dataset):

    def __init__(self, path, patch_size,  augment=True, filenames='*'):

        self.data = np.concatenate([imread(f) for f in Path(path).rglob((f'{filenames}.tif*'))], 0)
        self.augment = augment
        #TODO handle axes 

    def __len__(self):
        return self.data.shape[0] * 8 # TODO calculate number of patches
    
    def __getitem__(self, idx):
        patches = utils.extract_patches() # TODO Replace with view as windows ?
        if self.augment:
            patches = utils.augment_data() 
        
        #TODO custom collate/sampler to mix patches from different slices !

        return ''
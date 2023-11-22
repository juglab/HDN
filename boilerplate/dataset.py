import os
import torch
import logging
import tifffile
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Union, Optional, Callable

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


class CustomGridPatchDataset(torch.utils.data.IterableDataset):
    """Dataset to extract patches from a list of images and apply transforms to the patches.
    """
    #TODO add napari style axes params, add asserts
    def __init__(
        self,
        data_path: List[str],
        patch_size: Tuple[int],
        patch_iter: Union[np.ndarray, Callable],
        image_level_transform: Optional[Callable] = None,
        patch_level_transform: Optional[Callable] = None,
    ) -> None:
        """
        Parameters
        ----------
        data_path : List[str]
            List of filenames to read image data from.
        patch_size : Tuple[int]
            The size of the patch to extract from the image. Must be a tuple of len either 2 or 3 depending on number of spatial dimension in the data.
        patch_iter : Union[np.ndarray, Callable]
            converts an input image (item from dataset) into a iterable of image patches.
            `patch_iter(dataset[idx])` must yield a tuple: (patches, coordinates).
        image_level_transform : Optional[Callable], optional
            _description_, by default None
        patch_level_transform : Optional[Callable], optional
            _description_, by default None
        """
        # super().__init__(data=data, transform=None)
        self.data = data_path
        self.patch_size = patch_size
        self.patch_iter = patch_iter
        self.image_transform = image_level_transform
        self.patch_transform = patch_level_transform

        #Assert input data
        assert isinstance(data_path, list), f'Incorrect patch_size. Must be a tuple, given{type(data_path)}'
        assert len(data_path) > 0, 'Data source is empty'

        #Assert patch_size
        assert isinstance(patch_size, tuple), f'Incorrect patch_size. Must be a tuple, given{type(patch_size)}'
        assert len(patch_size) in (2, 3), f'Incorrect patch_size. Must be a 2 or 3, given{len(patch_size)}'
    
    @staticmethod
    def read_data_source(self, data_source: str):
        """
        Read data source and correct dimensions

        Parameters
        ----------
        data_source : str
            Path to data source
        
        Returns
        -------
        image volume : np.ndarray
        """

        if not os.path.exists(data_source):
            raise ValueError(f"Data source {data_source} does not exist")

        arr = tifffile.imread(data_source)
        # Assert data dimensions are correct
        assert len(arr.shape) in (2, 3, 4), f'Incorrect data dimensions. Must be 2, 3 or 4, given {arr.shape} for file {data_source}'

        # Adding channel dimension if necessary. If present, check correctness
        if len(arr.shape) == 2 or (len(arr.shape) == 3 and len(self.patch_size) == 3):
            arr = np.expand_dims(arr, axis=0)
        elif len(arr.shape) == 3 and len(self.patch_size) == 2 and arr.shape[0] > 4:
            raise ValueError(f'Incorrect number of channels {arr.shape[0]}')
        elif len(arr.shape) > 3 and len(self.patch_size) == 2:
            raise ValueError(f'Incorrect data dimensions {arr.shape} for given dimensionality {len(self.patch_size)}D in file {data_source}')
        #TODO add other asserts
        return arr
        

    def __iter_source__(self):
        """
        Iterate over data source and yield whole image. Optional transform is applied to the images.

        Yields
        ------
        np.ndarray
        """
        info = torch.utils.data.get_worker_info()
        num_workers = info.num_workers if info is not None else 1
        id = info.id if info is not None else 0
        self.source = iter(self.data)
        #TODO check for mem leaks, explicitly gc the arr after iterator is exhausted
        for i, filename in enumerate(self.source):
            try:
                arr = self.read_data_source(self, filename)
            except (ValueError, FileNotFoundError, OSError) as e:
                logging.exception(f'Exception in file {filename}, skipping')
                raise e
            if i % num_workers == id:
                yield self.image_transform(arr) if self.image_transform is not None else arr

    def __iter__(self):
        """
        Iterate over data source and yield single patch. Optional transform is applied to the patches.

        Yields
        ------
        np.ndarray
        """
        for image in self.__iter_source__():
            for patch in  self.patch_iter(image, self.patch_size):
                #TODO add patch manip n2v 
                yield self.patch_transform(patch) if self.patch_transform is not None else patch



import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from torch.nn import init
from torch.optim.optimizer import Optimizer
import os
import glob
import random
from tifffile import imread
from matplotlib import pyplot as plt
from tqdm import tqdm

from models.lvae import LadderVAE
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
    
def _make_optimizer_and_scheduler(model, lr, weight_decay) -> Optimizer:
    """
    Implements Adamax optimizer and learning rate scheduler.
    Args:
        model: An instance of ladderVAE class
        lr (float): learning rate
        weight_decay (float): weight decay
    """
    optimizer = optim.Adamax(model.parameters(),
                             lr=lr,
                             weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     'min',
                                                      patience=10,
                                                      factor=0.5,
                                                      min_lr=1e-12,
                                                      verbose=True)
    return optimizer, scheduler
        
def forward_pass(x, y, device, model, gaussian_noise_std)-> dict:
    x = x.to(device, non_blocking=True)
    model_out = model(x)
    if model.mode_pred is False:
        
        recons_sep = -model_out['ll']
        kl_sep = model_out['kl_sep']
        kl = model_out['kl']
        kl_loss = model_out['kl_loss']/float(x.shape[2]*x.shape[3])
        
        if gaussian_noise_std is None:
            recons_loss = recons_sep.mean()
        else:
            recons_loss = recons_sep.mean()/ ((gaussian_noise_std/model.data_std)**2)

        
        output = {
                'recons_loss': recons_loss,
                'kl_loss': kl_loss,
                'out_mean': model_out['out_mean'],
                'out_sample': model_out['out_sample']
            }

    else:
        output = {
                'recons_loss': None,
                'kl_loss': None,
                'out_mean': model_out['out_mean'],
                'out_sample': model_out['out_sample']
            }
        
    if 'kl_avg_layerwise' in model_out:
        output['kl_avg_layerwise'] = model_out['kl_avg_layerwise']
        
    return output
    
def img_grid_pad_value(imgs, thresh = .2) -> float:
    """Returns padding value (black or white) for a grid of images.
    Hack to visualize boundaries between images with torchvision's
    save_image(). If the median border value of all images is below the
    threshold, use white, otherwise black (which is the default).
    Args:
        imgs (torch.Tensor): A 4d tensor
        thresh (float, optional): Threshold in (0, 1).
    Returns:
        pad_value (float): The padding value
    """

    assert imgs.dim() == 4
    imgs = imgs.clamp(min=0., max=1.)
    assert 0. < thresh < 1.

    imgs = imgs.mean(1)  # reduce to 1 channel
    h = imgs.size(1)
    w = imgs.size(2)
    borders = list()
    borders.append(imgs[:, 0].flatten())
    borders.append(imgs[:, h - 1].flatten())
    borders.append(imgs[:, 1:h - 1, 0].flatten())
    borders.append(imgs[:, 1:h - 1, w - 1].flatten())
    borders = torch.cat(borders)
    if torch.median(borders) < thresh:
        return 1.0
    return 0.0
    
def save_image_grid(images,filename,nrows):
    """Saves images on disk.
    Args:
        images (torch.Tensor): A 4d tensor
        filename (string): Threshold in (0, 1)
        nrows (int): Number of rows in the image grid to be saved.
    """
    pad = img_grid_pad_value(images)
    save_image(images, filename, nrow=nrows, pad_value=pad, normalize=True)
    
def generate_and_save_samples(model, filename, nrows = 4) -> None:
    """Save generated images at intermediate training steps.
       Args:
           model: instance of LadderVAE class
           filename (str): filename where to save denoised images
           nrows (int): Number of rows in which to arrange denoised/generated images.
           
    """
    samples = model.sample_prior(nrows**2)
    save_image_grid(samples, filename, nrows=nrows)
    return samples
    
def save_image_grid_reconstructions(inputs,recons,filename):
    assert inputs.shape == recons.shape
    n_img = inputs.shape[0]
    n = int(np.sqrt(2 * n_img))
    imgs = torch.stack([inputs.cpu(), recons.cpu()])
    imgs = imgs.permute(1, 0, 2, 3, 4)
    imgs = imgs.reshape(n**2, inputs.size(1), inputs.size(2), inputs.size(3))
    save_image_grid(imgs, filename, nrows=n)
    
def generate_and_save_reconstructions(x,filename,device,model,gaussian_noise_std,data_std,nrows) -> None:
    """Save denoised images at intermediate training steps.
       Args:
           x (Torch.tensor): Batch of images from test set
           filename (str): filename where to save denoised images
           device: cuda device
           model: instance of LadderVAE class
           gaussian_noise_std (float or None): Noise std if data is corrupted with Gaussian noise, else None.
           data_std (float): std of combined train and validation data.
           nrows (int): Number of rows in which to arrange denoised/generated images.
           
    """
    n_img = nrows**2 // 2
    if x.shape[0] < n_img:
        msg = ("{} data points required, but given batch has size {}. "
               "Please use a larger batch.".format(n_img, x.shape[0]))
        raise RuntimeError(msg)
    x = x.to(device)
    
    outputs = forward_pass(x, x, device, model, gaussian_noise_std)

    # Try to get reconstruction from different sources in order
    recons = None
    possible_recons_names = ['out_recons', 'out_mean', 'out_sample']
    for key in possible_recons_names:
        try:
            recons = outputs[key]
            if recons is not None:
                break  # if we found it and it's not None
        except KeyError:
            pass
    if recons is None:
        msg = ("Couldn't find reconstruction in the output dictionary. "
               "Tried keys: {}".format(possible_recons_names))
        raise RuntimeError(msg)

    # Pick required number of images
    x = x[:n_img]
    recons = recons[:n_img]

    # Save inputs and reconstructions in a grid
    save_image_grid_reconstructions(x, recons, filename)
    
def save_images(img_folder, device, model,  test_loader, gaussian_noise_std, data_std, nrows) -> None:
    """Save generated images and denoised images at intermediate training steps.
       Args:
           img_folder (str): Folder where to save images
           device: cuda device
           model: instance of LadderVAE class
           test_loader: Test loader used to denoise during intermediate traing steps.
           gaussian_noise_std (float or None): Noise std if data is corrupted with Gaussian noise, else None.
           data_std (float): std of combined train and validation data.
           nrows (int): Number of rows in which to arrange denoised/generated images.
           
    """
    step = model.global_step

    # Save model samples
    fname = os.path.join(img_folder, 'sample_' + str(step) + '.png')
    generate_and_save_samples(model, fname, nrows)

    # Get first test batch
    (x, _) = next(iter(test_loader))
    x = x.unsqueeze(1)
    x = x.to(device=device, dtype=torch.float)
    # Save model original/reconstructions
    fname = os.path.join(img_folder, 'reconstruction_' + str(step) + '.png')

    generate_and_save_reconstructions(x, fname, device, model, gaussian_noise_std, data_std, nrows)
    
def _test(epoch, img_folder, device, model,  test_loader, gaussian_noise_std, data_std, nrows):
    """Perform a test step at intermediate training steps.
       Args:
           epoch (int): Current training epoch
           img_folder (str): Folder where to save images
           device: cuda device
           model: instance of LadderVAE class
           test_loader: Test loader used to denoise during intermediate traing steps.
           gaussian_noise_std (float or None): Noise std if data is corrupted with Gaussian noise, else None.
           data_std (float): std of combined train and validation data.
           nrows (int): Number of rows in which to arrange denoised/generated images.
           
    """
    # Evaluation mode
    model.eval()
    # Save images
    save_images(img_folder, device, model, test_loader, gaussian_noise_std, data_std, nrows)
    

def get_normalized_tensor(img,model,device):
    '''
    Normalizes tensor with mean and std.
    Parameters
    ----------
    img: array
        Image.
    model: Hierarchical DivNoising model
    device: GPU device.
    '''
    test_images = torch.from_numpy(img.copy()).to(device)
    data_mean = model.data_mean
    data_std = model.data_std
    test_images = (test_images-data_mean)/data_std
    return test_images


def predict_sample(img, model, gaussian_noise_std, device):
    """
    Predicts a sample.
    Parameters
    ----------
    img: array
        Image for which denoised MMSE estimate needs to be computed.
    model: Ladder VAE object
        Hierarchical DivNoising model.
    gaussian_noise_std: float
        std of Gaussian noise used to corrupty data. For intrinsically noisy data, set to None.
    device: GPU device
    """
    outputs = forward_pass(img, img, device, model, gaussian_noise_std)
    recon = outputs['out_mean']
    recon_denormalized = recon*model.data_std+model.data_mean
    recon_cpu = recon_denormalized.cpu()
    recon_numpy = recon_cpu.detach().numpy()
    return recon_numpy


def predict_mmse(img_n, num_samples, model, gaussian_noise_std, device, return_samples=False):
    """
    Predicts desired number of samples and computes MMSE estimate.
    Parameters
    ----------
    img: array
        Image for which denoised MMSE estimate needs to be computed.
    num_samples: int
        Number of samples to average for computing MMSE estimate.
    model: Ladder VAE object
        Hierarchical DivNoising model.
    gaussian_noise_std: float
        std of Gaussian noise used to corrupty data. For intrinsically noisy data, set to None.
    device: GPU device
    """
    img_height,img_width=img_n.shape[0],img_n.shape[1]
    img_t = get_normalized_tensor(img_n,model,device)
    image_sample = img_t.view(1,1,img_height,img_width)
    image_sample = image_sample.to(device=device, dtype=torch.float)
    samples = []
        
    for j in tqdm(range(num_samples)):
        sample = predict_sample(image_sample, model, gaussian_noise_std, device=device)
        samples.append(np.squeeze(sample))
    
    img_mmse = np.mean(np.array(samples),axis=0)
    if return_samples:
        return img_mmse, samples
    else:
        return img_mmse


def predict(img, num_samples, model, gaussian_noise_std, device, tta):
    """
    Predicts desired number of samples and computes MMSE estimate.
    Parameters
    ----------
    img: array
        Image for which denoised MMSE estimate needs to be computed.
    num_samples: int
        Number of samples to average for computing MMSE estimate.
    model: Ladder VAE object
        Hierarchical DivNoising model.
    gaussian_noise_std: float
        std of Gaussian noise used to corrupty data. For intrinsically noisy data, set to None.
    device: GPU device
    tta: bool
        if True, test-time augmentation will be performed else not.
    """
    if tta:
        aug_imgs = tta_forward(img)
        mmse_aug=[]
        for j in range(len(aug_imgs)):
            if(j==0):
                img_mmse,samples = predict_mmse(aug_imgs[j], num_samples, model, gaussian_noise_std, 
                                                device=device, return_samples=True)
            else:
                img_mmse = predict_mmse(aug_imgs[j], num_samples, model, gaussian_noise_std, 
                                    device=device, return_samples=False)
            mmse_aug.append(img_mmse)

        mmse_back_transformed = tta_backward(mmse_aug)
        return mmse_back_transformed, samples
    else:
        img_mmse,samples = predict_mmse(img, num_samples, model, gaussian_noise_std, 
                                                device=device, return_samples=True)
       
        return img_mmse, samples


def tta_forward(x):
    """
    Augments x 8-fold: all 90 deg rotations plus lr flip of the four rotated versions.
    Parameters
    ----------
    x: data to augment
    Returns
    -------
    Stack of augmented x.
    """
    x_aug = [x, np.rot90(x, 1), np.rot90(x, 2), np.rot90(x, 3)]
    x_aug_flip = x_aug.copy()
    for x_ in x_aug:
        x_aug_flip.append(np.fliplr(x_))
    return x_aug_flip


def tta_backward(x_aug):
    """
    Inverts `tta_forward` and averages the 8 images.
    Parameters
    ----------
    x_aug: stack of 8-fold augmented images.
    Returns
    -------
    average of de-augmented x_aug.
    """
    x_deaug = [x_aug[0], 
               np.rot90(x_aug[1], -1), 
               np.rot90(x_aug[2], -2), 
               np.rot90(x_aug[3], -3),
               np.fliplr(x_aug[4]), 
               np.rot90(np.fliplr(x_aug[5]), -1), 
               np.rot90(np.fliplr(x_aug[6]), -2), 
               np.rot90(np.fliplr(x_aug[7]), -3)]
    return np.mean(x_deaug, 0)
       

def generate_arbitrary_sized_samples(sample_shape, num_samples, model, save_path):
    assert isinstance(sample_shape [0], int) and isinstance(sample_shape [1], int)
    torch.set_grad_enabled(False)
    n = num_samples
    model.eval()
    orig_model_img_shape = model.img_shape
    fname = os.path.join(save_path, "samples_"+str(sample_shape[0])+"x"+str(sample_shape[1])+".png")
    
    if sample_shape[0]<orig_model_img_shape[0] or sample_shape[1]<orig_model_img_shape[1]:
        raise RuntimeError('Cannot generate samples of this size, too small, network has too many down/upsamplings') from error
        return
    
    if orig_model_img_shape == sample_shape:
        samples = generate_and_save_samples(model, fname, nrows=n)
    
    if orig_model_img_shape != sample_shape:
        model.img_shape = sample_shape
        change_original_prior_shape = (int(sample_shape[0]/orig_model_img_shape[0]), int(sample_shape[0]/orig_model_img_shape[0]))
        top_layer = model.top_down_layers[-1]
        assert top_layer.is_top_layer
        orig_top_prior_params = top_layer.top_prior_params
        modified_top_prior_params = orig_top_prior_params.repeat(1, 1, change_original_prior_shape[0], change_original_prior_shape[1])
        top_layer.top_prior_params = torch.nn.Parameter(modified_top_prior_params)
        samples = generate_and_save_samples(model, fname, nrows=n)
        
    samples_denormalized = (samples*model.data_std)+model.data_mean
    samples_numpy = samples_denormalized.detach().cpu().numpy()
    samples_numpy = samples_numpy[:,0,:,:]
    
    return samples_numpy
import os
import glob
import random
import numpy as np
import math
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from torch.nn import init
from torch.optim.optimizer import Optimizer
from torch.cuda.amp import GradScaler

from tifffile import imread
from matplotlib import pyplot as plt
from tqdm import tqdm

from boilerplate import boilerplate
from models.lvae import LadderVAE
import lib.utils as utils

import warnings
warnings.filterwarnings('ignore')
# We import all our dependencies.
import numpy as np
import torch
import sys
from models.lvae import LadderVAE
from lib.gaussianMixtureNoiseModel import GaussianMixtureNoiseModel
from boilerplate import boilerplate
import lib.utils as utils
from tifffile import imread
from matplotlib import pyplot as plt
from tqdm import tqdm
import wandb
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Data-specific
gaussian_noise_std = None #changed from 0 to None
#noise_model_params= np.load("data/GMMNoiseModel_convallaria_3_2_calibration.npz")
noiseModel = None #GaussianMixtureNoiseModel(params = noise_model_params, device = device)
image_size = 128
image_shape = (image_size, image_size)

# Training-specific
batch_size = 5
virtual_batch = 20
lr=1e-4 
max_epochs = 500
steps_per_epoch = 300
test_batch_size=1

# Model-specific
num_latents = 5
z_dims = [64]*int(num_latents) #TODO what is this? 
#z_dims = [32,16,8,8]
blocks_per_layer = 5
batchnorm = True
free_bits = 1.0
use_uncond_mode_at=[]


from data import Dataset
from pathlib import Path

data_mean = 0.0
data_std = 1.0


model = LadderVAE(z_dims=z_dims,blocks_per_layer=blocks_per_layer,data_mean=data_mean,data_std=data_std,noiseModel=noiseModel,
                  device=device,batchnorm=batchnorm,free_bits=free_bits,img_shape=image_shape,
                  use_uncond_mode_at=use_uncond_mode_at).cuda()
model = torch.load("./Trained_model/model/TalleyLines_best_vae.net")
model.mode_pred=True
model.eval()

path = "/group/jug/Anirban/Datasets/TalleySim_1024/test/confocal_au_3000.tif"
# The test data is just one quater of the full image ([:,:512,:512]) following the works which have used this data earlier
observation = imread(path)[0:1,...].astype("float32")
signal=observation#np.mean(observation[:,...],axis=0)[np.newaxis,...]
img_width, img_height = signal.shape[1], signal.shape[2]
out = model(torch.from_numpy(np.expand_dims(signal,0)).cuda())

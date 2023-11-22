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
import training
from tifffile import imread
from matplotlib import pyplot as plt
from tqdm import tqdm
import wandb
from ra_psnr import RangeInvariantPsnr, PSNR
from data import Dataset
from pathlib import Path


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model_name = "TalleyLines"
# Data-specific
gaussian_noise_std = None
noiseModel = None 
image_size = 128
# Training-specific
batch_size = 64
virtual_batch = 16
lr=1e-4
max_epochs = 600
free_bits = 0.0
use_uncond_mode_at=[]

#=================================================================================================================================================================
# Model-specific
debug             = False #[True, False]
save_output       = True #[True, False]
project           = 'HDN_Baseline_Modular'
num_latents       = 5
z_dims            = [32]*int(num_latents) 
blocks_per_layer  = 3
n_filters         = [64, 64, 128, 256, 512, 1024] 
stochasticity     = True #[True, False]
likelihood        = 'GaussianLikelihood' # [GaussianLikelihood, GaussianLikelihood_HDN]
recons_weight     = 1.0
dropout           = 0.2
isdropout         = True
batchnorm         = True

if likelihood      == 'GaussianLikelihood':
    logvar_clip    = 5.0
    lvclip         = False
    lvtype         = 'global' #['global', 'pixelwise']
    kl_weight      = 1.0
    gradient_scale = 1.0
elif likelihood    == 'GaussianLikelihood_HDN':
    logvar_clip    = None
    lvclip         = None
    lvtype         = None
    kl_weight      = 0.0001
    gradient_scale = 8192

directory_path = f"NL[{num_latents}]zD[{z_dims[0]}]bpl[{blocks_per_layer}]len_nF[{len(n_filters)}]lh[{likelihood}]Stoc[{stochasticity}]lvClip[{lvclip}]_lvtp[{lvtype}]_DO_[{isdropout}]_BN_[{batchnorm}]" 
name = 'T[1,7]->T0_' + directory_path
#=================================================================================================================================================================

dataset_train = Dataset(Path('/group/jug/Anirban/Datasets/TalleySim_1024/train/'), image_size=image_size)
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
dataset_val = Dataset(Path('/group/jug/Anirban/Datasets/TalleySim_1024/val/'), image_size=image_size)
val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=10, shuffle=False, num_workers=4, drop_last=True)
#not normalizing the data
data_mean = 0.0
data_std = 1.0
max_grad_norm = None


model = LadderVAE(z_dims=z_dims,blocks_per_layer=blocks_per_layer,data_mean=data_mean,data_std=data_std,noiseModel=noiseModel, \
                   device=device,batchnorm=batchnorm,free_bits=free_bits,img_shape=(128,128), \
                    use_uncond_mode_at=use_uncond_mode_at, likelihood_form=likelihood, logvar_clip = logvar_clip, lvclip=lvclip, \
                        lvtype=lvtype, stochasticity=stochasticity, n_filters=n_filters, dropout=dropout).cuda()

model.train() # Model set in training mode

#print(model.top_down_layers)
#print(model.final_top_down)

def count_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def train_network(model, lr, max_epochs,train_loader, val_loader,  
                  virtual_batch, model_name, directory_path="./Trained_Models",
                  max_grad_norm=None, amp=True, gradient_scale=gradient_scale, 
                  project_name=project, name_=name, debug=debug, kl_weight=kl_weight, stochasticity=stochasticity, 
                  recons_weight=recons_weight, save_output=True):
    
    if debug == False:
        use_wandb = True
    else:
        use_wandb = False

    if use_wandb == True:
        experiment = wandb.init(project = project_name,
                                name = name_,
                                resume = 'allow',
                                anonymous = 'must',
                                mode = 'online',
                                reinit = True,
                                save_code = True)

        experiment.config.update(dict(epochs=max_epochs,
                                        batch_size=batch_size,
                                        ))

        wandb.run.log_code(("."), include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"))

    
    model_folder = directory_path+"/model/"
    img_folder = directory_path+"/imgs/"
    device = model.device
    optimizer, scheduler = boilerplate._make_optimizer_and_scheduler(model,lr,0.0)
    loss_train_history = []
    reconstruction_loss_train_history = []
    kl_loss_train_history = []
    loss_val_history = []
    psnr_ra_best = 0.0  
    patience_ = 0
    
    if save_output == True:
        try:
            #make directory inside the directory_path
            os.makedirs(model_folder)
        except FileExistsError:
        # directory already exists
            pass
        
        try:
            os.makedirs(img_folder)
        except FileExistsError:
        # directory already exists
            pass
    
    seconds_last = time.time()

    #AMP gradscaler
    scaler = GradScaler(init_scale=gradient_scale, enabled=amp)

    steps = 0

    for epoch in tqdm(range(0, max_epochs)):
        print(f'Epoch {epoch}/{max_epochs}')
        running_training_loss = []
        running_reconstruction_loss = []
        running_kl_loss = []

        for images in tqdm(train_loader, leave=False): # x and y are the same

            images = images.to(device=device, dtype=torch.float)       
            optimizer.zero_grad()
     
            ### Make smaller batches 
            virtual_batches = torch.split(images,virtual_batch,0)
            for batch in virtual_batches:

                images_input  = batch[:,1:2,:,:]
                images_target = batch[:,0:1,:,:]

                outputs = boilerplate.forward_pass(images_input, images_target, device, model, gaussian_noise_std=None, amp=amp, stochasticity=stochasticity)
                recons_loss = outputs['recons_loss']
                if stochasticity == True:
                    kl_loss = outputs['kl_loss']
                    loss = recons_weight * recons_loss + kl_loss * kl_weight
                else:
                    loss = recons_weight * recons_loss
                steps += 1 
                scaler.scale(loss).backward()

                psnr = PSNR(images_target[:,0,...], outputs['out_img'][:,0,...])
                psnr = psnr.mean()
                psnr_ra = RangeInvariantPsnr(images_target[:,0,...], outputs['out_img'][:,0,...])
                psnr_ra = psnr_ra.mean()

                if use_wandb == True:
                    if stochasticity == True:
                        experiment.log({
                            'recons_loss': recons_loss,
                            'kl_loss': kl_loss,
                            'loss': loss.item(),
                            'psnr': psnr.item(),
                            'psnr_ra': psnr_ra.item()
                        })
                    else:

                        experiment.log({
                            'recons_loss': recons_loss,
                            'loss': loss.item(),
                            'psnr': psnr.item(),
                            'psnr_ra': psnr_ra.item()
                        })


                running_training_loss.append(loss.item())
                running_reconstruction_loss.append(recons_loss.item())
                if stochasticity == True:
                    running_kl_loss.append(kl_loss.item())



            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            
            model.increment_global_step()
            step = model.global_step

        if epoch % 50 == 0 and save_output == True:
            psnr_fig = PSNR(images_target[0:1:,0,...], outputs['out_img'][0:1:,0,...])
            psnr_fig_ra = RangeInvariantPsnr(images_target[0:1:,0,...], outputs['out_img'][0:1:,0,...])
            plt.figure(figsize=(10,10))
            plt.subplot(1,3,1)
            plt.imshow(images_input[0,0,:,:].detach().cpu().numpy())
            plt.axis('off')
            plt.title(f'Input')
            plt.subplot(1,3,2)
            plt.imshow(outputs['out_img'][0,0,:,:].detach().cpu().numpy())
            plt.title(f'Output Image PSNR: {psnr_fig.item():.2f}')
            plt.xlabel(f'Output Image PSNR_RA: {psnr_fig_ra.item():.2f}')
            plt.subplot(1,3,3)
            plt.imshow(images_target[0,0,:,:].detach().cpu().numpy())
            plt.axis('off')
            plt.title(f'Target')
            plt.savefig(img_folder+'batch_train_'+str(epoch)+'.png')
            plt.close()


        ### Print training losses
        if stochasticity == True:
            to_print = "Epoch[{}/{}] Training Loss: {:.3f} Reconstruction Loss: {:.3f} KL Loss: {:.3f}"
        else:
            to_print = "Epoch[{}/{}] Training Loss: {:.3f} Reconstruction Loss: {:.3f}"
        if stochasticity == True:
            to_print = to_print.format(epoch,
                                        max_epochs, 
                                        np.mean(running_training_loss),
                                        np.mean(running_reconstruction_loss),
                                        np.mean(running_kl_loss))
        else:
            to_print = to_print.format(epoch,
                                        max_epochs, 
                                        np.mean(running_training_loss),
                                        np.mean(running_reconstruction_loss))


        print(to_print)
        if debug == False:
            print('saving',model_folder+model_name+"_last_vae.net")
            torch.save(model, model_folder+model_name+"_last_vae.net")

            ### Save training losses 
            loss_train_history.append(np.mean(running_training_loss))
            reconstruction_loss_train_history.append(np.mean(running_reconstruction_loss))            
            np.save(model_folder+"train_loss.npy", np.array(loss_train_history))
            np.save(model_folder+"train_reco_loss.npy", np.array(reconstruction_loss_train_history))
            if stochasticity == True:
                kl_loss_train_history.append(np.mean(running_kl_loss))
                np.save(model_folder+"train_kl_loss.npy", np.array(kl_loss_train_history))


        ### Validation step
        running_validation_loss = []
        psnr_ra_val_list = []
        model.eval()
        with torch.no_grad():
            for images_val in tqdm(val_loader, leave=False):

                images_val = images_val.to(device=device, dtype=torch.float)        

                ### Make smaller batches 
                virtual_batches_val = torch.split(images_val,virtual_batch,0)
                for batch_val in virtual_batches_val:

                    images_input_val  = batch_val[:,1:2,:,:]
                    images_target_val = batch_val[:,0:1,:,:]

                    val_outputs = boilerplate.forward_pass(images_input_val, images_target_val, device, model, gaussian_noise_std=None, stochasticity=stochasticity)
                    val_recons_loss = val_outputs['recons_loss']
                    if stochasticity == True:
                        val_kl_loss = val_outputs['kl_loss']
                        val_loss = recons_weight * val_recons_loss + val_kl_loss * kl_weight
                    else:
                        val_loss = recons_weight * val_recons_loss
                    running_validation_loss.append(val_loss)

                    psnr_val = PSNR(images_target_val[:,0,...], val_outputs['out_img'][:,0,...])
                    psnr_val = psnr_val.mean()
                    psnr_ra_val = RangeInvariantPsnr(images_target_val[:,0,...], val_outputs['out_img'][:,0,...],)
                    psnr_ra_val = psnr_val.mean()
                    psnr_ra_val_list.append(psnr_ra_val)


                    if use_wandb == True:
                        if stochasticity == True:
                            experiment.log({
                                'recons_loss_val': val_recons_loss,
                                'kl_loss_val': val_kl_loss,
                                'loss_val': val_loss.item(),
                                'psnr_val': psnr_val.item(),
                                'psnr_ra_val': psnr_ra_val.item()
                            })
                        else:
                            experiment.log({
                                'recons_loss_val': val_recons_loss,
                                'loss_val': val_loss.item(),
                                'psnr_val': psnr_val.item(),
                                'psnr_ra_val': psnr_ra_val.item()
                            })
                            
            if epoch % 10 == 0 and save_output == True:
                psnr_fig_val = PSNR(images_target_val[0:1:,0,...], val_outputs['out_img'][0:1:,0,...])
                psnr_fig_ra_val = RangeInvariantPsnr(images_target_val[0:1:,0,...], val_outputs['out_img'][0:1:,0,...])
                plt.figure(figsize=(10,10))
                plt.subplot(1,3,1)
                plt.imshow(images_input_val[0,0,:,:].detach().cpu().numpy())
                plt.axis('off')
                plt.title(f'Input')
                plt.subplot(1,3,2)
                plt.imshow(val_outputs['out_img'][0,0,:,:].detach().cpu().numpy())
                plt.title(f'Output Image PSNR: {psnr_fig_val.item():.2f}')
                plt.xlabel(f'Output Image PSNR_RA: {psnr_fig_ra_val.item():.2f}')
                plt.subplot(1,3,3)
                plt.imshow(images_target_val[0,0,:,:].detach().cpu().numpy())
                plt.axis('off')
                plt.title(f'Target')
                plt.savefig(img_folder+'batch_val_'+str(epoch)+'.png')
                plt.close()

        model.train()

        total_epoch_loss_val = torch.mean(torch.stack(running_validation_loss))
        scheduler.step(total_epoch_loss_val)

        ### Save validation losses      
        loss_val_history.append(total_epoch_loss_val.item())
        if save_output == True:
            np.save(model_folder+"val_loss.npy", np.array(loss_val_history))

        if debug == False and save_output == True:
            if torch.mean(torch.stack(psnr_ra_val_list)).item() > psnr_ra_best:
                psnr_ra_best = torch.mean(torch.stack(psnr_ra_val_list)).item()
                print('saving',model_folder+model_name+"_best_vae.net")
                torch.save(model, model_folder+model_name+"_best_vae.net")

            else:
                patience_ +=1

        print("Patience:", patience_,
                "Validation Loss:", total_epoch_loss_val.item(),
                "Min validation loss:", np.min(loss_val_history))

        seconds=time.time()
        secondsElapsed=np.float(seconds-seconds_last)
        seconds_last=seconds
        remainingEps=(max_epochs+1)-(epoch+1)
        estRemainSeconds=(secondsElapsed)*(remainingEps)
        estRemainSecondsInt=int(secondsElapsed)*(remainingEps)
        print('Time for epoch: '+ str(int(secondsElapsed))+ 'seconds')

        print('Est remaining time: '+
                str(datetime.timedelta(seconds= estRemainSecondsInt)) +
                ' or ' +
                str(estRemainSecondsInt)+ 
                ' seconds')

        print("----------------------------------------", flush=True)

if __name__ == '__main__':
    print(count_parameters(model)/ 1e6, 'Million trainable parameters')
    train_network(model=model,lr=lr,max_epochs=max_epochs,directory_path="./Trained_Models/"+directory_path,train_loader=train_loader,val_loader=val_loader,
                            virtual_batch=virtual_batch, max_grad_norm = max_grad_norm,  
                            model_name=model_name, 
                                kl_weight = kl_weight, gradient_scale=gradient_scale, project_name=project, 
                                name_=name, debug=debug, stochasticity=stochasticity, recons_weight=recons_weight, save_output=save_output)
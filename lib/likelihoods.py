import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F
from typing import Type, Union
 
class LikelihoodModule(nn.Module):

    def distr_params(self, x):
        pass

    @staticmethod
    def mean(params):
        pass

    @staticmethod
    def mode(params):
        pass

    @staticmethod
    def sample(params):
        pass

    def log_likelihood(self, x, params):
        pass

    def forward(self, out, y):
        distr_params = self.distr_params(out)
        mean = self.mean(distr_params)
        mode = self.mode(distr_params)
        sample = self.sample(distr_params)
        if y is None:
            ll = None
        else:
            ll = self.log_likelihood(y, distr_params)
        dct = {
            'mean': mean,
            'mode': mode,
            'sample': sample,
            'params': distr_params,
        }
        return ll, dct
    
  
class GaussianLikelihood(LikelihoodModule):

    def __init__(self, ch_in, color_channels, logvar_clip, conv_mult=2, lvclip=False, lv_type='pixelwise'):
        super().__init__()

        conv_type: Type[Union[nn.Conv2d, nn.Conv3d]] = getattr(nn, f'Conv{conv_mult}d')
        self.parameter_net = conv_type(ch_in,
                                       2 * color_channels,
                                       kernel_size=3,
                                       padding=1)
        self.logvar_clip = logvar_clip
        self.lvclip = lvclip
        self.lv_type = lv_type
    #out goes here
    def distr_params(self, x):
        x = self.parameter_net(x)

        if self.lv_type == 'pixelwise':
            mean, lv = x.chunk(2, dim=1) #pixelwise mean and logvar
        elif self.lv_type == 'global':    
            mean, lv = get_mean_lv(x)

        #clipping situation
        if self.lvclip == True:
            lv = torch.clip(lv, min=-self.logvar_clip, max=self.logvar_clip)
        
        params = {
            'mean': mean,
            'logvar': lv,
        }
        return params

    @staticmethod
    def mean(params):
        return params['mean']

    @staticmethod
    def mode(params):
        return params['mean']

    @staticmethod
    def sample(params):
        p = Normal(params['mean'], (params['logvar'] / 2).exp())
        return p.rsample()

    def log_likelihood(self, y, params):
        logprob = log_normal(y, params['mean'], params['logvar'], reduce='none')
        return logprob


def log_normal(x, mean, logvar, reduce='none', eps=1e-6):
    """
    Log of the probability density of the values x under the Normal
    distribution with parameters mean and logvar. The sum is taken over all
    dimensions except for the first one (assumed to be batch). Reduction
    is applied at the end.
    :param x: tensor of points, with shape (batch, channels, dim1, dim2)
    :param mean: tensor with mean of distribution, shape
                 (batch, channels, dim1, dim2)
    :param logvar: tensor with log-variance of distribution, shape has to be
                   either scalar or broadcastable
    :param reduce: reduction over batch: 'mean' | 'sum' | 'none'
    :return:
    """
    logvar = _input_check(x, mean, logvar, reduce)
    var = torch.exp(logvar)
    log_prob = -0.5 * (((x - mean)**2) / var + logvar + torch.tensor(2 * math.pi).log())  
    #log_prob = log_prob.sum((1, 2, 3))
    return _reduce(log_prob, reduce)

def _input_check(x, mean, scale_param, reduce):
    assert x.dim() == 4
    assert x.size() == mean.size()
    if scale_param.numel() == 1:
        scale_param = scale_param.view(1, 1, 1, 1)
    if reduce not in ['mean', 'sum', 'none']:
        msg = "unrecognized reduction method '{}'".format(reduce)
        raise RuntimeError(msg)
    return scale_param

def _reduce(x, reduce):
    if reduce == 'mean':
        x = x.mean()
    elif reduce == 'sum':
        x = x.sum()
    return x

#from Ashesh
def get_mean_lv(x, predict_logvar='global', logvar_lowerbound=None):
    if predict_logvar is not None:
        # pixelwise mean and logvar
        mean, lv = x.chunk(2, dim=1)
        if predict_logvar in ['channelwise', 'global']:
            if predict_logvar == 'channelwise':
                # logvar should be of the following shape (batch,num_channels). Other dims would be singletons.
                N = np.prod(lv.shape[:2])
                new_shape = (*mean.shape[:2], *([1] * len(mean.shape[2:])))
            elif predict_logvar == 'global':
                # logvar should be of the following shape (batch). Other dims would be singletons.
                N = lv.shape[0]
                new_shape = (*mean.shape[:1], *([1] * len(mean.shape[1:])))
            else:
                raise ValueError(f"Invalid value for self.predict_logvar:{predict_logvar}")

            lv = torch.mean(lv.reshape(N, -1), dim=1)
            lv = lv.reshape(new_shape)

        if logvar_lowerbound is not None:
            lv = torch.clip(lv, min=logvar_lowerbound)
    else:
        mean = x
        lv = None
    return mean, lv


class GaussianLikelihood_HDN(LikelihoodModule):

    '''
    Description: Gaussian likelihood

    '''
    #TODO might need to change this to the original
    # https://github.com/addtt/ladder-vae-pytorch/blob/master/lib/likelihoods.py

    def __init__(self, ch_in, color_channels, conv_mult=2):
        super().__init__()

        conv_type: Type[Union[nn.Conv2d, nn.Conv3d]] = getattr(nn, f'Conv{conv_mult}d')
        self.parameter_net = conv_type(ch_in,
                                       color_channels,
                                       kernel_size=3,
                                       padding=1)
    def distr_params(self, x):
        x = self.parameter_net(x)
        mean = x
        lv = None
        params = {
            'mean': mean,
            'logvar': lv,
        }
        return params

    @staticmethod
    def mean(params):
        return params['mean']

    @staticmethod
    def mode(params):
        return params['mean']

    @staticmethod
    def sample(params):
        # p = Normal(params['mean'], (params['logvar'] / 2).exp())
        # return p.rsample()
        return params['mean']

    def log_likelihood(self, x, params):
        logprob = -0.5 *(params['mean']-x)**2
        #logprob = log_normal(x, params['mean'], params['logvar'], reduce='none')
        return logprob
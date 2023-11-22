import torch
from torch import nn
from torch.distributions import kl_divergence
from torch.distributions.normal import Normal
from typing import Type, Union


class NormalStochasticConvBlock(nn.Module):
    """
    Transform input parameters to q(z) with a convolution, optionally do the
    same for p(z), then sample z ~ q(z) and return conv(z).

    If q's parameters are not given, do the same but sample from p(z).
    """

    def __init__(self, c_in, c_vars, c_out, conv_mult, kernel=3, transform_p_params=True):
        super().__init__()
        assert kernel % 2 == 1
        pad = kernel // 2
        self.transform_p_params = transform_p_params
        self.c_in = c_in
        self.c_out = c_out
        self.c_vars = c_vars

        conv_type: Type[Union[nn.Conv2d, nn.Conv3d]] = getattr(nn, f'Conv{conv_mult}d')

        if transform_p_params:
            self.conv_in_p = conv_type(c_in, 2 * c_vars, kernel, padding=pad)
        self.conv_in_q = conv_type(c_in, 2 * c_vars, kernel, padding=pad)
        self.conv_out = conv_type(c_vars, c_out, kernel, padding=pad)

    def forward(self,
                p_params,
                q_params=None,
                forced_latent=None,
                use_mode=False,
                force_constant_output=False,
                analytical_kl=False,
                mode_pred=False,
                use_uncond_mode=False):

        assert (forced_latent is None) or (not use_mode)

        if self.transform_p_params:
            p_params = self.conv_in_p(p_params)
        else:
            #TODO better assertion logic
            assert max(p_params.shape) == 2 * self.c_vars

        # Define p(z)
        p_mu, p_lv = p_params.chunk(2, dim=1)
        p = Normal(p_mu, (p_lv / 2).exp())

        if q_params is not None:
            # Define q(z)
            q_params = self.conv_in_q(q_params)
            q_mu, q_lv = q_params.chunk(2, dim=1)
            q = Normal(q_mu, (q_lv / 2).exp())
            # Sample from q(z)
            sampling_distrib = q
        else:
            # Sample from p(z)
            sampling_distrib = p

        # Generate latent variable (typically by sampling)
        if forced_latent is None:
            if use_mode:
                z = sampling_distrib.mean
            else:
                if mode_pred:
                    if use_uncond_mode:
                        z = sampling_distrib.mean
#                         z = sampling_distrib.rsample()
                    else:
                        z = sampling_distrib.rsample()
                else:
                    z = sampling_distrib.rsample()
        else:
            z = forced_latent

        # Copy one sample (and distrib parameters) over the whole batch.
        # This is used when doing experiment from the prior - q is not used.
        if force_constant_output:
            z = z[0:1].expand_as(z).clone()
            p_params = p_params[0:1].expand_as(p_params).clone()

        # Output of stochastic layer
        out = self.conv_out(z)

        # Compute log p(z)
        if mode_pred is False:
            #Summing over all dims but batch
            logprob_p = p.log_prob(z).sum(list(range(1, z.dim())))
        else:
            logprob_p = None

        if q_params is not None:

            # Compute log q(z)
            logprob_q = q.log_prob(z).sum(list(range(1, z.dim())))

            if mode_pred is False: # if not predicting
                # Compute KL (analytical or MC estimate)
                kl_analytical = kl_divergence(q, p)
                if analytical_kl:
                    kl_elementwise = kl_analytical
                else:
                    kl_elementwise = kl_normal_mc(z, p_params, q_params)
                kl_samplewise = kl_elementwise.sum(list(range(1, z.dim())))

                # Compute spatial KL analytically (but conditioned on samples from
                # previous layers)
                kl_spatial_analytical = kl_analytical.sum(1)
            else: # if predicting, no need to compute KL
                kl_analytical = None
                kl_elementwise = None
                kl_samplewise = None
                kl_spatial_analytical = None

        else:
            kl_elementwise = kl_samplewise = kl_spatial_analytical = None
            logprob_q = None

        data = {
            'z': z,  # sampled variable at this layer (batch, ch, h, w)
            'p_params': p_params,  # (b, ch, h, w) where b is 1 or batch size
            'q_params': q_params,  # (batch, ch, h, w)
            'logprob_p': logprob_p,  # (batch, )
            'logprob_q': logprob_q,  # (batch, )
            'kl_elementwise': kl_elementwise,  # (batch, ch, h, w)
            'kl_samplewise': kl_samplewise,  # (batch, )
            'kl_spatial': kl_spatial_analytical,  # (batch, h, w)
        }
        return out, data


def kl_normal_mc(z, p_mulv, q_mulv):
    """
    One-sample estimation of element-wise KL between two diagonal
    multivariate normal distributions. Any number of dimensions,
    broadcasting supported (be careful).

    :param z:
    :param p_mulv:
    :param q_mulv:
    :return:
    """
    p_mu, p_lv = torch.chunk(p_mulv, 2, dim=1)
    q_mu, q_lv = torch.chunk(q_mulv, 2, dim=1)
    p_std = (p_lv / 2).exp()
    q_std = (q_lv / 2).exp()
    p_distrib = Normal(p_mu, p_std)
    q_distrib = Normal(q_mu, q_std)
    return q_distrib.log_prob(z) - p_distrib.log_prob(z)

"""
Adapted from https://github.com/juglab/HDN/blob/e30edf7ec2cd55c902e469b890d8fe44d15cbb7e/lib/stochastic.py
"""
class NonStochasticBlock2d(nn.Module):
    """
    Non-stochastic version of the NormalStochasticBlock2d
    """

    def __init__(self,
                 c_in: int,
                 c_vars: int,
                 c_out,
                 kernel: int = 3,
                 groups=1,
                 conv2d_bias: bool = True,
                 transform_p_params: bool = True):
        """
        Args:
            c_in:   This is the channel count of the tensor input to this module.
            c_vars: This is the size of the latent space
            c_out:  Output of the stochastic layer. Note that this is different from z.
            kernel: kernel used in convolutional layers.
            transform_p_params: p_params are transformed if this is set to True.
        """
        super().__init__()
        assert kernel % 2 == 1
        pad = kernel // 2
        self.transform_p_params = transform_p_params
        self.c_in = c_in
        self.c_out = c_out
        self.c_vars = c_vars

        if transform_p_params:
            self.conv_in_p = nn.Conv2d(c_in, 2 * c_vars, kernel, padding=pad, bias=conv2d_bias, groups=groups)
        self.conv_in_q = nn.Conv2d(c_in, 2 * c_vars, kernel, padding=pad, bias=conv2d_bias, groups=groups)
        self.conv_out = nn.Conv2d(c_vars, c_out, kernel, padding=pad, bias=conv2d_bias, groups=groups)

    def compute_kl_metrics(self, p, p_params, q, q_params, mode_pred, analytical_kl, z):
        """
        Compute KL (analytical or MC estimate) and then process it in multiple ways.
        """

        kl_dict = {
            'kl_elementwise': None,  # (batch, ch, h, w)
            'kl_samplewise': None,  # (batch, )
            'kl_spatial': None,  # (batch, h, w)
            'kl_channelwise': None  # (batch, ch)
        }
        return kl_dict

    def process_p_params(self, p_params, var_clip_max):
        if self.transform_p_params:
            p_params = self.conv_in_p(p_params)
        else:

            assert p_params.size(1) == 2 * self.c_vars, f'{p_params.shape} {self.c_vars}'

        # Define p(z)
        p_mu, p_lv = p_params.chunk(2, dim=1)
        return p_mu, None

    def process_q_params(self, q_params, var_clip_max, allow_oddsizes=False):
        # Define q(z)
        q_params = self.conv_in_q(q_params)
        q_mu, q_lv = q_params.chunk(2, dim=1)

        if q_mu.shape[-1] % 2 == 1 and allow_oddsizes is False:
            q_mu = F.center_crop(q_mu, q_mu.shape[-1] - 1)

        return q_mu, None

    def forward(self,
                p_params: torch.Tensor,
                q_params: torch.Tensor = None,
                forced_latent: Union[None, torch.Tensor] = None,
                use_mode: bool = False,
                force_constant_output: bool = False,
                analytical_kl: bool = False,
                mode_pred: bool = False,
                use_uncond_mode: bool = False,
                var_clip_max: Union[None, float] = None):
        """
        Args:
            p_params: this is passed from top layers.
            q_params: this is the merge of bottom up layer at this level and top down layers above this level.
            forced_latent: If this is a tensor, then in stochastic layer, we don't sample by using p() & q(). We simply 
                            use this as the latent space sampling.
            use_mode:   If it is true, we still don't sample from the q(). We simply 
                            use the mean of the distribution as the latent space.
            force_constant_output: This ensures that only the first sample of the batch is used. Typically used 
                                when infernce_mode is False
            analytical_kl: If True, typical KL divergence is calculated. Otherwise, a one-sample approximate of it is
                            calculated.
            mode_pred: If True, then only prediction happens. Otherwise, KL divergence loss also gets computed.
            use_uncond_mode: Used only when mode_pred=True
            var_clip_max: This is the maximum value the log of the variance of the latent vector for any layer can reach.
            
        """

        debug_qvar_max = 0
        assert (forced_latent is None) or (not use_mode)

        p_mu, _ = self.process_p_params(p_params, var_clip_max)

        p_params = (p_mu, None)

        if q_params is not None:
            # At inference time, just don't centercrop the q_params even if they are odd in size.
            q_mu, _ = self.process_q_params(q_params, var_clip_max, allow_oddsizes=mode_pred is True)
            q_params = (q_mu, None)
            debug_qvar_max = torch.Tensor([1]).to(q_mu.device)
            # Sample from q(z)
            sampling_distrib = q_mu
            q_size = q_mu.shape[-1]
            if p_mu.shape[-1] != q_size and mode_pred is False:
                p_mu.centercrop_to_size(q_size)
        else:
            # Sample from p(z)
            sampling_distrib = p_mu

        # Generate latent variable (typically by sampling)
        z = sampling_distrib

        # Copy one sample (and distrib parameters) over the whole batch.
        # This is used when doing experiment from the prior - q is not used.
        if force_constant_output:
            z = z[0:1].expand_as(z).clone()
            p_params = (p_params[0][0:1].expand_as(p_params[0]).clone(),
                        p_params[1][0:1].expand_as(p_params[1]).clone())

        # Output of stochastic layer
        out = self.conv_out(z)

        kl_dict = {}
        logprob_q = None
        logprob_p = None

        data = kl_dict
        data['z'] = z  # sampled variable at this layer (batch, ch, h, w)
        data['p_params'] = p_params  # (b, ch, h, w) where b is 1 or batch size
        data['q_params'] = q_params  # (batch, ch, h, w)
        data['logprob_q'] = logprob_q  # (batch, )
        data['logprob_p'] = logprob_p  # (batch, )
        data['qvar_max'] = debug_qvar_max
        data['kl_elementwise'] = None
        data['kl_samplewise'] = None
        data['kl_spatial'] = None

        return out, data
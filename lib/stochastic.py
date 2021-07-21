import torch
from torch import nn
from torch.distributions import kl_divergence
from torch.distributions.normal import Normal


class NormalStochasticBlock2d(nn.Module):
    """
    Transform input parameters to q(z) with a convolution, optionally do the
    same for p(z), then sample z ~ q(z) and return conv(z).

    If q's parameters are not given, do the same but sample from p(z).
    """

    def __init__(self, c_in, c_vars, c_out, kernel=3, transform_p_params=True):
        super().__init__()
        assert kernel % 2 == 1
        pad = kernel // 2
        self.transform_p_params = transform_p_params
        self.c_in = c_in
        self.c_out = c_out
        self.c_vars = c_vars

        if transform_p_params:
            self.conv_in_p = nn.Conv2d(c_in, 2 * c_vars, kernel, padding=pad)
        self.conv_in_q = nn.Conv2d(c_in, 2 * c_vars, kernel, padding=pad)
        self.conv_out = nn.Conv2d(c_vars, c_out, kernel, padding=pad)

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
            assert p_params.size(1) == 2 * self.c_vars

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
            logprob_p = p.log_prob(z).sum((1, 2, 3))
        else: 
            logprob_p = None

        if q_params is not None:

            # Compute log q(z)
            logprob_q = q.log_prob(z).sum((1, 2, 3))
            
            if mode_pred is False: # if not predicting
                # Compute KL (analytical or MC estimate)
                kl_analytical = kl_divergence(q, p)
                if analytical_kl:
                    kl_elementwise = kl_analytical
                else:
                    kl_elementwise = kl_normal_mc(z, p_params, q_params)
                kl_samplewise = kl_elementwise.sum((1, 2, 3))

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

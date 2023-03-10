# Software Name: PaNVC
# Copyright (c) 2023 Ateme
# Licence :  BSD-3-Clause "New" 
# This software is distributed under the BSD-3-Clause license.

import torch
import torch.nn as nn
from torch.distributions import Laplace
from torch import log2, clamp, sqrt, zeros_like
from model.math_func import PROBA_MIN, PROBA_MAX, LOG_VAR_MIN, LOG_VAR_MAX



class Hyperprior_Proba_Model(nn.Module):
    """
    Hyperprior probability model balle(2018) 
    """
    def __init__(self, channel_y = 256):
        super(Hyperprior_Proba_Model, self).__init__()
        self.channel_y = channel_y
        
    def forward(self, y, mu, sigma, device='cuda:0' ):               #  parametric entropy model supposing that the distribution is laplacien with mean mu
        normalization = torch.sqrt(torch.tensor([2.0], device=device))
        scale_factor = sigma / normalization
        laplace = torch.distributions.laplace.Laplace(mu, scale_factor)
        proba = laplace.cdf(y + 0.5) - laplace.cdf(y - 0.5)
        return proba

"""
 This module is a modified vesion of the "PdfParamParameterizer" module introduced in : https://github.com/Orange-OpenSource/AIVC
 The original module is licenced under BSD 3-Clause "New"
 
 The licences on which our framework depends are to be found in the folder named "licence_dependency"
"""
   

class PdfParamParameterizer_simp(nn.Module):
    """
    Extract mu and sigma from the output of the hyper-decoder 
    """

    def __init__(self, nb_ft):
        super(PdfParamParameterizer_simp, self).__init__()
        # nb_ft: number of feature maps parameterized by x
        self.nb_ft = nb_ft

    def forward(self, x):
        """
        Arguments:
            x: Parameters to be re-parameterize (namely sigma)

        Returns:
            pdf parameters stored as a list of dic
        """
        cur_device = x.device
        C = self.nb_ft
        B, _, H, W = x.size()
        K = 1
        # Retrieve all parameters for all the component of the mixture
        # The dimension 1 (K) corresponds to the index of the
        # different components of the gaussian mixture.
        # Mu:
        all_mu = torch.zeros((B, K, C, H, W), device=cur_device)
        start_idx = 0
        for k in range(K):
            all_mu[:, k, :, :, :] = x[:, start_idx: start_idx + C, :, :]
            start_idx += C

        # Sigma. Use of the so-called log var trick to reparemeterize sigma > 0
        all_sigma = torch.zeros((B, K, C, H, W), device=cur_device)
        for k in range(K):
            all_sigma[:, k, :, :, :] = torch.exp(
                0.5 * torch.clamp(
                    x[:, start_idx: start_idx + C, :, :],
                    min=LOG_VAR_MIN,
                    max=LOG_VAR_MAX
                )
            )
            start_idx += C
        pdf_param = []

        for k in range(K):
            pdf_param.append(
                {
                    'mu': all_mu[:, k, :, :, :],
                    'sigma': all_sigma[:, k, :, :, :],
                }
            )

        return pdf_param    
    

"""
 The following modules are from : https://github.com/Orange-OpenSource/AIVC
 They are licenced under BSD 3-Clause "New"
 
 The licences on which our framework depends are to be found in the folder named "licence_dependency"
"""

def xavier_init(param_shape):
    """Performs manually the xavier weights initialization namely:

    Weights are sampled from N(0, sqrt(2 / nb_weights))

    Arguments:
    param_shape {[torch.Size]} --
        [A torch.Size object describing the shape of the parameters
        to initialize]

    Returns:
        [torch.tensor] -- [the initialized parameters]
    """
    param_shape = torch.Size(param_shape)
    xav_norm = torch.sqrt(torch.tensor([2.0]) / param_shape.numel())
    init_w = torch.randn(param_shape, requires_grad=True) * xav_norm
    return init_w

class BallePdfEstim(nn.Module):
    """
    This module is a small neural network which has to learn the function
    p_x_tilde (x_tilde) which is the PDF of x_tilde = x * u.

    We want a forward method which returns p_x_tilde(x_tilde) so it is further
    optimized by minimizing -log p_x_tilde (which represents both rate and neg.
    log likelihood).

    However the architecture is made to represent a CDF, namely X (not x_tilde)
    CDF denoted c_x. Two reasons for this:
    - A cdf is easier to represent because of its inherent properties (cf.
    Ballé)

    - p_x_tilde (x_tilde) = c_x(x_tilde + 0.5) - c_x(x_tilde - 0.5)

    Therefore, there is two functions: cdf which computes c_x and forward which
    computes p_x_tilde.

    When used for inference, x_tilde = x_hat = Quantization(x). In this case,
    the forward pass return p_x_hat (the discrete quantization bins proba),
    needed for entropy coding.
    """

    def __init__(self, nb_channel, pdf_family=None, verbose=True):
        """
        Replicate the architecture and computation of "Variational Image
        Compression with a Scale Hyperprior", Ballé et al 2018 (Appendix 6)
        """
        super(BallePdfEstim, self).__init__()

        self.nb_channel = nb_channel
        #self.pdf_family = pdf_family
        # Number of layers
        self.K = 4
        # Dimension of each hidden feature vector
        self.r = 3
        #print_log_msg('INFO', '__init__ BallePdfEstim', 'K', self.K)
        #print_log_msg('INFO', '__init__ BallePdfEstim', 'r', self.r)

        # Build Pdf Estimator Network
        self.matrix_h = nn.ParameterList()
        self.bias_b = nn.ParameterList()
        self.bias_a = nn.ParameterList()

        # We multiply by torch.sqrt(self.nb_channel) because xavier init
        # distribution have a variance of sqrt(2 / nb_weights) i.e:
        # sqrt(2 / (nb_channel * r_d * r_k)). However, we want to have
        # our matrix_h (3-d) as a 'list' of 2-d r_d * r_k matrix so
        # the normalisation factor is only sqrt(2 / (r_d * r_k))
        # Thus the multiplication by xav_correct
        xav_correc = torch.sqrt(torch.tensor([self.nb_channel]).float())

        for i in range(self.K):
            if i == 0:  # First Layer
                self.matrix_h.append(
                    nn.Parameter(
                        xavier_init((self.nb_channel, 1, self.r)) * xav_correc
                        )
                    )
                self.bias_a.append(
                    nn.Parameter(
                        xavier_init((self.nb_channel, self.r)) * xav_correc
                        )
                    )
                self.bias_b.append(
                    nn.Parameter(
                        xavier_init((self.nb_channel, self.r)) * xav_correc
                        )
                    )
            elif i == self.K - 1:  # Last layer
                self.matrix_h.append(
                    nn.Parameter(
                        xavier_init(
                            (self.nb_channel, self.r, 1)) * xav_correc
                        )
                    )
                self.bias_b.append(
                    nn.Parameter(
                        xavier_init((self.nb_channel, 1)) * xav_correc
                        )
                    )
            else:
                self.matrix_h.append(
                    nn.Parameter(
                        xavier_init((self.nb_channel, self.r, self.r)) * xav_correc
                        )
                    )
                self.bias_a.append(
                    nn.Parameter(
                        xavier_init((self.nb_channel, self.r)) * xav_correc
                        )
                    )
                self.bias_b.append(
                    nn.Parameter(
                        xavier_init((self.nb_channel, self.r)) * xav_correc
                        )
                    )

    def forward(self, x_tilde, pdf_param=None):
        """
        Compute p_x_tilde(x_tilde) = (p_x * U) (x_tilde)
                                   = cdf(x_tilde + 0.5) - cdf(x_tilde - 0.5)

        p_x_tilde (x_tilde) is a float tensor of shape:
            [B, C, H, W]
        (x_tilde has the same shape)
        Where:
            - B is the minibatch index
            - C is the features map index
        """
        cur_device = x_tilde.device
        # Scale factor replaces sigma if needed
        # TODO: Correct this, which no longer works with the new pdf_param
        """
        if 'sigma' in self.pdf_family.split('_'):
            scale_factor = pdf_param[0].get('sigma') / torch.sqrt(torch.tensor([2.0], device=cur_device))
        else:
            scale_factor = torch.ones_like(x_tilde, device=cur_device)
        """
        # Reshape x_tilde from [B, C, H, W] to
        # [B, C, H * W, 1]
        # Same for scale_factor
        B, C, H, W = x_tilde.size()
        x_tilde = x_tilde.view(B, C, H * W, 1)
        #scale_factor = scale_factor.view(B, C, H * W, 1)
        p_x_tilde = self.cdf((x_tilde + 0.5))\
            - self.cdf((x_tilde - 0.5))

        # [B, C, H, W]
        return p_x_tilde.view(B, C, H, W)

    def cdf(self, x_tilde):
        # tmp_var is a placeholder for different calculation results throughout
        # the for loop.
        tmp_var = x_tilde
        for i in range(self.K):
            h_softplus = nn.functional.softplus(self.matrix_h[i])

            # h_softplus dimension is: [C, D, R]
            # tmp_var dimension is [B, C, E, X]
            # Where X is the nb of 'features' in the pdf estimator (i.e r)
            # When i == 0 ==> X = 1
            # Otherwise X = self.r

            # Perform Hx with a different H (2-d) matrix for each channel of x
            # tmp_var[i, :, :] = H[i, :, :] @ x [i, :, :]
            # Without the minibatch index

            # m: batch
            # c: channel
            # e: component in the c-th channel of the m-th minibatch
            # d and r: H goes from d to r
            tmp_var = torch.einsum('bced, cdr-> bcer', [tmp_var, h_softplus])
            # tmp_var dim is: [B, C, E, X]

            # bias_b dim is [C, X], so we repeat it * columns E times to obtain
            # bias_b [C, XE] where bias_b[:, X] == bias_b[:, X + kE]
            # We then reshape it to [C, E, X] (tmp_var.size()[1:])
            tmp_var += self.bias_b[i].repeat(1, tmp_var.size()[2]).view(tmp_var.size()[1:])

            # Non linearity is different for the last layer
            if i != self.K - 1:
                # Same thing than bias_b
                tmp_var = tmp_var + torch.mul(
                    torch.tanh(
                        self.bias_a[i].repeat(1, tmp_var.size()[2]).view(tmp_var.size()[1:])
                        ),
                    torch.tanh(tmp_var)
                    )
            else:
                p_x_tilde = torch.sigmoid(tmp_var)

        return p_x_tilde
    
    
class EntropyCoder(nn.Module):
    """
    Directly estimates the rate from probability
    """
    def __init__(self):
        super(EntropyCoder, self).__init__()

    def forward(self, prob_x, x):
        # Avoid NaN and p_y_tilde > 1
        prob_x = torch.clamp(prob_x, PROBA_MIN, PROBA_MAX)
        rate = -torch.log2(prob_x)
        return rate
    
    
    
class Quantizer(nn.Module):
    """
    Quantize latents : 
        - add uniform noise in training mode
        - apply round in inference mode 
    """

    def __init__(self):
        super(Quantizer, self).__init__()

    def forward(self, x, fine_tune=False):
        cur_device = x.device
        if self.training or fine_tune:
            res = x + (torch.rand(x.size(), device=cur_device) - 0.5)
        else:
            res = torch.round(x)

        return res
    

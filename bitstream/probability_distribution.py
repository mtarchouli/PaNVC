# Software Name: PaNVC
# Copyright (c) 2023 Ateme
# Licence :  BSD-3-Clause "New" 
# This software is distributed under the BSD-3-Clause license.

import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from model.Estimate_rate import BallePdfEstim
from torch.distributions import Laplace

"""
 The following modules were inspired from https://github.com/Orange-OpenSource/AIVC
 We adapted the original code to our use-case.
 
 The licences on which our framework depends are to be found in the folder named "licence_dependency"
"""

class estimate_pdf_z(nn.Module):
    """
    Estimates the probability distribution of z for entropy coding 
    """
    def __init__(self, device, AC_MAX_VAL= 256, channel_z=64):
        super(estimate_pdf_z, self).__init__()
        self.channel_z = channel_z
        self.AC_MAX_VAL = AC_MAX_VAL
        self.device = device
        self.BallePdfEstim = BallePdfEstim(nb_channel=channel_z) 
    def forward(self, param):
        """
        Pre-compute (at the encoder side) the CDF of z for indices in 
        [-AC_MAX_VAL, AC_MAX_VAL - 1]. This is done once for all and then
        used to perform arithmetic coding, decoding.
        """
        
        nb_ft_z =  self.channel_z #64 #self.balle_pdf_estim.nb_channel

        # According to the torchac documentation, the symbols sent with entropy
        # coding are in the range [0, Lp - 2]. We have 2 * max_val value to
        # transmit, so we want: Lp - 2 = 2 * max_val
        Lp = 2 * self.AC_MAX_VAL + 2

        # We compute the CDF for all this indices
        # idx are in [-AC_MAX_VAL - 0.5, AC_MAX_VAL + 1.5]
        idx = torch.arange(Lp, device=self.device).float() - self.AC_MAX_VAL - 0.5
        
        # It is slightly different than the laplace mode, because the balle pmf
        # only accepts 4D inputs with the last dimension equals to one. Thus,
        # we consider idx as a [1, 1, -1, 1] tensor.
        idx = idx.view(1, 1, -1, 1) 
        
        # Because the cumulative are the same for a given feature map,
        # we can spare some computation by just computing them once
        # per feature map. We'll replicate the <ouput_cdf> variables
        # accross dimensions B, H and W according to what we have to transmit
        idx = idx.repeat(1, nb_ft_z, 1, 1)
        
        # Compute cdf and add back the W channel
        output_cdf = self.BallePdfEstim.cdf(idx)
        output_cdf =output_cdf.squeeze(-1).unsqueeze(-2 ).unsqueeze(-3)
        # unaccuracy. # ! Not needed anymore with the pre-computation!
        # output_cdf = torch.round(output_cdf * 1024).to(torch.int) / 1024.   
        return output_cdf
    
    def proba(self, z):                                # non parametric fully factorised  entropy model
         return  self.BallePdfEstim(z)        
            
        
class estimate_pdf_y(nn.Module):
    """
    Estimates the probability distribution of y for entropy coding 
    """
    def __init__(self, device,  AC_MAX_VAL= 256, channel_y=256):
        super(estimate_pdf_y, self).__init__()
        self.channel_y = channel_y
        self.AC_MAX_VAL = AC_MAX_VAL
        self.device = device
        
    def forward(self, sigma, mu):
        cur_device = sigma.device
        #cur_device = self.device
        B, C, H, W = sigma.size()

        # According to the torchac documentation, the symbols sent with entropy
        # coding are in the range [0, Lp - 2]. We have 2 * max_val value to
        # transmit, so we want: Lp - 2 = 2 * max_val
        Lp = 2 * self.AC_MAX_VAL + 2

        # We compute the CDF for all this indices
        # idx are in [-AC_MAX_VAL - 0.5, AC_MAX_VAL + 1.5]
        idx = torch.arange(Lp, device=cur_device).float() - self.AC_MAX_VAL - 0.5

        # Add a 5th dimension to mu and sigma
        sigma = sigma.unsqueeze(-1)
        mu = mu.unsqueeze(-1)

        # Compute the scale parameter
        b = sigma / torch.sqrt(torch.tensor([2.0], device=cur_device))
        # Centered distribution
        #mu = torch.zeros_like(b, device=cur_device)
        # Get the distribution
        my_pdf = Laplace(mu, b)

        # Compute cdf
        idx = idx.view(1, 1, 1, 1, -1).repeat(B, C, H, W, 1)
        output_cdf = my_pdf.cdf(idx)

        return output_cdf
    
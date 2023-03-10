# Software Name: PaNVC
# Copyright (c) 2023 Ateme
# Licence :  BSD-3-Clause "New" 
# This software is distributed under the BSD-3-Clause license.

import numpy as np
import torch
import torch.nn as nn
from model.Cheng2020_AI import ga, gs
from model.Cheng_hyper_ae import ha, hs
from model.Estimate_rate import EntropyCoder, Hyperprior_Proba_Model, BallePdfEstim, PdfParamParameterizer_simp, Quantizer 

class encode(nn.Module):
    """
    Encode input :
        - Main encoder
        - Hyper encoder
    """
    def __init__(self, nbr_ftr):
        super(encode, self).__init__()
        self.encoder = ga(nb_ftr=192, nb_latent=256)    
        self.priorEncoder = ha( in_ftr=256)
        self.quantizer = Quantizer()
        self.entropy_coder = EntropyCoder()
        self.pz =  BallePdfEstim(nb_channel = 64)
        self.AC_MAX_VAL = 256
        
    def forward(self, xt):
        
        latent = self.encoder(xt)
        latent = torch.clamp(latent, -self.AC_MAX_VAL, self.AC_MAX_VAL - 1)
        z = self.priorEncoder(latent)
        z = torch.clamp(z, -self.AC_MAX_VAL, self.AC_MAX_VAL - 1)
        z_hat = self.quantizer(z)
        # Compute rate of z
        p_z_hat = self.pz(z_hat)
        rate_z_hat = self.entropy_coder(p_z_hat, z_hat)
        y_hat =  self.quantizer(latent)
        return rate_z_hat, z_hat, y_hat
        
class decode(nn.Module):
    """
    Decode latents :
        - Hyper edcoder
        - Main decoder
    """
    def __init__(self, nbr_ftr):
        super(decode, self).__init__()
        self.decoder = gs(nb_ftr=192, nb_latent=256)   
        self.priorDecoder = hs(nb_ylatent = 256)
        self.pdf_param = PdfParamParameterizer_simp(256)
        
        self.py = Hyperprior_Proba_Model(channel_y = 256)
        self.quantizer = Quantizer()
        self.entropy_coder = EntropyCoder()

    def forward(self, z_hat, y_hat=None):
        z_decoded = self.priorDecoder(z_hat) #[:, :, :y_h, :y_w]
        sigma_mu = self.pdf_param(z_decoded)
        if y_hat == None:
            return sigma_mu
        else :
            # Compute rate of y 
            y_h = y_hat.size()[2]
            y_w = y_hat.size()[3]
            mu = sigma_mu[0].get('mu') [:, :, :y_h, :y_w]
            sigma = sigma_mu[0].get('sigma') [:, :, :y_h, :y_w]
            p_y_hat = self.py(y_hat, mu, sigma, device=y_hat.device)
            rate_y_hat = self.entropy_coder(p_y_hat, y_hat)

            rec_frame = self.decoder(y_hat)
            return sigma_mu, rate_y_hat, rec_frame
        
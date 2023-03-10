# Software Name: PaNVC
# Copyright (c) 2023 Ateme
# Licence :  BSD-3-Clause "New" 
# This software is distributed under the BSD-3-Clause license.

import torch
import numpy as np
from torch.nn import functional as F
from model.codec_AI import encode, decode


def run_encoder(enc, x, device=None):
    """
    Run encoder : main encoder and the hyper encoder
    """
    if device is None: 
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
    if x.size(3)%32 !=0:
        pad = 32*(int(x.size(3)/32) + 1) - x.size(3)
        x = F.pad(x, ( 0, pad, 0, 0), mode='reflect')   
    if x.size(2)%32 !=0:
        pad = 32*(int(x.size(2)/32) + 1) - x.size(2)
        x = F.pad(x, ( 0, 0, 0, pad), mode='reflect')
    rate_z, z_hat, y_hat = enc(x.to(torch.device(device)))
    rate_z = rate_z.cpu().detach()
    z_hat = z_hat.cpu().detach()
    y_hat = y_hat.cpu().detach()
    return rate_z, z_hat, y_hat

def run_decoder(dec, x_in1, x_in2=None, device=None):
    """
    Run decoder : main decoder or/and the hyper-decoder
    """
    if device is None: 
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if x_in2 != None :
        sigma_mu, rate_y_hat, rec_frame = dec(x_in1.to(torch.device(device)), x_in2.to(torch.device(device))) 
        sigma_mu[0]['mu'] = sigma_mu[0].get('mu').cpu().detach()
        sigma_mu[0]['sigma'] = sigma_mu[0].get('sigma').cpu().detach()
        rate_y_hat = rate_y_hat.cpu().detach()
        rec_frame = rec_frame.cpu().detach()
        return sigma_mu, rate_y_hat, rec_frame
    else :
        sigma_mu = dec(x_in1.to(torch.device(device)))
        sigma_mu[0]['mu'] = sigma_mu[0].get('mu').cpu().detach()
        sigma_mu[0]['sigma'] = sigma_mu[0].get('sigma').cpu().detach()
        return sigma_mu 

def load_encoder(path_to_encoder, device= None): 
    """
    load the weight of the encoder model
    """
    if device is None: 
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
    enc = encode(192).to(device) 
    checkpoints = torch.load(path_to_encoder, map_location=device) #.to(device)
    enc.load_state_dict(checkpoints['state_dict_enc'], strict=False)
    params = enc.eval()
    return enc

def load_decoder(path_to_decoder, device=None):  
    """
    load the weights of the decoder model
    """
    if device is None: 
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
    dec = decode(192).to(device) 
    checkpoints = torch.load(path_to_decoder, map_location=device) #.to(device)
    dec.load_state_dict(checkpoints['state_dict_dec'], strict=False)
    params = dec.eval()
    return dec
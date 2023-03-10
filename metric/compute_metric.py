# Software Name: PaNVC
# Copyright (c) 2023 Ateme
# Licence :  BSD-3-Clause "New" 
# This software is distributed under the BSD-3-Clause license.

import torch
import numpy as np
from metric.ms_ssim import msssim as msssim_func
from Frame_process.read_config import get_config
from Frame_process.pre_process import preprocess

def compute_metrics(seq_rec, norm):
    """
    Compute MS-SSIM and PSNR between the input sequence and reconstructed one
    """
    config = get_config('config_enc.json')
    yuv = preprocess(config)
    msssim = compute_msssim_yuv444(yuv, seq_rec)
    psnr =  compute_psnr_yuv444(yuv, seq_rec, norm=norm)
    metrics = {}
    metrics['msssim'] = np.array(msssim)
    msssim_db = -10 * (np.log(1-np.array(msssim)) / np.log(10))
    metrics['msssim_db'] = msssim_db
    metrics['psnr'] = np.array(psnr)
    return metrics

def compute_msssim_yuv444(seq_ori, seq_rec): 
    """
    Compute MS-SSIM between the input sequence and reconstructed one
    """
    msssim = msssim_func(seq_rec, seq_ori, normalize=False)
    return msssim
    
def compute_psnr_yuv444(seq_ori, seq_rec, norm=False):    
    """
    Compute PSNR between the input sequence and reconstructed one
    """
    mse = torch.mean((seq_rec - seq_ori).pow(2))
    if norm == True : 
        d  = 1    
    else :
        d  = 255
        
    psnr = 10 * (torch.log(1 * (d*d) / mse) / np.log(10))  
    return psnr 
# Software Name: PaNVC
# Copyright (c) 2023 Ateme
# Licence :  BSD-3-Clause "New" 
# This software is distributed under the BSD-3-Clause license.

import torch
import numpy as np
from torch.nn.functional import interpolate
from Frame_process.utils import add_dim, del_dim, yuv_read
from Frame_process.read_config import get_config

def yuv_420_to_444(y, u, v):
    """
    Transform YUV420 to YUV444
    """
    y, u, v = add_dim(y), add_dim(u, 2), add_dim(v, 2)
    u =  interpolate(u, scale_factor = [2,2], mode='nearest')
    v =  interpolate(v, scale_factor = [2,2], mode='nearest')
    u, v = del_dim(u), del_dim(v)
    if (y.size()!= u.size() or y.size()!=v.size()):
        u = u[:,0:y.size(1), 0:y.size(2)]
        v = v[:,0:y.size(1), 0:y.size(2)]
    yuv = torch.cat((y,u,v), dim=0)
    return yuv

def normalize_yuv(y, u, v):
    """
    Normalize y, u and v vectors
    """
    y, u, v = y.float().div(255), u.float().div(255),  v.float().div(255)
    return y, u, v

def preprocess(config) :
    """
    Pre-process input sequence:
        - Read YUV 420
        - Normmalise YUV 420
        - Trandform YUV420 to YUV 444 
    """
    y, u, v = yuv_read(config)
    y, u, v = normalize_yuv(y, u, v)
    yuv = torch.zeros((config['nbr_frame'], 3, config['h'], config['w']))
    for i in range(0, config['nbr_frame'] ):
        yuv[i,:,:,:] = yuv_420_to_444(y[i,:,:], u[i,:,:], v[i,:,:])
    return yuv


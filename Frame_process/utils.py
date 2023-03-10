# Software Name: PaNVC
# Copyright (c) 2023 Ateme
# Licence :  BSD-3-Clause "New" 
# This software is distributed under the BSD-3-Clause license.

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
import yuvio


def add_dim(x, N=1 ):
    """
    add a dimension to the x vector
    """
    for i in range(N):
        x = x.unsqueeze(0)
    return x

def del_dim(x, N=1):
    """
    delete a dimension from the x vector
    """
    for i in range(N):
        x = x.squeeze(0)
    return x

def crop_random(img, cropx, cropy, startx, starty): 
    """
    Crop a patch from an frame
    """
    x = img.shape[3]
    y = img.shape[2]
    patch = img[:, :, starty:starty+cropy,startx:startx+cropx]
    if startx+cropx > x : 
        pad = (startx+cropx) - x
        patch = F.pad(patch, ( 0, pad, 0, 0), mode='reflect')
    if starty+cropy > y :
        pad = (starty+cropy) - y
        patch = F.pad(patch, ( 0, 0, 0, pad), mode='reflect')
    return patch

def yuv_read(config):
    """
    Read YUV 420 and return y, u ,v tensors
    """
    yuv_frame = yuvio.mimread(config['yuv_path'], config['w'], config['h'], "yuv420p", index=config['start_frame'], count=config['nbr_frame'])
    y = np.zeros((config['nbr_frame'], config['h'], config['w']))
    u = np.zeros((config['nbr_frame'], int(config['h']/2), int(config['w']/2)))
    v = np.zeros((config['nbr_frame'], int(config['h']/2), int(config['w']/2)))
    for i in range(config['nbr_frame']): 
        y[i,:,:] = np.array(yuv_frame)[i][0]
        u[i,:,:] = np.array(yuv_frame)[i][1]
        v[i,:,:] = np.array(yuv_frame)[i][2]
    return torch.tensor(y), torch.tensor(u), torch.tensor(v)

def yuv_write(y, u, v):
    """
    Write YUV 420  from y, u ,v tensors
    """
    nframe = y.size(0)
    y = np.array(y)
    u = np.array(u)
    v = np.array(v)
    yuvframes = []
    #yuvframes = yuvio.frame((y, u, v), "yuv420p")
    for i in range(nframe):
        yuvframes.append(yuvio.frame((y[i,:,:], u[i,:,:], v[i,:,:]), "yuv420p"))
    yuvio.mimwrite('rec420.yuv', yuvframes)
    
    
"""
 The following modules are from : https://github.com/Orange-OpenSource/AIVC
 They are licenced under BSD 3-Clause "New"
 
 The licences on which our framework depends are to be found in the folder named "licence_dependency"
"""    

global BITSTREAM_SUFFIX
BITSTREAM_SUFFIX = ''

def get_value(key, dic, default_dic):
    """
    Return value of the entry <key> in <dic>. If it is not defined
    in dic, look into <default_dic>
    """

    v = dic.get(key)

    if v is None:
        if key in default_dic:
            v = default_dic.get(key)
        else:
            print_log_msg(
                'ERROR', 'get_param', 'key not in default_dic', key
            )

    return v    

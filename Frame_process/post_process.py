# Software Name: PaNVC
# Copyright (c) 2023 Ateme
# Licence :  BSD-3-Clause "New" 
# This software is distributed under the BSD-3-Clause license.

import torch
import numpy as np
from torch.nn.functional import interpolate
from Frame_process.utils import add_dim, del_dim, yuv_read, yuv_write
from Frame_process.read_config import get_config
global config 
config = get_config('config_enc.json')


def yuv_444_to_420(yuv444): # size of x [B, C, H, W]
    """
    Transform YUV444 to YUV420
    """
    y = yuv444[:, 0, :, :]
    u = yuv444[:, 1, :, :].unsqueeze(0)
    v = yuv444[:, 2, :, :].unsqueeze(0)
    u = interpolate( u, scale_factor=0.5, mode='bilinear', align_corners=False, recompute_scale_factor=False)
    v = interpolate( v, scale_factor=0.5, mode='bilinear', align_corners=False, recompute_scale_factor=False)
    return y, u.squeeze(0), v.squeeze(0)

def postprocess( rec_yuv):
    """
    Post-process yuv output:
        - Trandform YUV444 to YUV 420 
        - Denormmalise YUV 420
        - Write YUV 420
    """
    y, u, v = yuv_444_to_420(rec_yuv)
    y, u, v = denormalize_yuv(y, u, v)
    yuv_write(y,u,v)
    return y,u,v

def denormalize_yuv(y, u, v):
    """
    Denormalize y, u and v vectors
    """
    y, u, v = y.float().mul(255), u.float().mul(255),  v.float().mul(255)
    return y, u, v

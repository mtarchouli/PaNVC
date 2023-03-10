# Software Name: PaNVC
# Copyright (c) 2023 Ateme
# Licence :  BSD-3-Clause "New" 
# This software is distributed under the BSD-3-Clause license.

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from Frame_process.utils import crop_random
from Frame_process.pre_process import preprocess


def frame_to_patches_overlap(config, yuv_frame): 
    """
    Divide yuv sequence into overlapping patches
    """
    h_ori =  yuv_frame.size(2)
    w_ori =  yuv_frame.size(3)
    patch_w_init, patch_h_init = config['patch_w'], config['patch_h']
    overlap = config['N_overlap']
    w_pad = w_ori
    while w_pad % (patch_w_init)!= 0:
        w_pad += 1 
        if  w_pad % (patch_w_init)==0:
            break
    h_pad = h_ori
    while h_pad % (patch_h_init) != 0:
        h_pad += 1 
        if  h_pad % (patch_h_init)==0:
            break
    pad_gop = F.pad(yuv_frame, ( 0, (w_pad-w_ori), 0, (h_pad-h_ori)), mode='reflect')
    HR_gop = []
    for starty in range(0, h_pad, patch_h_init):
        for startx in range(0, w_pad, patch_w_init):
            crop_gop = crop_random(pad_gop, patch_w_init + overlap, patch_h_init + overlap, startx, starty ) 
            HR_gop.append(crop_gop)      
    allcrops =  torch.Tensor(config['nbr_frame'], len(HR_gop), 3, patch_h_init + overlap, patch_w_init + overlap )
    for idx_crop in range(len(HR_gop)):
        for n in range(config['nbr_frame']):
            allcrops[n, idx_crop,:,:,:] = HR_gop[idx_crop][n,:,:, :]
    return allcrops

def patches_to_frame_overlap(rec_patches, config):
    """
    Gather patches to reconstruct  yuv sequence
    """
    w_ori, h_ori, overlap, patch_w_init, patch_h_init = config['w'], config['h'], config['N_overlap'], config['patch_w'], config['patch_h']
    w_pad = w_ori
    while w_pad % (patch_w_init)!= 0:
        w_pad += 1 
        if  w_pad % (patch_w_init)==0:
            break
    h_pad = h_ori
    while h_pad % (patch_h_init) != 0:
        h_pad += 1 
        if  h_pad % (patch_h_init)==0:
            break
    
    len_wpatches = int(w_pad/patch_w_init)
    len_hpatches = int(h_pad/patch_h_init)
    rec_frame = torch.Tensor(config['nbr_frame'], 3, len_hpatches*(patch_h_init+overlap), len_wpatches*(patch_w_init+overlap))
    for j in range(0, len_hpatches):
        cpt_j = j * patch_h_init
        for i in range(0, len_wpatches):
            cpt_i = i * patch_w_init  
            idx_patch =i + j * len_wpatches
            rec_patches[:, idx_patch] = rec_patches[:, idx_patch].cpu().detach()
            patch = rec_patches[:, idx_patch]
            patch_h = patch.size(2)  
            patch_w = patch.size(3) 
            if  i != 0 :
                for pix in range(overlap):
                    patch[:,:,:, pix] = (pix/(overlap-1))*patch[:,:,:, pix]+ (1-pix/(overlap-1))*rec_frame[:,:, cpt_j:cpt_j+ patch_h, cpt_i+ pix]
            if  j != 0 :
                for pix in range(overlap):
                    patch[:,:,pix, :] = (pix/(overlap-1))*patch[:, :, pix,:]+ (1-pix/(overlap-1))*rec_frame[:, :, cpt_j+ pix, cpt_i:cpt_i+ patch_w]
            rec_frame[:, :, cpt_j:cpt_j+ patch_h, cpt_i:cpt_i+ patch_w] = patch 
    rec_frame = rec_frame[:, :, 0:config['h'], 0:config['w']]
    cat_frames = {}
    cat_frames['y'] = rec_frame[:,0,:,:]
    cat_frames['u'] = rec_frame[:,1,:,:]
    cat_frames['v'] = rec_frame[:,2,:,:]
    return rec_frame, cat_frames
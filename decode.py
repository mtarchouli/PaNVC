# Software Name: PaNVC
# Copyright (c) 2023 Ateme
# Licence :  BSD-3-Clause "New" 
# This software is distributed under the BSD-3-Clause license.

import torch
import os
import numpy as np
from Frame_process.load_model import load_decoder, run_decoder
from Frame_process.post_process import postprocess
from Frame_process.utils import get_value, add_dim
from Frame_process.patch import patches_to_frame_overlap
from Frame_process.read_config import get_config
from bitstream.bitstream_patch_overlapping import ArithmeticCoder
from metric.compute_metric import compute_metrics
import time 
import logging
import Frame_process.setting
import json


def get_pad(patch_w, patch_h):
    """
     The learned model requires input resolutions divisible by 32
     If the input is not divisible by 32, padding  is applied. 
     This function return the resolution of the model input after the padding. 
    """
    if patch_w%32 !=0:
        pad_w = 32*(int(patch_w/32) + 1) - patch_w
    else:
        pad_w = 0
    if patch_h%32 !=0:
        pad_h = 32*(int(patch_h/32) + 1) - patch_h 
    else :
        pad_h = 0
    return pad_w, pad_h

def decode_z(config, flag_md5sum): 
    """
    This function decode the bistream z and apply the hyper decoder. 
    It returns sigma and mu of all patches of the sequence
    """
    AE_z = Frame_process.setting.AE_z
    ############# Decode bitstream z  of the first frame along with the header of z ##################### 
    rec_z_patch_0 = AE_z.decode_one_img(config, {
                                        'mode': 'pmf',    
                                        'bitstream_path': "./bin/bitstream_z_Frame_"+ str(0) + ".bin",
                                        # data_dim of the first patch is deduced from the header
                                        'data_dim': None,
                                        # Entropy coding is run on cpu
                                        'device': 'cpu',
                                        'flag_debug': True,
                                        'latent_name': 'z',
                                        'flag_md5sum': flag_md5sum,
                                        'isFirstFr' : True,
                                        })
    ############# get information translitted in the header of z  ##########################
    header_z = json.load(open("header_z.json"))
    nframe = header_z['nbr_frame']
    Wz, Hz = header_z['W_z'], header_z['H_z']
    rec_z_patches = torch.zeros((nframe, 64, Hz, Wz))
    rec_z_patches = add_dim(rec_z_patch_0)
    ############# Decode bitstream z  of the rest of the frames ##################### 
    for i in range(1, nframe):
        rec_z_patche_i = AE_z.decode_one_img(config, {
                                             'mode': 'pmf',
                                             'bitstream_path': "./bin/bitstream_z_Frame_"+ str(i) + ".bin",
                                             'data_dim': (1, 64, Hz, Wz),
                                             'device': 'cpu',
                                             'flag_debug': True,
                                             'latent_name': 'z',
                                             'flag_md5sum': flag_md5sum,
                                             'isFirstFr' : False,
                                             })

        rec_z_patches = torch.cat((rec_z_patches, add_dim(rec_z_patche_i)),0 )
    ################################# Load Hyper-decoder to decode z  ################
    PATH_dec = config["model_path_dec"]
    hyper_dec = load_decoder(PATH_dec, device=config['device'])
    nframe = rec_z_patches.size(0)
    len_patches = rec_z_patches.size(1)
    N_parallel = config['N_parallel']
    ################################# Run  Hyper-decoder on all patches using parallelization to get sigma and mu ################
    for i in range(nframe):
        for b in range(0, len_patches, N_parallel):
            if b + N_parallel > len_patches : 
                sigma_mu = run_decoder(hyper_dec, rec_z_patches[i,b:len_patches,:,:,:], x_in2=None, device=config['device'])
                mu_all_p = torch.cat((mu_all_p, sigma_mu[0].get('mu')), 0)
                sigma_all_p = torch.cat((sigma_all_p, sigma_mu[0].get('sigma')), 0)
            else :
                sigma_mu = run_decoder(hyper_dec, rec_z_patches[i,b:b+N_parallel,:,:,:], x_in2=None, device=config['device'])

                if b  == 0 : 
                    mu_all_p = sigma_mu[0].get('mu')
                    sigma_all_p = sigma_mu[0].get('sigma')
                else  : 
                    mu_all_p = torch.cat((mu_all_p, sigma_mu[0].get('mu')), 0)
                    sigma_all_p = torch.cat((sigma_all_p, sigma_mu[0].get('sigma')), 0)

        if i == 0 : 
            mu_all_f = add_dim(mu_all_p)
            sigma_all_f = add_dim(sigma_all_p)
        else :
            mu_all_p = add_dim(mu_all_p)
            mu_all_f = torch.cat((mu_all_f, mu_all_p), 0)
            sigma_all_p = add_dim(sigma_all_p)
            sigma_all_f = torch.cat((sigma_all_f, sigma_all_p), 0)
    mu_all_f, sigma_all_f = mu_all_f.cpu().detach(),  sigma_all_f.cpu().detach()
    Hy, Wy = header_z['H_y'], header_z['W_y']
    sigma_all_f = sigma_all_f[:,:,:,:Hy, :Wy]
    mu_all_f = mu_all_f[:,:,:,:Hy, :Wy]
    return mu_all_f, sigma_all_f, rec_z_patches

    
def Decode_seq():
    """
    Apply decoding, post-processing and write the output YUV
    Return the decoded sequence.
    """
    logger = logging.getLogger("Decoding")
    stdhandler = logging.StreamHandler()
    stdhandler.setLevel(logging.INFO)
    logger.addHandler(stdhandler)
    logger.setLevel(logging.INFO)
    logger.info("#"*20 + '         Decoding Starts        ' + "#"*20)
    config = get_config()
    flag_md5sum = config["flag_md5sum"]
    
    logger.info("#"*20 + '        Decoding Bistream z     ' + "#"*20)
    mu_all_f, sigma_all_f, rec_z_patches =  decode_z(config, flag_md5sum)
    logger.info("#"*20 + '        Decoding Bistream y     ' + "#"*20)
    ############# Decode bitstream y  of the first frame along with the header of y  ##################### 
    AE_y = Frame_process.setting.AE_y
    rec_y0_patches = AE_y.decode_one_img(config, {
            'mode': 'laplace',
            'sigma': sigma_all_f[0,:,:,:,:],
            'mu': mu_all_f[0,:,:,:,:],     
            'bitstream_path': "./bin/bitstream_y_Frame_0.bin",
            'data_dim':sigma_all_f[0,:,:,:,:].size(),
            'device': 'cpu',
            'flag_debug': True,
            'latent_name': 'y',
            'flag_md5sum': flag_md5sum,
            'isFirstFr': True,
        })
    ############# get information transmitted in the header of y  ##########################
    config_dec = json.load(open("header.json"))
    config = {**config, **config_dec}
    nframe = config['nbr_frame']
    #### get len_patches ###
    w_pad = config['w']
    while w_pad % (config['patch_w'] )!= 0:
        w_pad += 1 
        if  w_pad % (config['patch_w'])==0:
            break
    h_pad = config['h']
    while h_pad % (config['patch_h']) != 0:
        h_pad += 1 
        if  h_pad % (config['patch_h'])==0:
            break
            
    len_wpatches = int(w_pad/config['patch_w'])
    len_hpatches = int(h_pad/config['patch_h'])
    len_patches = len_wpatches* len_hpatches
    N_parallel = config["N_parallel"]
    rec_y_patches = torch.zeros(nframe, rec_y0_patches.size(0),  rec_y0_patches.size(1),  rec_y0_patches.size(2),  rec_y0_patches.size(3))
    rec_y_patches[0,:,:,:,:] = rec_y0_patches
    ############# Decode bitstream y  of the rest of the frames  ##################### 
    for i in range(1, nframe):
        rec_y_patches[i,:,:,:,:] = AE_y.decode_one_img(config, {
            'mode': 'laplace',
            'sigma': sigma_all_f[i,:,:,:,:],
            'mu': mu_all_f[i,:,:,:,:],     
            'bitstream_path': "./bin/bitstream_y_Frame_"+str(i) + ".bin",
            'data_dim':sigma_all_f[i,:,:,:,:].size(),
            'device': 'cpu',
            'flag_debug': True,
            'latent_name': 'y',
            'flag_md5sum': flag_md5sum,
            'isFirstFr': False,
        })
    ############################################ Decode patches using parallelization  ################################################
    logger.info("#"*20 + '        Load Decoder         ' + "#"*20)
    PATH_dec = config["model_path_dec"]
    decoder = load_decoder(PATH_dec, device=config['device'])
    logger.info("#"*20 + '        Decoding Patches     ' + "#"*20)
    DT_runDec_Total = 0
    for i in range(nframe):
        for b in range(0, len_patches, N_parallel):
            if b + N_parallel > len_patches : 
                DT_runDec = time.time()
                _, rate_y_hat, rec_patch =  run_decoder(decoder, rec_z_patches[i,b:len_patches,:,:,:], rec_y_patches[i,b:len_patches,:,:,:], device=config['device'])
                DT_runDec = time.time() - DT_runDec
                pad_w, pad_h = get_pad( config['patch_w']+config['N_overlap'], config['patch_h']+config['N_overlap'])
                rec_patch =rec_patch [:,:,0:rec_patch.size(2)-pad_h, 0:rec_patch.size(3)-pad_w]
                rate_y_hat_all_p = torch.cat((rate_y_hat_all_p, rate_y_hat ), 0)
                rec_all_p = torch.cat((rec_all_p, rec_patch),0)
            else :
                DT_runDec = time.time() 
                _, rate_y_hat, rec_patch =  run_decoder(decoder, rec_z_patches[i,b:b+N_parallel,:,:,:], rec_y_patches[i,b:b+N_parallel,:,:,:], device=config['device'])
                DT_runDec = time.time() - DT_runDec
                pad_w, pad_h = get_pad(config['patch_w']+config['N_overlap'], config['patch_h']+config['N_overlap'])
                rec_patch =rec_patch [:,:,0:rec_patch.size(2)-pad_h, 0:rec_patch.size(3)-pad_w]
                if b  == 0 : 
                    rec_all_p = rec_patch
                    rate_y_hat_all_p = rate_y_hat
                else  : 
                    rec_all_p = torch.cat((rec_all_p, rec_patch))
                    rate_y_hat_all_p = torch.cat((rate_y_hat_all_p, rate_y_hat))
            DT_runDec_Total += DT_runDec
        if i == 0 : 
            rec_all_f = add_dim(rec_all_p)
            rate_y_hat_all_f = add_dim(rate_y_hat_all_p)
        else :
            rec_all_p = add_dim(rec_all_p)
            rec_all_f = torch.cat((rec_all_f, rec_all_p), 0)
            rate_y_hat_all_p = add_dim(rate_y_hat_all_p)
            rate_y_hat_all_f = torch.cat((rate_y_hat_all_f, rate_y_hat_all_p), 0)
            
    rec_all_f, rate_y_hat_all_f = rec_all_f.cpu().detach(),  rate_y_hat_all_f.cpu().detach()
    #################### Gather patches to recontruct the decoded sequence ##########################
    logger.info("#"*20 + '        Gathering Patches     ' + "#"*20)
    rec_seq, _ = patches_to_frame_overlap(rec_all_f, config)
    ###################################### Post-process ###########################################
    DT_postpro = time.time()
    logger.info("#"*20 + '        Post-Processing     ' + "#"*20)
    y,u,v = postprocess(rec_seq)
    DT_postpro =  time.time() - DT_postpro
    
    #Print deocding time which include : time of running decoder model and post-processing
    Dec_time = DT_runDec_Total + DT_postpro
    print(' Decoding Time [s]:',  Dec_time)
    
    return rec_y_patches, rec_all_f, rec_seq



if __name__ == '__main__':
    _, _, rec_seq = Decode_seq()
    ###################################### Compute metrics #########################
    metrics = compute_metrics(rec_seq, norm=True)
    print (' MS-SSIM          : ', metrics['msssim']) 
    print (' MS-SSIM      [db]: ', metrics['msssim_db']) 
    print (' PSNR         [db]: ',  metrics['psnr'])

    
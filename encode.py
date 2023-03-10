# Software Name: PaNVC
# Copyright (c) 2023 Ateme
# Licence :  BSD-3-Clause "New" 
# This software is distributed under the BSD-3-Clause license.

import torch
import math
import os
import sys
import numpy as np
from Frame_process.load_model import load_encoder, load_decoder, run_encoder, run_decoder
from Frame_process.pre_process import preprocess
from Frame_process.utils import get_value, add_dim
from Frame_process.patch import frame_to_patches_overlap
from Frame_process.read_config import get_config, log_config
from bitstream.bitstream_patch_overlapping import ArithmeticCoder
import Frame_process.setting
import json
import time
import logging

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
        
def Encode_seq():
    """
    Encoding sequence 
    write bitstream z and y 
    """
    ########################## Read config ##########################
    config_file =  'config_enc.json'
    config = get_config(config_file)
    logger = log_config(config)
    flag_md5sum = config["flag_md5sum"]
    ######################### Pre-processing sequence ################
    logger.info('\n'+ '#'*20 + ' Pre-processing sequence    '+ '#'*20)
    ET_prepro = time.time()
    yuv = preprocess(config)
    ET_prepro = time.time() - ET_prepro
    logger.info('\n' + '#'*20 + ' Divide frames into patches ' + '#'*20)
    ET_DivPatch = time.time()
    ###################### Divide sequence into patches #############
    patches = frame_to_patches_overlap(config, yuv)
    ET_DivPatch = time.time() - ET_DivPatch
    ###################### Load encoder ############################
    PATH_enc = config["model_path_enc"]
    #print('#'*20, ' Load Encoder ', '#'*20)
    logger.info('\n'+ '#'*20 + '      Load Encoder          '+ '#'*20 )
    encoder = load_encoder(PATH_enc, device=config['device'])
    ###################### Encoding patches using paralellization #############
    nframe = config["nbr_frame"]
    len_patches = patches.size(1)
    N_parallel = config["N_parallel"]
    logger.info('\n' + '#'*20 + "      Patch-Encoding       " + '#'*20)
    y_hat_list = []
    z_hat_list = []
    ET_runEnc_Total = 0
    for i in range(nframe):
        for b in range(0, len_patches, N_parallel):
            if b + N_parallel > len_patches : 
                ET_runEnc = time.time()
                rate_z, z_hat, y_hat = run_encoder(encoder, patches[i,b:len_patches,:,:,:], device=config['device'])
                ET_runEnc = time.time()- ET_runEnc
                y_hat_all_p = torch.cat((y_hat_all_p, y_hat), 0)
                z_hat_all_p = torch.cat((z_hat_all_p, z_hat), 0) 
                rate_z_all_p = torch.cat((rate_z_all_p, rate_z), 0)
            else :
                ET_runEnc = time.time()
                rate_z, z_hat, y_hat = run_encoder(encoder, patches[i,b:b+N_parallel,:,:,:],  device=config['device'])
                ET_runEnc = time.time()- ET_runEnc
                if b  == 0 : 
                    y_hat_all_p = y_hat
                    z_hat_all_p = z_hat
                    rate_z_all_p = rate_z
                else  : 
                    y_hat_all_p = torch.cat((y_hat_all_p, y_hat), 0)
                    z_hat_all_p = torch.cat((z_hat_all_p, z_hat), 0)
                    rate_z_all_p = torch.cat((rate_z_all_p, rate_z), 0)
            ET_runEnc_Total +=  ET_runEnc
        if i == 0 : 
            y_hat_all_f = add_dim(y_hat_all_p)
            z_hat_all_f = add_dim(z_hat_all_p)
            rate_z_all_f = add_dim(rate_z_all_p)
        else :
            y_hat_all_p = add_dim(y_hat_all_p)
            y_hat_all_f = torch.cat((y_hat_all_f, y_hat_all_p), 0)
            z_hat_all_p = add_dim(z_hat_all_p)
            z_hat_all_f = torch.cat((z_hat_all_f, z_hat_all_p), 0)
            rate_z_all_p = add_dim(rate_z_all_p)
            rate_z_all_f = torch.cat((rate_z_all_f, rate_z_all_p), 0)
    y_hat_all_f, z_hat_all_f, rate_z_all_f = y_hat_all_f.cpu().detach(), z_hat_all_f.cpu().detach(), rate_z_all_f.cpu().detach()
    logger.info('\n'+ '#'*20 +'      Write Bitream z       ' + '#'*20 )
    
    ################################## Write Bitstream  z of all frames  ##########################
    AE_z = Frame_process.setting.AE_z
    rec_z_patches = torch.zeros_like(z_hat_all_f)
    
    if not os.path.exists("./bin/"):
        os.mkdir('./bin/')
    elif os.path.exists("./bin/"):
        os.system('rm ./bin/*')

    for i in range(nframe):
        if i == 0:
            # Write bitstream  z of the first patch  along of the header of z 
            byte_z_img =  AE_z.encode_one_img(config, {
                            'x': z_hat_all_f[i,:,:,:,:],
                            'mode': 'pmf',
                            # y_dim to be written in the header 
                            'y_dim' : y_hat_all_f[i,:,:,:,:].size(),
                            'bitstream_path': "./bin/bitstream_z_Frame_"+ str(i) + ".bin",
                            'flag_debug': True,
                            'latent_name': 'z',
                            'flag_md5sum': flag_md5sum,                      
                            'isFirstFr' : True,
                        })
        else : 
            # Write bitstream z of the rest of the patches 
            byte_z_img =  AE_z.encode_one_img(config, {
                            'x': z_hat_all_f[i,:,:,:,:],
                            'mode': 'pmf',
                            'bitstream_path': "./bin/bitstream_z_Frame_"+ str(i) + ".bin",
                            'flag_debug': True,
                            'latent_name': 'z',
                            'flag_md5sum': flag_md5sum,
                            'isFirstFr' : False,
                        })
    ################################## Read Bitstream  z of all frames ##########################
    logger.info('\n'+ '#'*20 +'  Read bitstream z   ' + '#'*20 )
    mu_all_f, sigma_all_f, rec_z_patches = decode_z(config, flag_md5sum) 
    
    ################################## Write Bitstream  y of all frames ##########################
    logger.info('\n'+ '#'*20 + '  Write bitstream y  '+ '#'*20)
    AE_y = Frame_process.setting.AE_y
    for i in range(nframe):
        if i == 0 : 
            # Write bitstream y of the first frame along with the header 
            byte_y_img =  AE_y.encode_one_img(config, {
                    'x': y_hat_all_f[i,:,:,:,:],
                    'mode': 'laplace',
                    'mu': mu_all_f[i,:,:,:,:],
                    'sigma': sigma_all_f[i,:,:,:,:],
                    'bitstream_path': "./bin/bitstream_y_Frame_"+str(i)+".bin",
                    'flag_debug': True,
                    'latent_name': 'y',
                    'flag_md5sum': flag_md5sum,
                    # Write header at the first frame
                    'isFirstFr': True,
                })
        else:
            # Write bitstream y of the rest of the frames 
            byte_y_img =  AE_y.encode_one_img(config, {
                    'x': y_hat_all_f[i,:,:,:,:],
                    'mode': 'laplace',
                    'mu': mu_all_f[i,:,:,:,:],
                    'sigma': sigma_all_f[i,:,:,:,:],
                    'bitstream_path': "./bin/bitstream_y_Frame_"+str(i)+".bin",
                    'flag_debug': True,
                    'latent_name': 'y',
                    'flag_md5sum': flag_md5sum,
                    'isFirstFr': False,
                })
    # print encoding time which enclude : time of pre-processing, dividing into patches and running encoder model
    Enc_time = ET_prepro + ET_DivPatch + ET_runEnc_Total
    print('Encoding Time [s] : ', Enc_time)

if __name__ == '__main__':
    Encode_seq()
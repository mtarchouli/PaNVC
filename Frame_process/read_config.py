# Software Name: PaNVC
# Copyright (c) 2023 Ateme
# Licence :  BSD-3-Clause "New" 
# This software is distributed under the BSD-3-Clause license.

import json
import logging

def extract_info_seq(yuv_path):
    """
    extract name, sequence resolution and frame-rate from yuv path
    yuv_path : path to sequence, must be name_wxh_frate_yuvformat.yuv
    """
    s = yuv_path.split('/')
    info = s[len(s)-1].split('_')
    name = info[0]
    resolution =info[1]
    w = int(resolution.split('x')[0])
    h = int(resolution.split('x')[1])
    frate = int(info[2])
    input_format = info[3][0:3]
    info={}
    info['seq_name'] = name
    info['w'] = w 
    info['h'] = h 
    info['frame_rate'] = frate 
    info['input_format'] = input_format
    return info

def read_config_enc(config_path):
    """
    Read user encoding configuration 
    Those information are to be transmitted in the bitstream
    """
    config = json.load(open(config_path))
    param = {}
    if 'yuv_path' in config:
        param['yuv_path']=config['yuv_path']
        info = extract_info_seq(param['yuv_path'])
        param = {**param, **info}
    if 'GOP_type' in config:
        param['GOP_type']=config['GOP_type']
    if 'GOP_size' in config:
        param['GOP_size']=config['GOP_size']
    if 'N_overlap' in config:
        param['N_overlap']=config['N_overlap']
    if 'patch_w' in config:
        param['patch_w']=config['patch_w']
    if 'patch_h' in config:
        param['patch_h']=config['patch_h']
    if 'start_frame' in config:
        param['start_frame']=config['start_frame'] 
    if 'nbr_frame' in config:
        param['nbr_frame']=config['nbr_frame']  
    if 'lambda' in config:
        param['lambda'] = config['lambda']  
    return param    

def read_config_dec(config_path):
    """
    Read  configuration fixed for both encoder and decoder
    Those information are not to be transmitted in the bitstream
    """
    config = json.load(open(config_path))
    param = {}
    if 'N_parallel' in config:
        param['N_parallel']=config['N_parallel']
    if 'model_path_enc' in config:
        param['model_path_enc']=config['model_path_enc']
        s = param['model_path_enc'].split('_')
        param['metric'] = s[1]
    if 'model_path_dec' in config:
        param['model_path_dec']=config['model_path_dec']
        s = param['model_path_dec'].split('_')
        param['metric'] = s[1]
    if 'flag_md5sum' in config:
        param['flag_md5sum'] = config['flag_md5sum']
    if 'device' in config:
        param['device'] = config['device']
    return param 

def log_config(config, config_path='config_enc.json'):
    """
    Write logs
    """
    logger = logging.getLogger("Encoding")
    stdhandler = logging.StreamHandler()
    stdhandler.setLevel(logging.INFO)
    logger.addHandler(stdhandler)
    logger.setLevel(logging.INFO)
    logger.info("Encoding Config :")
    logger.info("    YUV sequence     : "  + str(config['yuv_path']))
    logger.info("    Encoding Mode    : "  + str(config['GOP_type']))
    logger.info("    GOP size         : "  + str(config['GOP_size']))
    logger.info("    Patch width      : "  + str(config['patch_w']))
    logger.info("    Patch height     : "  + str(config['patch_h']))
    logger.info("    Lambda           : "  + str(config['lambda'] ))  
    logger.info("    Start frame      : "  + str(config['start_frame']))
    logger.info("    Number of frames : "  + str(config['nbr_frame']))

    logger.info('Few Constraints :')
    logger.info('  YUV sequence should be named in this format SeqName_WxH_FrameRate_420.yuv')
    logger.info('  GOP_type should be All_Intra or Inter ')
    logger.info('  Encoder and Decoder should correspond to the same Lambda and metric')
    if config['GOP_type'] == 'Inter':
        logger.error('Inter mode is not supported yet')
        
    if config['GOP_size'] != 1:
        logger.error('Only All Intra mode is supported so gop size is automaticaly 1')
        
    if not(config['lambda'] in [420, 220, 120, 64]) and config['metric'] == 'msssim' :
        logger.error('Error in Lambda Value ')
        logger.warning('Warning : For Ms-ssim models lambda should be {420, 220, 120, 64}')
        
    if not(config['lambda'] in [4096, 3140, 2048, 1024]) and config['metric'] == 'mse' :
        logger.error('Error in Lambda Value')
        logger.warning('Warning : For MSE models lambda should be {4096, 3140, 2048, 1024}')
    return logger

def get_config( enc_config=None, dec_config="config_dec.json"):
    """
    Get configuration from enc_config and/or dec_config
    """
    if enc_config != None :
        param_enc  = read_config_enc(enc_config)
    param_dec = read_config_dec(dec_config)
    if enc_config != None :
        config = {**param_enc, **param_dec}
        
    else:
        config = param_dec
    return config

# Software Name: PaNVC
# Copyright (c) 2023 Ateme
# Licence :  BSD-3-Clause "New" 
# This software is distributed under the BSD-3-Clause license.

import torchac
import torch
import math
import os
import sys
import json
import numpy as np
import scipy.stats
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from torch.distributions import Laplace
from bitstream.check_md5sum import write_md5sum, read_md5sum, compute_md5sum
from Frame_process.utils import get_value, BITSTREAM_SUFFIX
from bitstream.probability_distribution import estimate_pdf_z, estimate_pdf_y

"""
 The following modules were inspired from https://github.com/Orange-OpenSource/AIVC
 We modified the original code to adapt it to our use-case.
 
 The licences on which our framework depends are to be found in the folder named "licence_dependency"
"""


"""
Bitstream one frame per patch

1. z latents of all patches
        [Header z                                     (5 bytes)]
        
        [md5sum Patch 0                              (32 bytes)]
        [Header Patch 0                              (4  bytes)]
        [z_patch_0 entropy coded                     (variable)]
        
        [md5sum Patch 1                              (32 bytes)]
        [Header Patch 1                               (4 bytes)] 
        [z_patch_1 entropy coded                     (variable)]
                     .
                     .
        [md5sum Patch nframe-1                       (32 bytes)]            .
        [Header Patch nframe-1                        (4 bytes)] 
        [z_patch_nframe-1 entropy coded              (variable)]
                     
2. y latents of the patches forming one frame
        [Header y                                   (12 bytes)]
        
        [md5sum Patch 0                              (32 bytes)]
        [Header Patch 0                               (4 bytes)] 
        [y_patch_0 entropy coded                     (variable)]
        
        [md5sum Patch 1                              (32 bytes)]
        [Header Patch 1                               (4 bytes)] 
        [y_patch_1 entropy coded                     (variable)]
                     .
                     .
                     .
        [md5sum Patch nframe-1                       (32 bytes)]            .
        [Header Patch nframe-1                        (4 bytes)] 
        [y_patch_nframe-1 entropy coded              (variable)]        
"""

class ArithmeticCoder():
    def __init__(self, param):
        """
        Remark: for all the comments, [] means that the interval bounds are
        included and ][ means that they are excluded.
        """
        DEFAULT_PARAM = {
            # Number of channel of the latents
            'nb_channel': None,
            # Mode is either <laplace> or <pmf>
            'mode' : None,
            # On which device the code will run
            'device': None,
            # No value can be outside [-AC_MAX_VAL, AC_MAX_VAL]     
            'AC_MAX_VAL': 256,
        }
        self.nb_channel = get_value('nb_channel', param, DEFAULT_PARAM)
        self.mode =  get_value('mode', param, DEFAULT_PARAM)
        self.AC_MAX_VAL = get_value('AC_MAX_VAL', param, DEFAULT_PARAM)
        self.device = get_value('device', param, DEFAULT_PARAM)
        
        if self.mode == 'laplace': 
            self.pdf_y = estimate_pdf_y(self.device, self.AC_MAX_VAL, self.nb_channel)
        elif self.mode == 'pmf':
            self.pdf_z = estimate_pdf_z(self.device, self.AC_MAX_VAL, self.nb_channel)

    def compute_cdf(self, param):
        """
        Compute the CDF value of the different symbols according to a
        given distribution.
        """

        DEFAULT_PARAM = {
            # Mode is either <laplace> or <pmf>
            'mode': None,
            # If mode == 'laplace', we need the sigma and mu parameters
            'sigma': None,
            'mu': None,
            # Encoding with the PMF requires the data dimension (z.size())
            'data_dimension': None,
        }

        mode = get_value('mode', param, DEFAULT_PARAM)
        sigma = get_value('sigma', param, DEFAULT_PARAM)
        mu = get_value('mu', param, DEFAULT_PARAM)
        data_dimension = get_value('data_dimension', param, DEFAULT_PARAM)
        
        
        if mode == 'pmf':
            B, C, H, W = data_dimension
            # Add the spatial dimension to the pre-computed cdf
            output_cdf = self.pdf_z(param)
            output_cdf = output_cdf.repeat(B, 1, H, W, 1)   

        elif mode == 'laplace':
            # Compute the (quantized) scale parameter floating point sigma
            output_cdf = self.pdf_y(sigma, mu)
        
        return output_cdf
    
    def write_header_z(self, z_size, y_dim, config):
        """
        Write header of z :
            nframe : Number of frames
            z_size : size of latent z 
            y_size : size of latent y 
        return byte stream of the  above information
        """
        byte_nframe = b''
        byte_nframe = int(config['nbr_frame']).to_bytes(1, byteorder='big')
        Bz, Cz, Hz, Wz = z_size
        byte_z_size = b''
        byte_z_size = int(Hz).to_bytes(1, byteorder='big') + int(Wz).to_bytes(1, byteorder='big')
        By, Cy, Hy, Wy = y_dim
        byte_y_size = b''
        byte_y_size = int(Hy).to_bytes(1, byteorder='big') + int(Wy).to_bytes(1, byteorder='big')
        byte_header_z  = byte_nframe + byte_z_size + byte_y_size
        return byte_header_z
    
    def read_header_z(self, byte_stream):
        """
        Extract information in the header of z 

        """
        nframe = int.from_bytes(byte_stream[0:1], byteorder='big')
        byte_stream = byte_stream[1:]
        Hz =  int.from_bytes(byte_stream[0:1], byteorder='big')
        byte_stream = byte_stream[1:]
        Wz =  int.from_bytes(byte_stream[0:1], byteorder='big')
        byte_stream = byte_stream[1:]
        Hy =  int.from_bytes(byte_stream[0:1], byteorder='big')
        byte_stream = byte_stream[1:]
        Wy =  int.from_bytes(byte_stream[0:1], byteorder='big')
        byte_stream = byte_stream[1:]
        header_z ={}
        header_z['nbr_frame'] =nframe
        header_z['H_z'] = Hz
        header_z['W_z'] = Wz
        header_z['H_y'] = Hy
        header_z['W_y'] = Wy
        return header_z, byte_stream
        
    def write_header_img(self, config):
        """
        Write header of y :
            nframe : Number of frames
            GOP_Type : All_intra or Inter (currently only All_intra is supported)
            GOP_size :  1
            h : sequence height
            w : sequence width
            patch_w : patch height
            patch_h : patch width
            lambda : quality level
            N_overlap : Number of overlapping pixels
        return byte stream of the above information 
        """
        byte_pic_type = b''
        if config['GOP_type'] == 'All_Intra':
            t = 0
            byte_pic_type = t.to_bytes(1, byteorder='big')       
        byte_pic_order = b'' 
        byte_pic_res = b''
        byte_pic_res = int(config['h']).to_bytes(2, byteorder='big') + int(config['w']).to_bytes(2, byteorder='big') 
        byte_patch_res = b''
        byte_patch_res = int(config['patch_h']).to_bytes(2, byteorder='big') + int(config['patch_w']).to_bytes(2, byteorder='big') 
        byte_quality_level = b''
        if config['metric']== 'msssim':
            if config["lambda"] == 64:
                level = 0
                byte_quality_level = level.to_bytes(1, byteorder='big')
            if config["lambda"] == 120:
                level = 1
                byte_quality_level = level.to_bytes(1, byteorder='big')
            if config["lambda"] == 220:
                level = 2
                byte_quality_level = level.to_bytes(1, byteorder='big')
            if config["lambda"] == 420:
                level = 3
                byte_quality_level = level.to_bytes(1, byteorder='big')
        elif config['metric'] == 'mse':
            if config["lambda"] == 1024:
                level = 0
                byte_quality_level = level.to_bytes(1, byteorder='big')
            if config["lambda"] == 2048:
                level = 1
                byte_quality_level = level.to_bytes(1, byteorder='big')
            if config["lambda"] == 3140:
                level = 2
                byte_quality_level = level.to_bytes(1, byteorder='big')
            if config["lambda"] == 4096:
                level = 3
                byte_quality_level = level.to_bytes(1, byteorder='big')    
        byte_nframe = b''
        byte_nframe = int(config['nbr_frame']).to_bytes(1, byteorder='big')
        byte_N = b''
        byte_N = int(config['N_overlap']).to_bytes(1, byteorder='big')
        byte_header = byte_pic_type + byte_pic_res + byte_patch_res + byte_quality_level + byte_nframe + byte_N
        return byte_header      
    
    def read_header_img(self, byte_stream, config):
        """
        Extract information in the header of y 
        """
        len_bstream_init = len(byte_stream)
        pic_type = int.from_bytes(byte_stream[0:1], byteorder='big')
        if pic_type == 0:
            GOP_type = "All_Intra"
        byte_stream = byte_stream[1:]
        pic_h = int.from_bytes(byte_stream[0:2], byteorder='big')
        byte_stream = byte_stream[2:]
        pic_w = int.from_bytes(byte_stream[0:2], byteorder='big')
        byte_stream = byte_stream[2:]
        patch_h = int.from_bytes(byte_stream[0:2], byteorder='big')
        byte_stream = byte_stream[2:]
        patch_w = int.from_bytes(byte_stream[0:2], byteorder='big')
        byte_stream = byte_stream[2:]
        
        quality_level = int.from_bytes(byte_stream[0:1], byteorder='big')
        if config['metric']== 'msssim': 
            if quality_level == 0 :
                model_lambda = 64 
            if quality_level == 1 :
                model_lambda = 120 
            if quality_level == 2 :
                model_lambda = 220 
            if quality_level == 3 :
                model_lambda = 420 
        if config['metric']== 'mse': 
            if quality_level == 0 :
                model_lambda = 1024 
            if quality_level == 1 :
                model_lambda = 2048 
            if quality_level == 2 :
                model_lambda =  3140
            if quality_level == 3 :
                model_lambda = 4096 
        byte_stream = byte_stream[1:]
        
        nframe = int.from_bytes(byte_stream[0:1], byteorder='big')
        byte_stream = byte_stream[1:]
        header = {}
        N  = int.from_bytes(byte_stream[0:1], byteorder='big') #byte_stream[0], byteorder='big'
        byte_stream = byte_stream[1:]
        header['N_overlap'] = N
        len_header = len_bstream_init -len(byte_stream)
        
        header['GOP_type'] = GOP_type
        header['h'] = pic_h
        header['w'] = pic_w
        header['patch_h'] = patch_h
        header['patch_w'] = patch_w
        header['lambda'] = model_lambda
        header['nbr_frame'] = nframe
        return header, len_header, byte_stream
        
    def encode_one_patch(self, param):
        """
        Encode the features of y (mode laplace) or z (mode pmf) ), for one patch
        return the byte stream of those features. 
        """
        DEFAULT_PARAM = {
            # The 4-dimensional tensor to encode
            'x': None,
            # Mode is either <laplace> or <pmf>
            'mode': None,
            # If mode == 'laplace', we need the parameters mu and sigma
            'mu': None,
            'sigma': None,
            # Name (absolute path) of the bitstream
            'bitstream_path': None,
            # Debug. If true, print specific stuff
            'flag_debug': True,
            # Latent Name y ou z .
            'latent_name': '',
            # Compute a feature-wise md5sum by saving it in a temporary file.
            # Include this md5sum in the bitstream
            'flag_md5sum': False,
        }

        x = get_value('x', param, DEFAULT_PARAM) # latent of one patch example x[0, :,:,:]
        mode = get_value('mode', param, DEFAULT_PARAM)
        sigma = get_value('sigma', param, DEFAULT_PARAM)
        mu = get_value('mu', param, DEFAULT_PARAM)
        bitstream_path = get_value('bitstream_path', param, DEFAULT_PARAM)
        flag_debug     = get_value('flag_debug', param, DEFAULT_PARAM)
        latent_name    = get_value('latent_name', param, DEFAULT_PARAM)
        flag_md5sum    = get_value('flag_md5sum', param, DEFAULT_PARAM)
        
        byte_to_write = b''
        if not(bitstream_path.endswith(BITSTREAM_SUFFIX)):
            bitstream_path += BITSTREAM_SUFFIX

        x = x.unsqueeze(0)
        
       # Compute cdf of the features, necessary for entropy coding 
        if mode == 'laplace':
            sigma = sigma.unsqueeze(0)
            mu = mu.unsqueeze(0)
            # Compute the CDF for all symbols   
            output_cdf = self.compute_cdf({
                'mode': mode,
                'sigma': sigma,
                'mu': mu,
                'data_dimension': x.size(),
            })
        elif mode == 'pmf':
            # Compute the CDF for all symbols
            output_cdf = self.compute_cdf({
                'mode': mode,
                'data_dimension': x.size(),
            })
        
        header_overhead = len(byte_to_write)
        # CDF must be on CPU
        output_cdf = output_cdf.cpu()

        # Shift x from [-max_val, max_val - 1] to [0, 2 * max_val -1]
        symbol = (x + self.AC_MAX_VAL).cpu().to(torch.int16)
        entropy_coded_byte = torchac.encode_float_cdf(output_cdf, symbol, check_input_bounds=True)
        
        # Add the entropy coded part to the bytestream
        byte_to_write += entropy_coded_byte
        
        # Append the number of bytes at the beginning
        byte_to_write = len(byte_to_write).to_bytes(4, byteorder='big') + byte_to_write
        return byte_to_write
        
    def encode_one_img(self, config, param):
        """
        Write bitstream (y or z) of the patches forming one frame
        """
        DEFAULT_PARAM = {
            # The 4-dimensional tensor to encode
            'x': None,
            # Mode is either <laplace> or <pmf>
            'mode': None,
            # If mode == 'laplace', we need the parameters mu and sigma
            'mu': None,
            'sigma': None,
            # If mode == 'pmf ', we need the latent y size to write it in the header
            'y_dim': None,
            # Name (absolute path) of the bitstream
            'bitstream_path': None,
            # Debug. If true, print specific stuff
            'flag_debug': True,
            # Latent Name y or z
            'latent_name': None,
            # Compute a feature-wise md5sum by saving it in a temporary file.
            # Include this md5sum in the bitstream
            'flag_md5sum': False,
            # To write the header in the bitstream, the encoder needs to know if it's the first frame or not. 
            'isFirstFr': False,
        }
        x = get_value('x', param, DEFAULT_PARAM)
        mode = get_value('mode', param, DEFAULT_PARAM)
        mu = get_value('mu', param, DEFAULT_PARAM)
        sigma = get_value('sigma', param, DEFAULT_PARAM)
        y_dim = get_value('y_dim', param, DEFAULT_PARAM)
        isFirstFr = get_value('isFirstFr', param, DEFAULT_PARAM)
        
        bitstream_path = get_value('bitstream_path', param, DEFAULT_PARAM)
        flag_debug = get_value('flag_debug', param, DEFAULT_PARAM)
        latent_name = get_value('latent_name', param, DEFAULT_PARAM)
        flag_md5sum = get_value('flag_md5sum', param, DEFAULT_PARAM)
        
        # Write header if it's the first frame. The header (y or z) is written one for the whole sequence
        if mode == 'laplace' and isFirstFr == True :
            byte_header = self.write_header_img(config)
        elif mode == 'pmf' and isFirstFr == True  :
            byte_header = self.write_header_z(x.size(),y_dim, config)
        else : 
            byte_header  = b''
            
        nbr_patches = x.shape[0]
        byte_img = b''
        byte_img += byte_header 
        
        # Encode features (y or z) of the patches forming one frame 
        for i in range(nbr_patches):
            # encode md5sum for each patch if it's flag_md5sum is true
            if flag_md5sum:
                x_encoder = x[i,:,:,:].to(self.device).to(torch.int16).numpy().astype(int)
                np.savetxt( './tmp_tensor.npy', x_encoder.flatten())
                encoder_md5sum = compute_md5sum({'in_file': './tmp_tensor.npy'}).encode()
                byte_img += encoder_md5sum
                os.system('rm ./tmp_tensor.npy')
            # Encode features y of the patches forming one frame if mode is laplace
            if mode == 'laplace':
                byte_img += self.encode_one_patch({
                'x': x[i,:,:,:],
                'mode': 'laplace',
                'mu': mu[i,:,:,:],
                'sigma': sigma[i,:,:,:],
                'bitstream_path': bitstream_path,
                'flag_debug': True,
                'latent_name': 'y',
                'flag_md5sum': True, })   
            # Encode features z of the patches forming one frame if mode is pmf
            elif mode == 'pmf':
                byte_img += self.encode_one_patch({
                'x': x[i,:,:,:],
                'mode': 'pmf',
                'mu': None ,
                'sigma': None,
                'bitstream_path': bitstream_path,
                'flag_debug': True,
                'latent_name': 'z',
                'flag_md5sum': True, })
                
        header_overhead = len(byte_img)
        if os.path.isfile(bitstream_path):
            os.system('rm ' +  bitstream_path)
        old_size = 0
        
        # write bitstream file 
        with open(bitstream_path, 'ab') as fout:
            fout.write(byte_img)
        new_size = os.path.getsize(bitstream_path)
        
        # Check wether we have the same rate as in real life
        if flag_debug:
            if mode == 'laplace':
                # Compute the theoretical rate
                b = sigma / torch.sqrt(torch.tensor([2.0], device=x.device))
                my_pdf = Laplace(mu, b)
                proba = torch.clamp(my_pdf.cdf(x + 0.5) - my_pdf.cdf(x - 0.5), 2 ** -16, 1.)
            elif mode == 'pmf':
                proba = torch.clamp(self.pdf_z.proba(x), 2 ** -16, 1.)
                
            estimated_rate = (-torch.log2(proba).sum() / (8000)).item() + 1e-3 # Avoid having a perfect zero rate: minimum one byte
            ######## Real rate minus Bytes of md5sum 
            if flag_md5sum :
                real_rate = (len(byte_img)-32*nbr_patches) / 1000
            else :
                real_rate = len(byte_img) / 1000
            rate_overhead = (real_rate / estimated_rate - 1) * 100
            absolute_overhead = real_rate - estimated_rate

            print('Arithmetic coding of      : ' + str(bitstream_path.split('/')[-1].rstrip(BITSTREAM_SUFFIX)) + ' ' + latent_name)          
            print('Bitrate estimation [kByte]: ' + '%.3f' % (estimated_rate))
            print('Real bitstream     [kByte]: ' + '%.3f' % (real_rate))
            print('Rate overhead          [%]: ' + '%.1f' % (rate_overhead))
            print('Absolute overhead  [Kbyte]: ' + '%.3f' % (absolute_overhead))
            #print('Header overhead     [byte]: ' + '%.1f' % (header_overhead))
            print('Nb. bytes in file   [byte]: ' + '%.1f' % (new_size - old_size))


        # Check that entropy coding is lossless
        if flag_debug:
            x_decoded = self.decode_one_img(config, {
                'mode': mode,
                'mu': mu,
                'sigma': sigma,
                'bitstream_path': bitstream_path,
                'data_dim': x.size(),
                'device': x.device,
                'flag_debug': flag_debug,
                'latent_name': latent_name,
                'flag_md5sum': flag_md5sum,
                'isFirstFr': isFirstFr, 
            })
            
            if torch.all(torch.eq(x.to('cpu'), x_decoded)):
                print('Ok! Entropy coding is lossless\n')
            else:
                print('-' * 80)
                print('Ko! Entropy coding is not lossless: ' + str((x_decoded - (x.to('cpu')).abs().sum())) + '\n')
                print('-' * 80)
        return byte_img
   
    def decode_one_patch(self, patch_latent_size, byte_patch, param):
        """
        Decode the features of y (mode laplace) or z (mode pmf) ), for one patch
        return the decoded features 
        """
        DEFAULT_PARAM = {
            # Mode is either <laplace> or <pmf>
            'mode': 'laplace',
            # If mode == 'laplace', we need the parameters mu and sigma
            'sigma': None,
            'mu': None,
            # Name (absolute path) of the bitstream
            'bitstream_path': None,
            # Dimension of the data to decode, as a tuple (B, C, H, W)
            'data_dim': None,
            # On which device the code will run
            'device': 'cpu',
            # Debug. If true, print specific stuff
            'flag_debug': True,
            # Latent Name y or z.
            'latent_name': '',
            # A feature-wise md5sum is included in the bitstream, verify
            # that the decoded version is identical to the provided md5sum.
            'flag_md5sum': False,
        }
        mode = get_value('mode', param, DEFAULT_PARAM)
        sigma = get_value('sigma', param, DEFAULT_PARAM)
        mu = get_value('mu', param, DEFAULT_PARAM)
        bitstream_path = get_value('bitstream_path', param, DEFAULT_PARAM)
        data_dim = get_value('data_dim', param, DEFAULT_PARAM)
        device = get_value('device', param, DEFAULT_PARAM)
        flag_debug = get_value('flag_debug', param, DEFAULT_PARAM)
        latent_name = get_value('latent_name', param, DEFAULT_PARAM)
        flag_md5sum = get_value('flag_md5sum', param, DEFAULT_PARAM)

        if not(bitstream_path.endswith(BITSTREAM_SUFFIX)):
            bitstream_path += BITSTREAM_SUFFIX
        
        # get the bytes of one patch features 
        byte_patch = byte_patch[0:patch_latent_size]
        if mode == 'laplace':

            # Compute the CDF for all symbols
            output_cdf = self.compute_cdf({
                'mode': mode,
                'sigma': sigma,
                'mu': mu,
                'data_dimension': data_dim,
            })
            # cdf must be on cpu
            output_cdf = output_cdf.cpu()

            # Decode byte stream
            symbol = torchac.decode_float_cdf(
                output_cdf, byte_patch, needs_normalization=True
            )

            # Shift back symbol
            x_decoded = (symbol - self.AC_MAX_VAL)
            #x_decoded = x_decoded.to(torch.float).to(device)

        else:
            # Compute the CDF for all symbols
            output_cdf = self.compute_cdf({
                'mode': mode,
                'sigma': None,
                'mu': None, 
                'data_dimension': data_dim,
            })

            # cdf must be on cpu
            output_cdf = output_cdf.cpu()

            # Decode byte stream
            symbol = torchac.decode_float_cdf(output_cdf, byte_patch)

            # Shift back symbol
            x_decoded = (symbol - self.AC_MAX_VAL).to(torch.float).to(self.device) 
        return x_decoded

    def decode_one_img(self, config, param): 
        """
        Read bitstream (y or z) of the patches forming one frame
        return the decoded features of the patches forming one frame
        """
        DEFAULT_PARAM = {
            # Mode is either <laplace> or <pmf>
            'mode': None,
            # If mode == 'laplace', we need the parameters mu and sigma
            'sigma': None,
            'mu': None,
            # Name (absolute path) of the bitstream
            'bitstream_path': None,
            # Dimension of the data to decode, as a tuple (B, C, H, W)
            'data_dim': None,
            # On which device the code will run
            'device': 'cpu',
            # Debug. If true, print specific stuff
            'flag_debug': True,
            # Latent Name y or z
            'latent_name': None,
            # A feature-wise md5sum is included in the bitstream, verify
            # that the decoded version is identical to the provided md5sum.
            'flag_md5sum': False,
            # To read the header from the bistream, the decoder needs to know if it's the first frame or not.
            'isFirstFr': False, 
        }
        mode = get_value('mode', param, DEFAULT_PARAM)
        sigma = get_value('sigma', param, DEFAULT_PARAM)
        mu = get_value('mu', param, DEFAULT_PARAM)
        bitstream_path = get_value('bitstream_path', param, DEFAULT_PARAM)
        data_dim = get_value('data_dim', param, DEFAULT_PARAM)
        device = get_value('device', param, DEFAULT_PARAM)
        flag_debug = get_value('flag_debug', param, DEFAULT_PARAM)
        latent_name = get_value('latent_name', param, DEFAULT_PARAM)
        flag_md5sum = get_value('flag_md5sum', param, DEFAULT_PARAM)
        isFirstFr = get_value('isFirstFr', param, DEFAULT_PARAM)
        
         # Read byte-stream file
        with open(bitstream_path, 'rb') as fin:
            byte_stream = fin.read()
        # Read header only if it's the first frame, and write it in a json file
        if mode == 'laplace' and isFirstFr:
            header, len_header, byte_stream_rest = self.read_header_img(byte_stream, config)
            byte_stream = byte_stream_rest
            with open("header.json", "w") as outfile:
                json.dump(header, outfile)
        if mode == 'pmf' and isFirstFr:
            header_z, byte_stream_rest = self.read_header_z(byte_stream)
            byte_stream = byte_stream_rest
            with open("header_z.json", "w") as outfile:
                json.dump(header_z, outfile)
                
        # Read bytes of the patches forming one frame
        idx_patch = 0 
        AC_Ok_fr = True
        while byte_stream != b"":
            # Read md5sum bytes if flag_md5sum is activated
            if flag_md5sum: 
                encoder_md5sum = byte_stream[0:32].decode().encode()
                byte_stream = byte_stream[32:]
            # get the byte size of each patch features
            patch_latent_size = int.from_bytes(byte_stream[0:4], byteorder='big')
            byte_stream = byte_stream[4:]
            # Decode the features of each patch 
            if mode == 'laplace':
                patch_dim = (1, data_dim[1],  data_dim[2],  data_dim[3] )
                x_decoded = self.decode_one_patch(patch_latent_size, byte_stream, {
                    'mode': mode,
                    'mu': mu[idx_patch,:,:,:].unsqueeze(0),
                    'sigma': sigma[idx_patch,:,:,:].unsqueeze(0),
                    'bitstream_path': bitstream_path,
                    'data_dim': patch_dim,
                    'device': self.device,
                    'flag_debug': flag_debug,
                    'latent_name': latent_name,
                    'flag_md5sum': flag_md5sum,
                })
            elif mode == 'pmf':
                # The patch dimension is deduced from the header directly if it's the first frame
                # Otherwise it's given as an input 
                if isFirstFr :
                    patch_dim = (1, 64, header_z['H_z'], header_z['W_z'])
                else :
                    patch_dim = (1, data_dim[1],  data_dim[2],  data_dim[3] )
    
    
                x_decoded = self.decode_one_patch(patch_latent_size, byte_stream, {
                    'mode': mode,
                    'mu': None,
                    'sigma': None,
                    'bitstream_path': bitstream_path,
                    'data_dim':patch_dim,
                    'device': self.device,
                    'flag_debug': flag_debug,
                    'latent_name': latent_name,
                    'flag_md5sum': flag_md5sum,
                })
            
            # # Check the decoder feature-wise md5sum for each patch
            if flag_md5sum:
                x_decoder = x_decoded.to(self.device).to(torch.int16).numpy().astype(int)
                np.savetxt( './tmp_tensor.npy', x_decoder.flatten())
                dec_md5 = compute_md5sum({'in_file': './tmp_tensor.npy'}).encode()
                os.system('rm ./tmp_tensor.npy')
                if dec_md5 != encoder_md5sum:
                    # Write logs at the patch level   
                    AC_Ok =  False
                    print('[Error] lossy arithmetic coding for')
                    print('\t' + bitstream_path + ' ' + latent_name + ' For Patch ' + str(idx_patch))
                    print('-' * 80)
                else:
                    AC_Ok =  True   
                AC_Ok_fr = AC_Ok_fr and AC_Ok
                
            # get the decoded features of all patches in one vector        
            if idx_patch == 0:
                rec_patches = x_decoded
            else : 
                rec_patches = torch.cat((rec_patches, x_decoded), 0 )
            # Remove the bytes alredy decoded
            byte_stream = byte_stream[patch_latent_size:]
            idx_patch += 1
            
        # Write logs at the frame level   
        if flag_md5sum:    
            if AC_Ok_fr == True :
                print('All good for ' + bitstream_path + ' ' + latent_name)
            else : 
                print('Problem ' + bitstream_path + ' ' + latent_name)
        return rec_patches


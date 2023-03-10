# Software Name: PaNVC
# Copyright (c) 2023 Ateme
# Licence :  BSD-3-Clause "New" 
# This software is distributed under the BSD-3-Clause license.

import math
import torch.nn as nn
import torch
from model.tools_architecture import CustomConvLayer, UpscalingLayer, SimplifiedAttention, AttentionResBlock

class ha (nn.Module):
    '''
    Hyper Encoder architecture based on  " Z. Cheng, H. Sun, M. Takeuchi, and J. Katto. 2020. Learned Image Compression With Discretized Gaussian Mixture Likelihoods and Attention Modules.
                                     In 2020 IEEE/CVF CVPR. 7936–7945." (K =1)
    '''
    def __init__(self, in_ftr=256 ): #, in_chanel=6):
        super(ha, self).__init__()
        self.in_ftr = in_ftr
        self.nb_ftr = 192
        self.nb_zlatent = 64
        self.layers = nn.Sequential(CustomConvLayer( k_size=3, in_ft=self.in_ftr, out_ft= self.nb_ftr, flag_bias=True, non_linearity='leaky_relu', conv_stride=1, padding_mode='replicate' ),
                                    CustomConvLayer( k_size=3, in_ft=self.nb_ftr, out_ft= self.nb_ftr, flag_bias=True, non_linearity='leaky_relu', conv_stride=1, padding_mode='replicate' ),
                                    CustomConvLayer( k_size=3, in_ft=self.nb_ftr, out_ft= self.nb_ftr, flag_bias=True, non_linearity='leaky_relu', conv_stride=2, padding_mode='replicate' ),
                                    CustomConvLayer( k_size=3, in_ft=self.nb_ftr, out_ft= self.nb_ftr, flag_bias=True, non_linearity='leaky_relu', conv_stride=1, padding_mode='replicate' ),
                                    CustomConvLayer( k_size=3, in_ft=self.nb_ftr, out_ft= self.nb_zlatent, flag_bias=False, non_linearity='no', conv_stride=2,  padding_mode='replicate')         
                                    )
    def forward(self, y):
        return(self.layers(y))


class hs (nn.Module):
    '''
    Hyper Decoder architecture based on " Z. Cheng, H. Sun, M. Takeuchi, and J. Katto. 2020. Learned Image Compression With Discretized Gaussian Mixture Likelihoods and Attention Modules.
                                     In 2020 IEEE/CVF CVPR. 7936–7945." (K =1)
    '''
    def __init__(self, nb_ylatent = 256): #, in_chanel=6):
        super(hs, self).__init__()
        self.nb_zlatent = 64
        self.nb_ylatent = nb_ylatent
        self.nb_ftr = 192
        
        self.layers = nn.Sequential(CustomConvLayer( k_size=3, in_ft=self.nb_zlatent, out_ft=self.nb_ftr, flag_bias=True, non_linearity='leaky_relu', conv_stride=1, padding_mode='replicate'), 
                                    UpscalingLayer(k_size=3, in_ft=self.nb_ftr, out_ft=self.nb_ftr, flag_bias=True, non_linearity='leaky_relu', mode='transposed', flag_first_layer=False ),
                                    CustomConvLayer( k_size=3, in_ft=self.nb_ftr, out_ft= int(1.5*self.nb_ftr), flag_bias=True, non_linearity='leaky_relu', conv_stride=1, padding_mode='replicate'),
                                    UpscalingLayer(k_size=3, in_ft=int(1.5*self.nb_ftr), out_ft=int(1.5*self.nb_ftr), flag_bias=True, non_linearity='leaky_relu', mode='transposed', flag_first_layer=False ), 
                                    CustomConvLayer( k_size=3, in_ft=int(1.5*self.nb_ftr), out_ft= int(2*self.nb_ftr), flag_bias=True, non_linearity='leaky_relu', conv_stride=1, padding_mode='replicate'),
                                    CustomConvLayer( k_size=1, in_ft=int(2*self.nb_ftr), out_ft= int(2*self.nb_ylatent), flag_bias=True, non_linearity='leaky_relu', conv_stride=1, padding_mode='replicate' ),
                                    CustomConvLayer( k_size=1, in_ft=int(2*self.nb_ylatent), out_ft= int(2*self.nb_ylatent), flag_bias=True, non_linearity='leaky_relu', conv_stride=1, padding_mode='replicate'),
                                    CustomConvLayer( k_size=1, in_ft=int(2*self.nb_ylatent), out_ft= int(2*self.nb_ylatent), flag_bias=True, non_linearity='leaky_relu', conv_stride=1, padding_mode='replicate' )
                )

        
    def forward(self, y):
        return(self.layers(y))

    

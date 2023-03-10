# Software Name: PaNVC
# Copyright (c) 2023 Ateme
# Licence :  BSD-3-Clause "New" 
# This software is distributed under the BSD-3-Clause license.

import math
import torch.nn as nn
import torch
import numpy as np
from torch.nn import Conv2d, ReplicationPad2d
from model.tools_architecture import CustomConvLayer, UpscalingLayer, ChengResBlock, SimplifiedAttention, AttentionResBlock



class ga(nn.Module):
    '''
    Main Encoder architecture based on  " Z. Cheng, H. Sun, M. Takeuchi, and J. Katto. 2020. Learned Image Compression With Discretized Gaussian Mixture Likelihoods and Attention Modules.
                                     In 2020 IEEE/CVF CVPR. 7936–7945."
    '''    
    def __init__(self, nb_ftr=192, nb_latent=256, in_chanel=3):
        super(ga, self).__init__()
        self.nb_ftr = nb_ftr
        self.nb_latent =  nb_latent
        #k_size = 5
        self.layers = nn.Sequential( CustomConvLayer( k_size=3, in_ft=in_chanel, out_ft=self.nb_ftr, flag_bias=True, non_linearity='gdn'),
                                     ChengResBlock( nb_ft = self.nb_ftr, mode='down'),
                                     ChengResBlock( nb_ft = self.nb_ftr, mode='plain'),
                                     ChengResBlock( nb_ft = self.nb_ftr, mode='down'),
                                     ChengResBlock( nb_ft = self.nb_ftr, mode='plain'),
                                     SimplifiedAttention(self.nb_ftr, k_size=1, lightweight_resblock=True),
                                     ChengResBlock( nb_ft = self.nb_ftr, mode='plain'),
                                     ChengResBlock( nb_ft = self.nb_ftr, mode='down'),
                                     ChengResBlock( nb_ft = self.nb_ftr, mode='plain'),
                                     ChengResBlock( nb_ft = self.nb_ftr, mode='down'),
                                     CustomConvLayer(k_size=3, in_ft=self.nb_ftr, out_ft=self.nb_latent, non_linearity='no',conv_stride=2,flag_bias=False),
                                     SimplifiedAttention(self.nb_latent, k_size=1, lightweight_resblock=True),
                                    ) 
        
    def forward(self, xt): 
        return(self.layers(xt))


class gs(nn.Module):
    '''
    Main Decoder architecture based on " Z. Cheng, H. Sun, M. Takeuchi, and J. Katto. 2020. Learned Image Compression With Discretized Gaussian Mixture Likelihoods and Attention Modules.
                                     In 2020 IEEE/CVF CVPR. 7936–7945."
    '''    
    def __init__(self, nb_ftr=192, nb_latent=256):
        super(gs, self).__init__()
        self.nb_ftr = nb_ftr
        self.nb_latent =  nb_latent
        self.layers = nn.Sequential( SimplifiedAttention(self.nb_latent, k_size=1, lightweight_resblock=True),
                                     CustomConvLayer(k_size=3, in_ft=self.nb_latent, out_ft=self.nb_ftr, non_linearity='gdn'),
                                     ChengResBlock( nb_ft = self.nb_ftr, mode='plain'),
                                     ChengResBlock( nb_ft = self.nb_ftr, mode='up_tconv'),
                                     ChengResBlock( nb_ft = self.nb_ftr, mode='plain'),
                                     ChengResBlock( nb_ft = self.nb_ftr, mode='up_tconv'),
                                     SimplifiedAttention(self.nb_ftr, k_size=1, lightweight_resblock=True),
                                     ChengResBlock( nb_ft = self.nb_ftr, mode='plain'),
                                     ChengResBlock( nb_ft = self.nb_ftr, mode='up_tconv'),
                                     ChengResBlock( nb_ft = self.nb_ftr, mode='plain'),
                                     ChengResBlock( nb_ft = self.nb_ftr, mode='up_tconv'),
                                     UpscalingLayer(k_size=3, in_ft=self.nb_ftr, out_ft=3, non_linearity='no', mode='transposed')
                                  ) 

    def forward(self, x):
        return(self.layers(x))



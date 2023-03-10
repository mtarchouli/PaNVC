# Software Name: PaNVC
# Copyright (c) 2023 Ateme
# Licence :  BSD-3-Clause "New" 
# This software is distributed under the BSD-3-Clause license.

from bitstream.bitstream_patch_overlapping import ArithmeticCoder
import torch 
import numpy
import random
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"

"""
Ensure reproductibility of the weights of the entropy for z and y 
at the encoder and the decoder side
"""

def set_seed(seed=234):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True) 

    
global AE_z
set_seed()
AE_z = ArithmeticCoder({
        'nb_channel':  64, 
        'mode' : 'pmf',
        'device': 'cpu', 
        'AC_MAX_VAL': 256,}) 

global AE_y
set_seed()
AE_y = ArithmeticCoder({
            'nb_channel': 256, 
            'mode' : "laplace",
            'device': 'cpu',
            'AC_MAX_VAL': 256,
    })

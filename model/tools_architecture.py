# Software Name: PaNVC
# Copyright (c) 2023 Ateme
# Licence :  BSD-3-Clause "New" 
# This software is distributed under the BSD-3-Clause license.

import torch
import torch.utils.data
from torch import nn
from torch.autograd import Function
from torch.nn import Module, Conv2d, ReplicationPad2d,\
                     LeakyReLU, Sequential, ReLU, ConvTranspose2d, Sigmoid

import numpy as np
import torch.nn.functional as F
from model.math_func import LOG_VAR_MIN, LOG_VAR_MAX

"""
Layers for model architecture
"""


"""
# This module implements the Generalized Divise Normalization (GDN) Transform,
  proposed by Ballé et al. in:
    http://www.cns.nyu.edu/pub/lcv/balle16a-reprint.pdf, 2016.

# This non-linear reparametrization of a linearly transformed vector y = f(x)
  (where f is a convolutional or fully connected layer) acts as a non linearity,
  replacing the ReLU.
  
# This module was introduced in  :  https://github.com/jorge-pessoa/pytorch-msssim
# The following module is from the modified version presented in : https://github.com/Orange-OpenSource/AIVC

# The licences on which our framework depends are to be found in the folder named "licence_dependency"
"""
class LowerBound(Function):

    def __init__(self):
        super(LowerBound, self).__init__()

    @staticmethod
    def forward(ctx, inputs, bound):
        # ! Memory transfer, use device=inputs.device
        b = torch.ones(inputs.size(), device=inputs.device)*bound
        # b = b.to(inputs.device)
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors

        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None

class GDN(nn.Module):
    """Generalized divisive normalization layer.
    y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]))
    """

    def __init__(self,
                 ch,
                 inverse=False,
                 beta_min=1e-6,
                 gamma_init=.1,
                 reparam_offset=2**-18):
        super(GDN, self).__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.reparam_offset = torch.FloatTensor([reparam_offset])

        # Will be updated after the first batch
        self.current_device = 'cpu'

        self.build(ch)

    def build(self, ch):
        self.pedestal = self.reparam_offset**2
        self.beta_bound = (self.beta_min + self.reparam_offset**2)**.5
        self.gamma_bound = self.reparam_offset

        # Create beta param
        beta = torch.sqrt(torch.ones(ch)+self.pedestal)
        self.beta = nn.Parameter(beta)

        # Create gamma param
        eye = torch.eye(ch)
        g = self.gamma_init*eye
        g = g + self.pedestal
        gamma = torch.sqrt(g)

        self.gamma = nn.Parameter(gamma)
        self.pedestal = self.pedestal

        # Push data to current device
        self.push_to_current_device()

    def push_to_current_device(self):
        self.beta = self.beta.to(self.current_device)
        self.beta_bound = self.beta_bound.to(self.current_device)
        self.gamma = self.gamma.to(self.current_device)
        self.gamma_bound = self.gamma_bound.to(self.current_device)
        self.pedestal = self.pedestal.to(self.current_device)

    def forward(self, inputs):
        # The second condition is only here when reloading from a checkpoint
        # when doing this, the device in current device is the correct one
        # eventhough beta, gamma and so on are not on te current device
        if (inputs.device != self.current_device) or (inputs.device != self.beta_bound.device):
            # Push to a new device only if it has changed
            self.current_device = inputs.device
            self.push_to_current_device()

        unfold = False
        if inputs.dim() == 5:
            unfold = True
            bs, ch, d, w, h = inputs.size()
            inputs = inputs.view(bs, ch, d*w, h)

        _, ch, _, _ = inputs.size()

        # Beta bound and reparam
        beta = LowerBound().apply(self.beta, self.beta_bound)
        # lower_bound_fn = LowerBound()
        # beta = lower_bound_fn(self.beta, self.beta_bound)
        beta = beta**2 - self.pedestal

        # Gamma bound and reparam
        gamma = LowerBound().apply(self.gamma, self.gamma_bound)
        gamma = gamma**2 - self.pedestal
        gamma = gamma.view(ch, ch, 1, 1)

        # Norm pool calc
        norm_ = nn.functional.conv2d(inputs**2, gamma, bias=beta)
        norm_ = torch.sqrt(norm_)

        # Apply norm
        if self.inverse:
            outputs = inputs * norm_
        else:
            outputs = inputs / norm_

        if unfold:
            outputs = outputs.view(bs, ch, d, w, h)

        return outputs

"""
 The following modules are from : https://github.com/Orange-OpenSource/AIVC
 They are licenced under BSD 3-Clause "New"
 
 The licences on which our framework depends are to be found in the folder named "licence_dependency"
"""

class AttentionResBlock(Module):
    """
    From [3]
    """

    def __init__(self, nb_ft):
        super(AttentionResBlock, self).__init__()

        half_ft = int(nb_ft / 2)

        self.layers = Sequential(
            Conv2d(nb_ft, half_ft, 1),
            LeakyReLU(),
            ReplicationPad2d(1),
            Conv2d(half_ft, half_ft, 3),
            LeakyReLU(),
            Conv2d(half_ft, nb_ft, 1),
        )

    def forward(self, x):
        return F.leaky_relu(x + self.layers(x))


class SimplifiedAttention(Module):
    """
    From [3]
    """

    def __init__(self, nb_ft, k_size=3, lightweight_resblock=False):
        super(SimplifiedAttention, self).__init__()

        self.nb_ft = nb_ft
        self.k_size = k_size

        # In [3], the res block does not always operate at nb_ft, we denote
        # this option as light_resb, opposite to standard resblock operating
        # at full nb_ft everything
        # Two networks in this module: the trunk and the attention

        if lightweight_resblock:
            self.trunk = Sequential(
                AttentionResBlock(self.nb_ft),
                AttentionResBlock(self.nb_ft),
                AttentionResBlock(self.nb_ft)
            )

            self.attention = Sequential(
                AttentionResBlock(self.nb_ft),
                AttentionResBlock(self.nb_ft),
                AttentionResBlock(self.nb_ft),
                Conv2d(self.nb_ft, self.nb_ft, 1),
                Sigmoid()
            )

        else:
            self.trunk = Sequential(
                ResBlock(self.k_size, self.nb_ft),
                ResBlock(self.k_size, self.nb_ft),
                ResBlock(self.k_size, self.nb_ft)
            )

            self.attention = Sequential(
                ResBlock(self.k_size, self.nb_ft),
                ResBlock(self.k_size, self.nb_ft),
                ResBlock(self.k_size, self.nb_ft),
                Conv2d(self.nb_ft, self.nb_ft, 1),
                Sigmoid()
            )

    def forward(self, x):
        trunk_out = self.trunk(x)
        attention_out = self.attention(x)

        weighted_trunk = trunk_out * attention_out
        res = weighted_trunk + x
        return res

class ChengResBlock(Module):

    def __init__(self, nb_ft, mode='plain'):
        """
        Reimplementation of the residual blocks defined in [1]

        [1] "Deep Residual Learning for Image Compression", Cheng et al, 2019
        
        * <nb_ft>:
        ?       Number of internal features for all conv. layers

        * <mode>:
        ?       <plain>: standard mode, no upscaling nor downscaling
        ?       <down> : downscaling by 2
        ?       <up_tconv>: upscaling by 2 with a transposed conv
        """

        super(ChengResBlock, self).__init__()

        self.mode = mode
        # In plain mode, non_linearity is Leaky ReLU
        if self.mode == 'plain':
            self.layers = Sequential(
                CustomConvLayer(
                    k_size=3,
                    in_ft=nb_ft,
                    out_ft=nb_ft,
                    non_linearity='leaky_relu'
                    ),
                CustomConvLayer(
                    k_size=3,
                    in_ft=nb_ft,
                    out_ft=nb_ft,
                    non_linearity='leaky_relu'
                    )
                )
        elif self.mode == 'Aneigh':
            self.layers = Sequential(
                CustomConvLayer(
                    k_size=3,
                    in_ft=nb_ft + 2,
                    out_ft=nb_ft,
                    non_linearity='leaky_relu',
                    conv_stride=2
                    ),
                CustomConvLayer(
                    k_size=3,
                    in_ft=nb_ft,
                    out_ft=nb_ft,
                    non_linearity='gdn'
                    )
                )
                
            self.aux_layer = Conv2d(nb_ft+2, nb_ft, 1, stride=2)
        elif self.mode == 'down':
            self.layers = Sequential(
                CustomConvLayer(
                    k_size=3,
                    in_ft=nb_ft,
                    out_ft=nb_ft,
                    non_linearity='leaky_relu',
                    conv_stride=2
                    ),
                CustomConvLayer(
                    k_size=3,
                    in_ft=nb_ft,
                    out_ft=nb_ft,
                    non_linearity='gdn'
                    )
                )

            self.aux_layer = Conv2d(nb_ft, nb_ft, 1, stride=2)

        elif self.mode == 'up_tconv':
            self.layers = Sequential(
                UpscalingLayer(
                    k_size=3,
                    in_ft=nb_ft,
                    out_ft=nb_ft,
                    non_linearity='leaky_relu',
                    mode='transposed'
                    ),
                CustomConvLayer(
                    k_size=3,
                    in_ft=nb_ft,
                    out_ft=nb_ft,
                    non_linearity='gdn_inverse'
                    )
                )

            # ! WARNING: Modification here for a kernel size of 3 instead of 1
            # ! because a transposed conv with k_size = 1 doesn't really make
            # ! sense.
            self.aux_layer = UpscalingLayer(
                    k_size=3,
                    in_ft=nb_ft,
                    out_ft=nb_ft,
                    non_linearity='no',
                    mode='transposed'
                )

    def forward(self, x):
        if self.mode == 'plain':
            return x + self.layers(x)
        elif self.mode == 'down' or self.mode == 'up_tconv' or self.mode == 'Aneigh' :
            return self.aux_layer(x) + self.layers(x)


class ResBlock(Module):

    def __init__(self, k_size, nb_ft):
        super(ResBlock, self).__init__()
        nb_pix_to_pad = int(np.floor(k_size / 2))
        self.layers = Sequential(
            ReplicationPad2d(nb_pix_to_pad),
            Conv2d(nb_ft, nb_ft, k_size),
            ReLU(),
            ReplicationPad2d(nb_pix_to_pad),
            Conv2d(nb_ft, nb_ft, k_size),
        )

    def forward(self, x):
        return F.relu(x + self.layers(x))


class CustomConvLayer(nn.Module):
    """
    Easier way to use convolution. Perform automatically replication pad
    to preserve spatial dimension

    non_linearity is either:
        <gdn>
        <gdn_inverse>
        <leaky_relu>
        <relu>
        <no>
    """
    def __init__(self, k_size=5, in_ft=64, out_ft=64, flag_bias=True,
                 non_linearity='leaky_relu', conv_stride=1,
                 padding_mode='replicate'):
        super(CustomConvLayer, self).__init__()
        nb_pix_to_pad = int(np.floor(k_size / 2))

        if padding_mode == 'replicate':
            padding_fn = ReplicationPad2d(nb_pix_to_pad)

        self.layers = Sequential(
            padding_fn,
            Conv2d(in_ft, out_ft, k_size, stride=conv_stride, bias=flag_bias)
        )

        if non_linearity == 'gdn':
            self.layers.add_module(
                'non_linearity',
                GDN(out_ft, inverse=False)
            )

        elif non_linearity == 'gdn_inverse':
            self.layers.add_module(
                'non_linearity',
                GDN(out_ft, inverse=True)
            )

        elif non_linearity == 'leaky_relu':
            self.layers.add_module(
                'non_linearity',
                LeakyReLU()
            )

        elif non_linearity == 'relu':
            self.layers.add_module(
                'non_linearity',
                ReLU()
            )

    def forward(self, x):
        return self.layers(x)


class UpscalingLayer(nn.Module):
    def __init__(self, k_size=5, in_ft=64, out_ft=64, flag_bias=True,
                 non_linearity='leaky_relu', mode='transposed',
                 flag_first_layer=False):
        """
        Upscaling with a factor of two

        * <non_linearity>:
        ?   gdn, gdn_inverse, leaky_relu, relu, no

        * <mode>:
        ?   transposed:
        ?       Use a transposed conv to perform upsampling
        ?   transposedtransposed_no_bias:
        ?       Use a transposed conv to perform upsampling, without a bias

        """

        super(UpscalingLayer, self).__init__()

        # Transposed conv param, computed thanks to
        # https://pytorch.org/docs/stable/nn.html#convtranspose2d
        # t_dilat = 1 # defaut value
        t_stride = 2
        t_outpad = 1
        t_padding = int(((t_outpad + k_size) / 2) - 1)

        # Override flag_bias
        if mode == 'transposed_no_bias':
            flag_bias = False

        self.layers = Sequential(
            ConvTranspose2d(
                    in_ft,
                    out_ft,
                    k_size,
                    stride=t_stride,
                    padding=t_padding,
                    output_padding=t_outpad,
                    bias=flag_bias
                )
        )

        # Add the correct non-linearity
        if non_linearity == 'gdn':
            self.layers.add_module(
                'non_linearity',
                GDN(out_ft, inverse=False)
            )

        elif non_linearity == 'gdn_inverse':
            self.layers.add_module(
                'non_linearity',
                GDN(out_ft, inverse=True)
            )

        elif non_linearity == 'leaky_relu':
            self.layers.add_module(
                'non_linearity',
                LeakyReLU()
            )

        elif non_linearity == 'relu':
            self.layers.add_module(
                'non_linearity',
                ReLU()
            )
            
    def forward(self, x):
        return self.layers(x)




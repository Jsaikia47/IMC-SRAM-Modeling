"""
Apply PACT to the weights in Conv2d layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .quantizer import *

__all__ = ['PACT_conv2d']


class PACT_W(nn.Module):
    def init_param(self, beta, nbits):
        self.nbits = nbits
        self.beta = nn.Parameter(torch.Tensor([beta]))
    def quant_weight(self):
        w_l = self.weight

        w_mean = self.weight.mean()                     # mean of the weight

        w_l = w_l - w_mean                              # center the weights

        # w_l = self.beta * torch.tanh(w_l)       
        w_l = w_l.clamp(-self.beta.item(), self.beta.item())       
        w_l = self.beta * w_l / 2 / torch.max(torch.abs(w_l)) + self.beta / 2
        
        with torch.no_grad():
            scale, zero_point = quantizer(self.nbits, 0, self.beta)
        
        w_l = 2 * STEQuantizer.apply(w_l, scale, zero_point, True, False) - self.beta + w_mean

        return w_l


class PACT_conv2d(nn.Conv2d, PACT_W):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=True, nbits=4, beta=1.0):
        super().__init__(in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, dilation=dilation, groups=groups,
                bias=bias)
        self.init_param(beta, nbits)

    def forward(self, input):
        return F.conv2d(input, self.quant_weight(), self.bias, self.stride, self.padding, self.dilation, self.groups)


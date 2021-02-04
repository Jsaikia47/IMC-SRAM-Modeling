"""
Apply PACT to the weights in Conv2d layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .quantizer import *

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

        scale, zero_point = symmetric_linear_quantization_params(self.nbits, self.beta, restrict_qrange=True)
        w_l = STEQuantizer_weight.apply(w_l, scale, zero_point, True, False, self.nbits, True)

        return w_l


class PACT_conv2d(nn.Conv2d, PACT_W):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=False, nbits=4, beta=0.1):
        super().__init__(in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, dilation=dilation, groups=groups,
                bias=bias)
        self.init_param(beta=self.weight.max(), nbits=nbits)

    def forward(self, input):
        return F.conv2d(input, self.quant_weight(), self.bias, self.stride, self.padding, self.dilation, self.groups)

class LearnableParam(nn.Module):
    def init_param(self, qcoef, nbits, restrictRange):
        self.nbits = nbits
        self.qcoef = nn.Parameter(torch.Tensor([qcoef]))
        self.restrictRange = restrictRange
    def quant_weight(self):
        w_l = self.weight

        std = w_l.std()
        mean = w_l.mean()
        self.alpha_w = self.qcoef[0,0]*std + self.qcoef[0,1]*mean
        
        # w_l = w_l.clamp(-self.alpha_w.item(), self.alpha_w.item())
        w_l = self.alpha_w * torch.tanh(10*w_l)
        with torch.no_grad():
            scale, zero_point = symmetric_linear_quantization_params(self.nbits, self.alpha_w, restrict_qrange=self.restrictRange)
        w_l = STEQuantizer_weight.apply(w_l, scale, zero_point, True, False, self.nbits, self.restrictRange)

        return w_l

class LearnbaleQuantConv2d(nn.Conv2d, LearnableParam):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=True, nbits=4, qcoef=[12.987, -13.156]):
        super().__init__(in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, dilation=dilation, groups=groups,
                bias=bias)
        self.init_param(qcoef=qcoef, nbits=nbits, restrictRange=True)

    def forward(self, input):
        return F.conv2d(input, self.quant_weight(), self.bias, self.stride, self.padding, self.dilation, self.groups)
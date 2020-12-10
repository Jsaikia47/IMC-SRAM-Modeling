"""
Multi-bit quantization modules
"""

import torch
import numpy as np
from torch import nn
from torch import autograd
from .quantizers import *


def Quant_prob(x, cdf_table=None, H=60, levels=11, mode='software'):
    if cdf_table is None:
        y = nn.functional.hardtanh(x, -H, H)
    else:
        y = x.clone()
    scale = (levels - 1) / (2*H)
    if cdf_table is None: # ideal quantization
        # print('Ideal Quantization!')
        y.data.mul_(scale).round_()
    else:
        with torch.no_grad():
            sram_depth = (cdf_table.weight.shape[0]-1) / 2
            x_ind = x.add(sram_depth).type(torch.int64)
            x_cdf = cdf_table(x_ind)
            x_comp = torch.rand_like(x).unsqueeze(-1).expand_as(x_cdf).sub(x_cdf).sign()
            y.data = x_comp.sum(dim=-1).mul(0.5)
    if mode == 'software':
        #scale_inv = 1. / scale
        #y.data.mul_(scale_inv)
        y.data.div_(scale)
    return y

class QuantizedConv2d(nn.Conv2d):
    def __init__(self, *kargs, **kwargs):
        # precision
        self.a_bits = kwargs.pop('w_bits')
        self.w_bits = kwargs.pop('a_bits')

        # hardware specs
        self.sram_depth = 0. # will be over-written in main.py
        self.quant_bound = 60
        self.mode = 'software'
        self.noise_std = 0.
        self.noise_9_mean = 0.
        self.noise_9_std = 0.
        self.cdf_table = None
        self.cdf_table_9 = None
        self.quant_prob_levels = 32
        super(QuantizedConv2d, self).__init__(*kargs, **kwargs)

        # FLAG
        self.input_quant = True
        self.weight_quant = True
        self.bitBybit = True

        # floating point boundaries
        self.register_parameter('alpha', nn.Parameter(torch.Tensor([10.0]).cuda()))
        self.register_buffer('alpha_w', torch.tensor(1.).cuda())

        # quantizer
        self.activation_quantizer = PACT_Quant(n_bits=2, alpha=self.alpha, inplace=False)
        self.weight_quantizer = SAWB_Quant(n_bits=2, alpha_w=self.alpha_w)
        
        # decimal to multi-bit
        self.activation_converter = PACT_MultiBit(n_bits=2, base_a=None)
        self.weight_converter = SAWB_MultiBit(n_bits=2, base_w=None)

    def forward(self, input):
        
        alpha_w = get_scale_2bit(self.weight)   # update quantization boundary
        self.weight_quantizer.update_alpha(alpha_w)
        
        input_ = input.clone()

        # base
        base_a = 2.**(self.a_bits-1) / (2**self.a_bits-1.)*self.alpha
        base_w = 2.**(self.w_bits-1) / (2**self.w_bits-1.)*alpha_w
        self.base = base_a * base_w
        
        # print('=================================')
        # print(f'base_a = {base_a} | alpha act = {self.alpha.data} | precision = {self.a_bits}')
        # print(f'base_w = {base_w} | alpha weight = {self.alpha_w} | precision = {self.w_bits}')
        # print('---------------------------------')

        # update base
        # self.activation_converter.base_a = base_a
        # self.weight_converter.base_w = base_w

        if not self.weight_quant and self.input_quant:
            outputs = nn.functional.conv2d(
                input=input,
                weight=self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups
            )
            return outputs

        # Quantization
        if self.weight_quant:
            weight_q = self.weight_quantizer(self.weight)
        else:
            weight_q = self.weight

        if self.input_quant:
            if self.weight.size(2) > 1:
                input_q = self.activation_quantizer(input)
            else:
                self.alpha.data = input.max().view(1)   # reassure the quantization levels for the residual connections
                input_q = self.activation_quantizer(input)
        else:
            # print(f'First layer')
            input_q = input
            outputs = nn.functional.conv2d(
                input=input_q,
                weight=weight_q,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups
            )
            return outputs

        # Convolution computation
        if self.bitBybit is False:
            #print(f'input activations = {torch.unique(input_q)}')
            #print(f'weight = {torch.unique(weight_q)}')
            outputs = nn.functional.conv2d(
                input=input_q,
                weight=weight_q,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups
            )
            return outputs
        else:
            # bit by bit computation
            # print(f'sram depth={self.sram_depth} | inchannels={self.in_channels}')
            if self.sram_depth > 0 and self.in_channels > 3:
                if self.noise_std == 0:
                    # print(f'input activations = {torch.unique(input)}')
                    # print(f'quantized activations = {torch.unique(input_q)}')
                    # print(f'weight = {torch.unique(weight_q)} | size={weight_q.size()}')
                    # print('=================================')
                    input_padded = torch.nn.functional.pad(input_q, [self.padding[0]]*4, value=0.)
                    input_list = torch.split(input_padded, self.sram_depth, dim=1)
                    self.weight_list = torch.split(weight_q, self.sram_depth, dim=1)
                    out = 0
                    map_x, map_y = input_q.shape[2], input_q.shape[3]
                    for input_p, weight_p in zip(input_list, self.weight_list):
                        inputs_p_bits = dec2binary_act(input_p, n_bits=self.a_bits, alpha=self.alpha)
                        weight_p_bits = dec2binary_weight(weight_p, n_bits=self.w_bits, alpha=alpha_w)
                        for ii in range(self.a_bits):
                            for iw in range(self.w_bits):
                                out_temp = 0
                                for k in range(self.weight.shape[2]):
                                    for j in range(self.weight.shape[3]):
                                        input_kj = inputs_p_bits[ii][:,:,k:k+map_x, j:j+map_y]
                                        weight_kj = weight_p_bits[iw][:,:,k:k+1,j:j+1]
                                        partial_sum = nn.functional.conv2d(input_kj, weight_kj, None, self.stride, (0,0), self.dilation, self.groups)
                                        partial_sum_quantized = Quant_prob(partial_sum, cdf_table=self.cdf_table, H=self.quant_bound, levels=self.quant_prob_levels, mode=self.mode)
                                        out_temp += partial_sum_quantized
                                if self.mode == 'software' and (self.cdf_table_9 is not None or self.noise_9_std > 0):
                                    out_temp /= (self.quant_bound / 5.)
                                    if self.noise_9_std > 0:
                                        out_temp_noisy = out_temp.clone()
                                        out_temp_noisy.data += torch.normal(mean=self.noise_9_mean, std=self.noise_9_std, size=out_temp.shape, device=out_temp.device).round()
                                        
                                        out += out_temp_noisy * (2**(-ii-iw)) * self.base
                                    else:
                                        out += Quant_prob(out_temp, cdf_table=self.cdf_table_9, H=45., levels=91, mode=self.mode) * (2**(-ii-iw)) * self.base
                                else:
                                    out += out_temp * (2**(-ii-iw)) * self.base
                                    # print(f'self.base = {self.base}')
                    if self.mode == 'software' and (self.cdf_table_9 is not None or self.noise_9_std > 0):
                        out *= (self.quant_bound / 5.)
                    # import pdb;pdb.set_trace()    
                else: # fast noise injection mode
                    out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                       self.padding, self.dilation, self.groups)
                    out += torch.normal(mean=0, std=np.sqrt(self.in_channels / self.sram_depth) * self.noise_std, size=out.shape, device=out.device)
            else:
                out = nn.functional.conv2d(input_q, weight_q, None, self.stride,
                                        self.padding, self.dilation, self.groups)
                
            if not self.bias is None:
                out += self.bias.view(1, -1, 1, 1).expand_as(out)    
            return out


class QuantizedLinear(nn.Linear):
    def __init__(self, in_channels, out_channels, bias=True, a_bits=2, w_bits=2, alpha_init=10.0):
        super(QuantizedLinear, self).__init__(in_features=in_channels, out_features=out_channels, bias=bias)

        # FLAG
        self.input_quant = True
        self.weight_quant = True
        
        # precision
        self.a_bits = a_bits
        self.w_bits = w_bits

        # floating point boundaries
        self.register_parameter('alpha', nn.Parameter(torch.Tensor([alpha_init])))
        self.register_buffer('alpha_w', torch.tensor(1.).cuda())
        
        # quantizer
        self.activation_quantizer = PACT_Quant(n_bits=2, alpha=self.alpha)
        self.weight_quantizer = SAWB_Quant(n_bits=2, alpha_w=self.alpha_w)

        # pooling
        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.quant_bound = 64

    def forward(self, input):
        # print(f'Layer: {self.layer_idx}')
        alpha_w = get_scale_2bit(self.weight)   # update quantization boundary
        self.weight_quantizer.update_alpha(alpha_w)

        if self.weight_quant:
            weight_q = self.weight_quantizer(self.weight)
        else:
            weight_q = self.weight

        if self.input_quant:
            input_q = self.activation_quantizer(input)
        else:
            input_q = input
        
        # print(f'FC input: {torch.unique(input)} | alpha = {self.alpha.data}')
        # print(f'FC Quantized input: {torch.unique(input_q)} | alpha = {self.alpha.data}')
        # print(f'FC Quantized weight: {torch.unique(weight_q)} | alpha_w={self.alpha_w} | size={list(weight_q.size())}')
        # print('=============================')

        input_q = self.avgpool(input_q)
        input_q = input_q.reshape(input_q.size(0), -1)

        # print(f'Before linear: {torch.unique(input_q)}')
        outputs = nn.functional.linear(input_q, weight_q, self.bias)

        return outputs

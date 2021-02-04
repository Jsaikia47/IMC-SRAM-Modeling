from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np

def Quantize(tensor, H=1., n_bits=2):
    if n_bits == 1:
        #return tensor.sign() * H
        tensor=torch.where(tensor > 0, torch.tensor(H, device=tensor.device), torch.tensor(-H, device=tensor.device)) # quantize 0 to -1
        return tensor
    else:
        k = (2**n_bits - 1) / (2*H)
        tensor=torch.round((torch.clamp(tensor, -H, H)+H) * k) / k - H
    return tensor

def Quant_prob(x, cdf_table=None, H=60, levels=11, mode='software'):
    if cdf_table is None:
        y = nn.functional.hardtanh(x, -H, H)
    else:
        y = x.clone()
    scale = (levels - 1) / (2*H)
    if cdf_table is None: # ideal quantization
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

def dec2binary(x, n_bits):
    '''
    assuming x is quantized to [-1, +1]
    '''
    if n_bits == 1:
        return [x]
    base = 2.**(n_bits-1) / (2**n_bits-1.)
    y = []
    x_remain = x.clone()
    for i in range(n_bits): # from MSB to LSB
        xb = x * base
        xb.data = torch.sign(x_remain)
        y.append(xb)
        x_remain -= xb * base
        base /= 2.
    return y

class QuantizeActLayer(nn.Module):
    def __init__(self, n_bits=2, H=1., inplace=True):
        super(QuantizeActLayer, self).__init__()
        self.inplace = inplace
        self.n_bits = n_bits
        self.mode = 'software'

    def forward(self, x):
        y = nn.functional.hardtanh(x)
        y.data = Quantize(y.data, n_bits=self.n_bits, mode=self.mode)
        return y

    def extra_repr(self):
        return super(QuantizeActLayer, self).extra_repr() + 'n_bits={}'.format(self.n_bits)

class QuantizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        self.w_bits = kwargs.pop('w_bits')
        self.a_bits = kwargs.pop('a_bits')
        self.sram_depth = 0 # will be over-written in main.py
        self.quant_bound = 64
        self.cdf_table = None
        base_a = 2.**(self.a_bits-1) / (2**self.a_bits-1.)
        base_w = 2.**(self.w_bits-1) / (2**self.w_bits-1.)
        self.base = base_a * base_w
        self.mode = 'software'
        self.noise_std = 0.
        super(QuantizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        # Quantize input to a_bits
        input = nn.functional.hardtanh(input)
        input.data = Quantize(input.data, n_bits=self.a_bits)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        # Quantize weights to w_bits
        self.weight.data=Quantize(self.weight.org, n_bits=self.w_bits)
        if self.sram_depth > 0:
            if self.noise_std == 0:
                # split in input-channel dimension
                self.weight_list = torch.split(self.weight, self.sram_depth, dim=1)
                input_list = torch.split(input, self.sram_depth, dim=1)
                out = 0
                for input_p, weight_p in zip(input_list, self.weight_list):
                    # split input and weight in bits
                    inputs_p_bits = dec2binary(input_p, self.a_bits)
                    weight_p_bits = dec2binary(weight_p, self.w_bits)
                    for ii in range(self.a_bits):
                        for iw in range(self.w_bits):
                            partial_sum = nn.functional.linear(inputs_p_bits[ii], weight_p_bits[iw])
                            partial_sum_quantized = Quant_prob(partial_sum, cdf_table=self.cdf_table, H=self.quant_bound, mode=self.mode)
                            out += partial_sum_quantized * (2**(-ii-iw)) * self.base
            else:
                out = nn.functional.linear(input, self.weight)
                out += torch.normal(mean=0, std=np.sqrt(self.in_features / self.sram_depth) * self.noise_std / 3., size=out.shape, device=out.device)
        else:
            out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)
        return out

    def extra_repr(self):
        return super(QuantizeLinear, self).extra_repr() + ', a_bits={}'.format(self.a_bits) + ', w_bits={}'.format(self.w_bits)

class QuantizeConv2d(nn.Conv2d):
    def __init__(self, *kargs, **kwargs):
        self.w_bits = kwargs.pop('w_bits')
        self.a_bits = kwargs.pop('a_bits')
        self.sram_depth = 0 # will be over-written in main.py
        self.quant_bound = 64
        base_a = 2.**(self.a_bits-1) / (2**self.a_bits-1.)
        base_w = 2.**(self.w_bits-1) / (2**self.w_bits-1.)
        self.base = base_a * base_w
        self.mode = 'software'
        self.noise_std = 0.
        self.noise_9_mean = 0.
        self.noise_9_std = 0.
        self.cdf_table = None
        self.cdf_table_9 = None
        super(QuantizeConv2d, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        # Quantize input to a_bits
        if self.in_channels > 3:
            input = nn.functional.hardtanh(input)
            input.data = Quantize(input.data, n_bits=self.a_bits)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        # Quantize weights to w_bits
        self.weight.data=Quantize(self.weight.org, n_bits=self.w_bits)
        if self.sram_depth > 0 and self.in_channels > 3:
            # print(f'input activations = {torch.unique(input)}')
            # print(f'quantized activations = {torch.unique(input)}')
            # print(f'weight = {torch.unique(self.weight)}')
            # print('=================================')
            if self.noise_std == 0.:
                input_padded = torch.nn.functional.pad(input, [self.padding[0]]*4, value=1.)
                input_list = torch.split(input_padded, self.sram_depth, dim=1)
                self.weight_list = torch.split(self.weight, self.sram_depth, dim=1)
                out = 0
                map_x, map_y = input.shape[2], input.shape[3]
                for input_p, weight_p in zip(input_list, self.weight_list):
                    inputs_p_bits = dec2binary(input_p, self.a_bits)
                    weight_p_bits = dec2binary(weight_p, self.w_bits)
                    for ii in range(self.a_bits):
                        for iw in range(self.w_bits):
                            out_temp = 0
                            for k in range(self.weight.shape[2]):
                                for j in range(self.weight.shape[3]):
                                    input_kj = inputs_p_bits[ii][:,:,k:k+map_x, j:j+map_y]
                                    weight_kj = weight_p_bits[iw][:,:,k:k+1,j:j+1] 
                                    partial_sum = nn.functional.conv2d(input_kj, weight_kj, None, self.stride, (0,0), self.dilation, self.groups)
                                    partial_sum_quantized = Quant_prob(partial_sum, cdf_table=self.cdf_table, H=self.quant_bound, mode=self.mode)
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
                if self.mode == 'software' and (self.cdf_table_9 is not None or self.noise_9_std > 0):
                    out *= (self.quant_bound / 5.)
            else: # fast noise-injection mode
                out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                       self.padding, self.dilation, self.groups)
                out += torch.normal(mean=0, std=np.sqrt(self.in_channels / self.sram_depth) * self.noise_std, size=out.shape, device=out.device)
        else:
            out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                       self.padding, self.dilation, self.groups)
        if not self.bias is None:
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out

    def extra_repr(self):
        return super(QuantizeConv2d, self).extra_repr() + ', a_bits={}'.format(self.a_bits) + ', w_bits={}'.format(self.w_bits)

class BatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_status=True):
        self.mode = 'software' # 'software': normal batchnorm in software; 'hardware': simplified batchnorm in hardware
        self.weight_effective = torch.ones((num_features,)).cuda()
        self.bias_effective = torch.zeros((num_features,)).cuda()
        super(BatchNorm2d, self).__init__(num_features, eps, momentum, affine, track_running_status)

    def forward(self, input):
        if self.mode == 'software':
            return super(BatchNorm2d, self).forward(input)
        else:
            return input * self.weight_effective.view(1, -1, 1, 1).expand_as(input) + self.bias_effective.view(1, -1, 1, 1).expand_as(input)

class BatchNorm1d(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_status=True):
        self.mode = 'software' # 'software': normal batchnorm in software; 'hardware': simplified batchnorm in hardware
        self.weight_effective = torch.ones((num_features,)).cuda()
        self.bias_effective = torch.zeros((num_features,)).cuda()
        super(BatchNorm1d, self).__init__(num_features, eps, momentum, affine, track_running_status)

    def forward(self, input):
        if self.mode == 'software':
            return super(BatchNorm1d, self).forward(input)
        else:
            return input * self.weight_effective.view(1, -1).expand_as(input) + self.bias_effective.view(1, -1).expand_as(input)



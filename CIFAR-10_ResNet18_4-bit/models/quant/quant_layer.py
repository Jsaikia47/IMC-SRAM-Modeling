"""
quant_layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from .quantizer import *


def odd_symm_quant(input, nbit, mode='mean', k=2, dequantize=True, posQ=False):
    
    if mode == 'mean':
        alpha_w = k * input.abs().mean()
    elif mode == 'sawb':
        z_typical = {'4bit': [0.077, 1.013], '8bit':[0.027, 1.114]}               
        alpha_w = get_scale(input, z_typical[f'{int(nbit)}bit']).item()
    
    output = input.clamp(-alpha_w, alpha_w)

    if posQ:
        output = output + alpha_w

    scale, zero_point = symmetric_linear_quantization_params(nbit, abs(alpha_w), restrict_qrange=True)

    output = linear_quantize(output, scale, zero_point)
    
    if dequantize:
        output = linear_dequantize(output, scale, zero_point)

    return output, alpha_w, scale

def activation_quant(input, nbit, sat_val, dequantize=True):
    with torch.no_grad():
        scale, zero_point = quantizer(nbit, 0, sat_val)
    
    output = linear_quantize(input, scale, zero_point)

    if dequantize:
        output = linear_dequantize(output, scale, zero_point)

    return output, scale

class ClippedReLU(nn.Module):
    def __init__(self, num_bits, alpha=8.0, inplace=False, dequantize=True):
        super(ClippedReLU, self).__init__()
        self.alpha = nn.Parameter(torch.Tensor([alpha]))     
        self.num_bits = num_bits
        self.inplace = inplace
        self.dequantize = dequantize
        
    def forward(self, input):
        # print(f'ClippedRELU: input mean: {input.mean()} | input std: {input.std()}')
        input = F.relu(input)
        input = torch.where(input < self.alpha, input, self.alpha)
        
        with torch.no_grad():
            scale, zero_point = quantizer(self.num_bits, 0, self.alpha)
        input = STEQuantizer.apply(input, scale, zero_point, self.dequantize, self.inplace)
        return input

class ClippedHardTanh(nn.Module):
    def __init__(self, num_bits, inplace=False, dequantize=True):
        super(ClippedHardTanh, self).__init__()
        self.register_buffer('alpha_w', torch.Tensor([1.]))
        self.num_bits = num_bits
        self.inplace = inplace
        self.dequantize = dequantize
        
    def forward(self, input):
        input = F.hardtanh(input)
        with torch.no_grad():
            scale, zero_point = symmetric_linear_quantization_params(self.num_bits, self.alpha_w)
        input = STEQuantizer_tanh.apply(input, scale, zero_point, self.dequantize, self.inplace, self.num_bits, False)
        # print(f'quantized activation = {len(torch.unique(input))}')
        return input

    def extra_repr(self):
        return super(ClippedHardTanh, self).extra_repr() + 'nbit={}'.format(self.num_bits)

class int_quant_func(torch.autograd.Function):
    def __init__(self, nbit, alpha_w, restrictRange=True, ch_group=16, push=False):
        super(int_quant_func, self).__init__()
        self.nbit = nbit
        self.restrictRange = restrictRange
        self.alpha_w = alpha_w
        self.ch_group = ch_group
        self.push = push

    def forward(self, input):
        self.save_for_backward(input)
        output = input.clamp(-self.alpha_w.item(), self.alpha_w.item())
        scale, zero_point = symmetric_linear_quantization_params(self.nbit, self.alpha_w, restrict_qrange=self.restrictRange)
        output = STEQuantizer_weight.apply(output, scale, zero_point, True, False, self.nbit, self.restrictRange)   
        return output

    def backward(self, grad_output):
        grad_input = grad_output.clone()
        input, = self.saved_tensors
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input


def quantize(tensor, H=1., n =1):
    if n ==1:
        return tensor.sign()*H
    tensor = torch.round((torch.clamp(tensor,-H,H)+H) * (2**n-1)/(2*H)) *2*H/ (2**n -1) -H
    return tensor

def BinarizeInput(x):
    y =[]
    base = 8./15
    comp = x.clone()
    #x_remain = x
    for i in range(4):
        temp = torch.ge(comp,0).float().cuda()*2 -1
        comp -= temp*base
        y.append(temp)
        #x_remain = x_remain-x
        base = base/2
        #import pdb; pdb.set_trace()
    return y

def BinarizeWeight(x,alpha_w):
    y =[]
    base = (alpha_w*8)/15
    comp = x.clone()
    #x_remain = x
    for i in range(4):
        temp = torch.ge(comp,0).float().cuda()*2 -1
        comp -= temp*base
        y.append(temp)
        #x_remain = x_remain-x
        base = base/2
        #import pdb; pdb.set_trace()
    return y

def quant_XNORSRAM_2d(x,levels,edges,num_edges,bitlinenoise,offsetnoise):
     bitline = torch.randn(x.shape,device='cuda:0')*bitlinenoise
     x = x + bitline

     sum = 0
     edges = edges.unsqueeze(0).repeat(x.shape[1],1)
     
     #offset = torch.randn(x.shape[1]).cuda()
     #offset = offset*offsetnoise
     #offsets = offset.unsqueeze(1).repeat(1,num_edges)   
     #edges = edges + offsets.cuda() 
     
     for i in range(num_edges):
         y= edges[:,i].unsqueeze(0).repeat(x.shape[0],1)
         sum += torch.gt(x,y)
     output = levels[sum]
     return output
     # return x

def quant_XNORSRAM_4d(x,levels,edges,num_edges,bitlinenoise,offsetnoise):
     bitline = torch.randn(x.shape,device='cuda:0')*bitlinenoise
     x = x + bitline

     sum = 0
     edges = edges.unsqueeze(0).repeat(x.shape[1],1).cuda()
     
     #offset = torch.randn(x.shape[1]).cuda()
     #offset = offset*offsetnoise
     #offsets = offset.unsqueeze(1).repeat(1,num_edges)
     #edges = edges + offsets.cuda() 
     
     #import pdb; pdb.set_trace();
     for i in range(num_edges):
         y= edges[:,i].unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(x.shape[0],1,x.shape[2],x.shape[3])
         sum += torch.gt(x,y)
     #import pdb;pdb.set_trace();
     output = levels[sum]
     return output
     #return x

class clamp_conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                padding=0, dilation=1, groups=1, bias=False, nbit=4, mode='mean', k=2):
        super(clamp_conv2d, self).__init__(in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, dilation=dilation, groups=groups,
                bias=bias)
        self.nbits = 4
        self.mode = mode
        self.k = k
        self.alpha_w = 1.
        self.n_frac= 3
        #self.base = 2. **(self.n_bits-1)/(2. **(self.n_frac +1 ) - 1. )
        self.beta=1
        self.activation_list = []


    def forward(self, input):
        weight_c = self.weight.clone()

        if self.mode == 'mean':
            self.alpha_w = self.k * weight_c.abs().mean()
        elif self.mode == 'sawb':
            z_typical = {'4bit': [0.077, 1.013], '8bit':[0.027, 1.114]}               
            self.alpha_w = get_scale(weight_c, z_typical[f'{int(self.nbit)}bit'])

        weight_th = weight_c.clamp(-self.alpha_w.item(), self.alpha_w.item())
        weight_th = self.alpha_w * weight_th / 2 / torch.max(torch.abs(weight_th)) + self.alpha_w / 2

        scale, zero_point = quantizer(self.nbits, 0, abs(self.alpha_w))
        weight_th = STEQuantizer.apply(weight_th, scale, zero_point, True, False)
        weight = 2 * weight_th - self.alpha_w.item()

        alpha_tmp = self.alpha_w.item()

        out = 0
        if input.size(1) !=3:
            weightn = BinarizeWeight(weight,alpha_tmp)
            input_list = BinarizeInput(input)
            
            activation = 0            
            rows, cols = (self.nbits, self.nbits) 
            list_activation = [[0 for i in range(cols)] for j in range(rows)] 
            
            for k in range(self.nbits):
                for n in range(self.nbits):
                    self.activation_list = []
                    self.num_srams = 0
                    if SRAM_rows> input.size(1):
                        input_channels_per_depth =  int(SRAM_rows/input.size(1))
                        num_filter_elements =weightn[k].size(2)*weightn[k].size(3)
                        IX =[]
                        IY = list(range(weightn[k].size(3)))*weightn[k].size(2)
                        for i in range(weightn[k].size(2)):
                            IX.extend([i]*weightn[k].size(3))
                        start_index = 0

                        while start_index < num_filter_elements:
                            W_mask = np.zeros((weightn[k].size(0),weightn[k].size(1),weightn[k].size(2),weightn[k].size(3)))
                            W_mask= torch.from_numpy(W_mask).cuda().float() 
                            end_index = min(start_index + input_channels_per_depth, num_filter_elements)
                            W_mask[:,:,IX[start_index:end_index],IY[start_index:end_index]] = 1
 
                            start_index += input_channels_per_depth
                            partial_input = input_list[n]
                            W1 = weightn[k].data* W_mask
                            ps = nn.functional.conv2d(partial_input, weightn[k]*W_mask,
                               None, self.stride, self.padding, self.dilation, self.groups) 
                       
                            out = quant_XNORSRAM_4d(ps,levels,edges,len(edges),bitlinenoise,offsetnoise)
                            self.activation_list.append(out)
                            self.num_srams += 1
           
                    else:
                        num_input_folds = np.int(np.ceil(weightn[k].size(1)/SRAM_rows))
                        num_channels_per_fold = np.int(weightn[k].size(1)/num_input_folds)
                        for i in range(weightn[k].size(2)):
                            for j in range(weightn[k].size(3)):
                                W_mask = np.zeros((weightn[k].size(0),weightn[k].size(1),weightn[k].size(2),weightn[k].size(3)))
                                W_mask= torch.from_numpy(W_mask).cuda().float() 
                                W_mask[:,:,i,j] =1
                                W1 = weightn[k]* W_mask
                                for k1 in range(num_input_folds): 
                                    start = k1*num_channels_per_fold
                                    finish = (k1+1)*num_channels_per_fold
                                    if k1 == num_input_folds -1:
                                        finish = weightn[k].size(1)
                                    input_index = range(start,finish)
                                    partial_input = input_list[n][:,input_index,:,:]
                                    ps = nn.functional.conv2d(partial_input, weightn[k][:,input_index,:,:]*W_mask[:,input_index,:,:],
                                          None, self.stride, self.padding, self.dilation, self.groups)                               
                                    out = quant_XNORSRAM_4d(ps,levels,edges,len(edges),bitlinenoise,offsetnoise)
                                    self.activation_list.append(out)
                                    self.num_srams +=1
                                   
                    temp = torch.sum(torch.stack(self.activation_list), axis=0)*(1./225)*alpha_tmp      
                    list_activation[k][n] = temp   
           
            for i in range(4):
                for j in range(4):
                    index = i+j
                    if index == 0:
                        activation = 64*list_activation[i][j]
                    elif index == 1:
                        activation += 32*list_activation[i][j]
                    elif index == 2:
                        activation += 16*list_activation[i][j]
                    elif index == 3:
                        activation += 8*list_activation[i][j]
                    elif index == 4:
                        activation += 4*list_activation[i][j]
                    elif index == 5:
                        activation += 2*list_activation[i][j]
                    elif index == 6:
                        activation += 1*list_activation[i][j]
                      
        else:
             activation = nn.functional.conv2d(input, weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)


        

        if not self.bias is None:
             self.bias.org=self.bias.data.clone()
             activation += self.bias.view(1, -1, 1, 1).expand_as(activation)
        
        
        return activation


    def extra_repr(self):
        return super(clamp_conv2d, self).extra_repr() + ', nbit={}, mode={}, k={}'.format(self.nbit, self.mode, self.k)


class clamp_linear(nn.Linear):
    def __init__(self, in_channels, out_channels, bias=True, nbit=4, mode='mean', k=2):
        super(clamp_linear, self).__init__(in_features=in_channels, out_features=out_channels, bias=bias)
        self.nbit=nbit
        self.mode = mode
        self.k = k

    def forward(self, input):
        weight_c = self.weight.clone()

        if self.mode == 'mean':
            self.alpha_w = self.k * weight_c.abs().mean()
        elif self.mode == 'sawb':
            z_typical = {'4bit': [0.077, 1.013], '8bit':[0.027, 1.114]}               
            self.alpha_w = get_scale(weight_c, z_typical[f'{int(self.nbit)}bit'])

        weight_th = weight_c.clamp(-self.alpha_w.item(), self.alpha_w.item())
        weight_th = self.alpha_w * weight_th / 2 / torch.max(torch.abs(weight_th)) + self.alpha_w / 2

        scale, zero_point = quantizer(self.nbit, 0, abs(self.alpha_w))
        weight_th = STEQuantizer.apply(weight_th, scale, zero_point, True, False)
        weight_q = 2 * weight_th - self.alpha_w.item()

        
        SRAM_rows = 256
        self.input_dim = input.size(0)*input.size(1)
        self.num_srams = int(np.ceil(self.input_dim/SRAM_rows))
        weightn = BinarizeWeight(weight_q,self.alpha_w)

        input_data = quantize(input.data, n= self.nbit)
        input_list = BinarizeInput(input)

        for k in range(self.nbit):
            for n in range(self.nbit):
                self.activation_list = []
                for i in range(self.num_srams):
                    start_index = i*SRAM_rows
                    stop_index = np.min(((i+1)*SRAM_rows, self.input_dim))
                    ps = nn.functional.linear(input_list[n][:,start_index:stop_index], weightn[k][:,start_index:stop_index])
                    out = quant_XNORSRAM_2d(ps,levels,edges,len(edges),bitlinenoise,offsetnoise)
                  
                    self.activation_list.append(out)
                temp = torch.sum(torch.stack(self.activation_list), axis=0)*(1./225)*self.alpha_w    
                

                index = k+n
                if index == 0:
                    activation = 64*temp
                elif index == 1:
                    activation += 32*temp
                elif index == 2:
                    activation += 16*temp
                elif index == 3:
                    activation += 8*temp
                elif index == 4:
                    activation += 4*temp
                elif index == 5:
                    activation += 2*temp
                elif index == 6:
                    activation += 1*temp

        
                         
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            activation += self.bias.view(1, -1).expand_as(activation)

        
        return activation
    
    def extra_repr(self):
        return super(clamp_linear, self).extra_repr() + ', nbit={}, mode={}, k={}'.format(self.nbit, self.mode, self.k)


class int_conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=False, nbit=4, mode='mean', k=2, ch_group=16, push=False):
        super(int_conv2d, self).__init__(in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, dilation=dilation, groups=groups,
                bias=bias)
        self.nbit = nbit
        self.mode = mode
        self.k = k
        self.ch_group = ch_group
        self.push = push
        self.iter = 0
        self.mask = torch.ones_like(self.weight).cuda()

    def forward(self, input):
        w_l = self.weight.clone()

        if self.mode == 'mean':
            self.alpha_w = self.k * w_l.abs().mean()
        elif self.mode == 'sawb':
            z_typical = {'4bit': [0.077, 1.013], '8bit':[0.027, 1.114]}               
            self.alpha_w = get_scale(w_l, z_typical[f'{int(self.nbit)}bit'])
        else:
            raise ValueError("Quantization mode must be either 'mean' or 'sawb'! ")                         
            
        weight_q = int_quant_func(nbit=self.nbit, alpha_w=self.alpha_w, restrictRange=True, ch_group=self.ch_group, push=self.push)(w_l)
        w_p = weight_q.clone()
        num_group = w_p.size(0) * w_p.size(1) // self.ch_group
        # if self.push and self.iter is 0:
        if self.push and (self.iter+1) % 4000 == 0:
            print("Inference Prune!")
            kw = weight_q.size(2)
            num_group = w_p.size(0) * w_p.size(1) // self.ch_group
            w_p = w_p.contiguous().view((num_group, self.ch_group, kw, kw))
            
            self.mask = torch.ones_like(w_p)

            for j in range(num_group):
                idx = torch.nonzero(w_p[j, :, :, :])
                r = len(idx) / (self.ch_group * kw * kw)
                internal_sparse = 1 - r

                if internal_sparse >= 0.85 and internal_sparse != 1.0:
                    # print(internal_sparse)
                    self.mask[j, :, :, :] = 0.0

            w_p = w_p * self.mask
            w_p = w_p.contiguous().view((num_group, self.ch_group * kw * kw))
            grp_values = w_p.norm(p=2, dim=1)
            non_zero_idx = torch.nonzero(grp_values) 
            num_nonzeros = len(non_zero_idx)
            zero_groups = num_group - num_nonzeros 
            # print(f'zero groups = {zero_groups}')

            self.mask = self.mask.clone().resize_as_(weight_q)
        
        if not self.push:
            self.mask = torch.ones_like(self.weight)
        
        weight_q = self.mask * weight_q

        output = F.conv2d(input, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
        # if weight_q.size(2) != 1:
        #     print('=========================================')
        #     print(f'size of the input feature map: {list(input.size())}')
        #     print(f'size of the weight: {list(weight_q.size())}, number of groups/col:{num_group}| stride={self.stride} | padding={self.padding}')
        #     print(f'size of the output feature map: {list(output.size())}')
        #     print('=========================================\n')
        self.iter += 1
        return output
    
    def extra_repr(self):
        return super(int_conv2d, self).extra_repr() + ', nbit={}, mode={}, k={}, ch_group={}, push={}'.format(self.nbit, self.mode, self.k, self.ch_group, self.push)


class int_linear(nn.Linear):
    def __init__(self, in_channels, out_channels, bias=True, nbit=8, mode='mean', k=2, ch_group=16, push=False):
        super(int_linear, self).__init__(in_features=in_channels, out_features=out_channels, bias=bias)
        self.nbit=nbit
        self.mode = mode
        self.k = k
        self.ch_group = ch_group
        self.push = push

    def forward(self, input):
        w_l = self.weight.clone()

        if self.mode == 'mean':
            self.alpha_w = self.k * w_l.abs().mean()
        elif self.mode == 'sawb':
            z_typical = {'4bit': [0.077, 1.013], '8bit':[0.027, 1.114]}               
            self.alpha_w = get_scale(w_l, z_typical[f'{int(self.nbit)}bit'])
        else:
            raise ValueError("Quantization mode must be either 'mean' or 'sawb'! ")                         
            
        weight_q = int_quant_func(nbit=self.nbit, alpha_w=self.alpha_w, restrictRange=True, ch_group=self.ch_group, push=self.push)(w_l)
        output = F.linear(input, weight_q, self.bias)

        # print('=========================================')
        # print(f'size of the input feature map: {list(input.size())}')
        # print(f'size of the weight: {list(weight_q.size())} ')
        # print(f'size of the output feature map: {list(output.size())}')
        # print('=========================================\n')

        return output

    def extra_repr(self):
        return super(int_linear, self).extra_repr() + ', nbit={}, mode={}, k={}, ch_group={}, push={}'.format(self.nbit, self.mode, self.k, self.ch_group, self.push)

"""
2-bit quantization
"""

def w2_quant(input, mode='mean', k=2):
    if mode == 'mean':
            alpha_w = k * input.abs().mean()
    elif mode == 'sawb':
        alpha_w = get_scale_2bit(w_l)
    else:
        raise ValueError("Quantization mode must be either 'mean' or 'sawb'! ")
    
    output = input.clone()
    output[input.ge(alpha_w - alpha_w/3)] = alpha_w
    output[input.lt(-alpha_w + alpha_w/3)] = -alpha_w

    output[input.lt(alpha_w - alpha_w/3)*input.ge(0)] = alpha_w/3
    output[input.ge(-alpha_w + alpha_w/3)*input.lt(0)] = -alpha_w/3

    return output

class sawb_w2_Func(torch.autograd.Function):
    def __init__(self, alpha_w):
        super(sawb_w2_Func, self).__init__()
        self.alpha_w = alpha_w 

    def forward(self, input):
        self.save_for_backward(input)
        
        output = input.clone()
        output[input.ge(self.alpha_w - self.alpha_w/3)] = self.alpha_w
        output[input.lt(-self.alpha_w + self.alpha_w/3)] = -self.alpha_w

        output[input.lt(self.alpha_w - self.alpha_w/3)*input.ge(0)] = self.alpha_w/3
        output[input.ge(-self.alpha_w + self.alpha_w/3)*input.lt(0)] = -self.alpha_w/3

        return output
    
    def backward(self, grad_output):
    
        grad_input = grad_output.clone()
        input, = self.saved_tensors
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0

        return grad_input

class Conv2d_2bit(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=False, mode='mean', k=2):
        super(Conv2d_2bit, self).__init__(in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.mode = mode
        self.k = k
        self.alpha_w = 1.

    def forward(self, input):
        w_l = self.weight.clone()

        if self.mode == 'mean':
            self.alpha_w = self.k * w_l.abs().mean()
        elif self.mode == 'sawb':
            self.alpha_w = get_scale_2bit(w_l)
        else:
            raise ValueError("Quantization mode must be either 'mean' or 'sawb'! ") 

        
        weight = sawb_w2_Func(alpha_w=self.alpha_w)(w_l)
        # print(f'quantized weight = {torch.unique(weight)}')
        output = F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
        return output

class Linear2bit(nn.Linear):
    def __init__(self, in_channels, out_channels, bias=True, mode='mean', k=2):
        super(Linear2bit, self).__init__(in_features=in_channels, out_features=out_channels, bias=bias)
        self.mode = mode
        self.k = k

    def forward(self, input):
        w_l = self.weight.clone()

        if self.mode == 'mean':
            self.alpha_w = self.k * w_l.abs().mean()
        elif self.mode == 'sawb':
            self.alpha_w = get_scale_2bit(w_l)
        else:
            raise ValueError("Quantization mode must be either 'mean' or 'sawb'! ") 
        
        weight_q = sawb_w2_Func(alpha_w=self.alpha_w)(w_l)

        output = F.linear(input, weight_q, self.bias)
        return output

    def extra_repr(self):
        return super(Linear2bit, self).extra_repr() + ', mode={}, k={}'.format(self.mode, self.k)

"""
zero skipping quantization
"""

class zero_skp_quant(torch.autograd.Function):
    def __init__(self, nbit, coef, group_ch, alpha_w):
        super(zero_skp_quant, self).__init__()
        self.nbit = nbit
        self.coef = coef
        self.group_ch = group_ch
        self.alpha_w = alpha_w
    
    def forward(self, input):
        self.save_for_backward(input)
        interval = 2*self.alpha_w / (2**self.nbit - 1) / 2
        self.th = self.coef * interval

        cout = input.size(0)
        cin = input.size(1)
        kh = input.size(2)
        kw = input.size(3)
        num_group = (cout * cin) // self.group_ch

        w_t = input.view(num_group, self.group_ch*kh*kw)

        grp_values = w_t.norm(p=2, dim=1)                                               # L2 norm
        mask_1d = grp_values.gt(self.th*self.group_ch*kh*kw).float()
        mask_2d = mask_1d.view(w_t.size(0),1).expand(w_t.size()) 

        w_t = w_t * mask_2d

        non_zero_idx = torch.nonzero(mask_1d).squeeze(1)                             # get the indexes of the nonzero groups
        non_zero_grp = w_t[non_zero_idx]                                             # what about the distribution of non_zero_group?
        
        weight_q = non_zero_grp.clone()
        alpha_w = get_scale_2bit(weight_q)

        weight_q[non_zero_grp.ge(self.alpha_w - self.alpha_w/3)] = self.alpha_w
        weight_q[non_zero_grp.lt(-self.alpha_w + self.alpha_w/3)] = -self.alpha_w
        
        weight_q[non_zero_grp.lt(self.alpha_w - self.alpha_w/3)*non_zero_grp.ge(0)] = self.alpha_w/3
        weight_q[non_zero_grp.ge(-self.alpha_w + self.alpha_w/3)*non_zero_grp.lt(0)] = -self.alpha_w/3

        # print(f'INT levels:{weight_q.unique()}')
        w_t[non_zero_idx] = weight_q
        
        output = w_t.clone().resize_as_(input)
        return output

    def backward(self, grad_output):
        grad_input = grad_output.clone()
        input, = self.saved_tensors
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input    

class Conv2d_W2_IP(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=False, nbit=2, mode='mean', k=2, skp_group=16, gamma=0.3):
        super(Conv2d_W2_IP, self).__init__(in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, dilation=dilation, groups=groups,
                bias=bias)
        self.nbit = nbit
        self.coef = gamma
        self.skp_group = skp_group
        self.mode = mode
        self.k=k

    def forward(self, input):
        weight = self.weight

        if self.mode == 'mean':
            self.alpha_w = self.k * weight.abs().mean()
        elif self.mode == 'sawb':
            self.alpha_w = get_scale_2bit(weight)
        else:
            raise ValueError("Quantization mode must be either 'mean' or 'sawb'! ")

        weight_q = zero_skp_quant(nbit=self.nbit, coef=self.coef, group_ch=self.skp_group, alpha_w=self.alpha_w)(weight)
        output = F.conv2d(input, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
        return output
    
    def extra_repr(self):
        return super(zero_grp_skp_quant, self).extra_repr() + ', nbit={}, coef={}, skp_group={}'.format(
                self.nbit, self.coef, self.skp_group)       

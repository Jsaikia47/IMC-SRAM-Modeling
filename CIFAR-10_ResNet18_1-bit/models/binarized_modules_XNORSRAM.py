import torch
import pdb
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd import Function
import numpy as np

def Binarize(tensor,quant_mode='det'):
    if quant_mode=='det':
        return tensor.sign()
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)
    
      
def Ternarize(tensor,quant_mode='det'):
    if quant_mode=='det':
        return tensor.clamp_(-1,1).round()
    else: # ignore this branch
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)

def dec2binary(x, n):
    y =[]
    if n ==1:
        y.append(x)
        return y
    base = (2**(n-1))/(2**n -1)
    comp = x
    for i in range(n):
        temp = torch.ge(comp,0).float().cuda()*2 -1
        comp -= temp*base
        y.append(temp)
        base = base/2
    return y


class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss,self).__init__()
        self.margin=1.0

    def hinge_loss(self,input,target):
            #import pdb; pdb.set_trace()
            output=self.margin-input.mul(target)
            output[output.le(0)]=0
            return output.mean()

    def forward(self, input, target):
        return self.hinge_loss(input,target)

class SqrtHingeLossFunction(Function):
    def __init__(self):
        super(SqrtHingeLossFunction,self).__init__()
        self.margin=1.0

    def forward(self, input, target):
        output=self.margin-input.mul(target)
        output[output.le(0)]=0
        self.save_for_backward(input, target)
        loss=output.mul(output).sum(0).sum(1).div(target.numel())
        return loss

    def backward(self,grad_output):
       input, target = self.saved_tensors
       output=self.margin-input.mul(target)
       output[output.le(0)]=0
       import pdb; pdb.set_trace()
       grad_output.resize_as_(input).copy_(target).mul_(-2).mul_(output)
       grad_output.mul_(output.ne(0).float())
       grad_output.div_(input.numel())
       return grad_output,grad_output

def Quantize(tensor,quant_mode='det',  params=None, numBits=8):
    tensor.clamp_(-2**(numBits-1),2**(numBits-1))
    if quant_mode=='det':
        tensor=tensor.mul(2**(numBits-1)).round().div(2**(numBits-1))
    else:
        tensor=tensor.mul(2**(numBits-1)).round().add(torch.rand(tensor.size()).add(-0.5)).div(2**(numBits-1))
        quant_fixed(tensor, params)
    return tensor
      

def quantize(tensor, H=1., n =1):
    if n ==1:
        return tensor.sign()*H
    tensor = torch.round((torch.clamp(tensor,-H,H)+H) * (2**n-1)/(2*H)) *2*H/ (2**n -1) -H
    return tensor
    

def quant_XNORSRAM_2d(x,levels,edges,num_edges,bitlinenoise,offsetnoise):
     bitline = torch.randn(x.shape,device='cuda:0')*bitlinenoise
     x = x + bitline

     sum = 0
     edges = edges.unsqueeze(0).repeat(x.shape[1],1)
     
     offset = torch.randn(x.shape[1]).cuda()
     offset = offset*offsetnoise
     offsets = offset.unsqueeze(1).repeat(1,num_edges)   
     edges = edges + offsets.cuda() 
     
     for i in range(num_edges):
         y= edges[:,i].unsqueeze(0).repeat(x.shape[0],1)
         sum += torch.gt(x,y)
     output = levels[sum]
     return output
     
def quant_XNORSRAM_4d(x,levels,edges,num_edges,bitlinenoise,offsetnoise):
     bitline = torch.randn(x.shape,device='cuda:0')*bitlinenoise
     x = x + bitline

     sum = 0
     edges = edges.unsqueeze(0).repeat(x.shape[1],1).cuda()
    
     offset = torch.randn(x.shape[1]).cuda()
     offset = offset*offsetnoise
     offsets = offset.unsqueeze(1).repeat(1,num_edges)
     edges = edges + offsets.cuda() 
    
     for i in range(num_edges):
         y= edges[:,i].unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(x.shape[0],1,x.shape[2],x.shape[3])
         sum += torch.gt(x,y)
     output = levels[sum]
     return output


class LinearV1(nn.Linear):
    def __init__(self, *kargs, **kwargs):
        super(LinearV1, self).__init__(*kargs, **kwargs)
        self.activation_list =[]
        self.n_bits = n_bits
        self.n_frac = n_bits -1
        self.base = 2. **(self.n_bits-1)/(2. **(self.n_frac +1 ) - 1. )
        self.alpha = 1   # for non-binary models

    def forward(self, input):
        base = self.base
        activation = 0
        self.input_dim = input.size(0)*input.size(1)
        self.num_srams = int(np.ceil(self.input_dim/SRAM_rows))

        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=Binarize(self.weight.org)
        
        out = 0
        
        #For binary model
        input_list = quantize(input.data, n= self.n_bits)
        input_list = dec2binary(input_list, self.n_bits)

        bound = Bound
        n_levels = num_levels

        step = 2*bound/(n_levels-1)
        levels = np.linspace(-bound-step,bound,int(n_levels+1))
        levels[0] = -bound
        levels = torch.from_numpy(levels).cuda().float()
        levels = levels * 100.0
        levels = levels.long()
        levels = levels / 100.0

        edges = np.linspace(-bound-0.5*step,bound-0.5*step,int(n_levels))
        edges = torch.from_numpy(edges).cuda().float()
        edges = edges * 100.0
        edges = edges.long()
        edges = edges / 100.0
          
        for n in range(self.n_bits):
            self.activation_list = []
            for i in range(self.num_srams):
                start_index = i*SRAM_rows
                stop_index = np.min(((i+1)*SRAM_rows, self.input_dim))
                ps = nn.functional.linear(input_list[n][:,start_index:stop_index], self.weight[:,start_index:stop_index])
                out = quant_XNORSRAM_2d(ps,levels, edges, len(edges), bitlinenoise,offsetnoise)
                self.activation_list.append(out)
            if self.num_srams>1:
                activation += torch.sum(torch.stack(self.activation_list), axis=0) *base
            else:
                activation += self.activation_list[0] *base
            base/= 2.
        
        activation = activation*self.alpha
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            activation += self.bias.view(1, -1).expand_as(activation)
        return activation


class Conv2dV1(nn.Conv2d):

    def __init__(self,  *kargs, **kwargs):
        super(Conv2dV1, self).__init__(*kargs, **kwargs)
        self.activation_list = []
        self.partialsum_ideal_list = []
        self.partialsum_noisy_list = []
        self.n_bits= n_bits
        self.n_frac= n_bits-1
        self.base = 2. **(self.n_bits-1)/(2. **(self.n_frac +1 ) - 1. )
        self.alpha =1

    def forward(self, input):
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        
        self.weight.data=Binarize(self.weight.org)
        
        input_list1 = quantize(input.data, n= self.n_bits)
        self.ideal_out = nn.functional.conv2d(input_list1, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)
        
        bound = Bound
        n_levels = num_levels
        
        step = 2*bound/(n_levels-1)
        levels = np.linspace(-bound-step,bound,int(n_levels+1))
        levels[0] = -bound
        levels = torch.from_numpy(levels).cuda().float()
        levels = levels * 100.0
        levels = levels.long()
        levels = levels / 100.0

        edges = np.linspace(-bound-0.5*step,bound-0.5*step,int(n_levels))
        edges = torch.from_numpy(edges).cuda().float()
        edges = edges.long()
        
        out = 0
        if input.size(1) !=3:
            input_list = dec2binary(input_list1, self.n_bits)
               
            base = self.base
            activation = 0 
            for n in range(self.n_bits):
                self.activation_list = []
                self.partialsum_ideal_list = []
                self.partialsum_noisy_list = []
                self.num_srams = 0
                if SRAM_rows> input.size(1):
                    input_channels_per_depth =  int(SRAM_rows/input.size(1))
                    num_filter_elements = self.weight.size(2)*self.weight.size(3)
                    IX =[]
                    IY = list(range(self.weight.size(3)))*self.weight.size(2)
                    for i in range(self.weight.size(2)):
                        IX.extend([i]*self.weight.size(3))
                    start_index = 0

                    while start_index < num_filter_elements:
                        W_mask = np.zeros((self.weight.size(0),self.weight.size(1),self.weight.size(2),self.weight.size(3)))
                        W_mask= torch.from_numpy(W_mask).cuda().float() 
                        end_index = min(start_index + input_channels_per_depth, num_filter_elements)
                        W_mask[:,:,IX[start_index:end_index],IY[start_index:end_index]] = 1
 
                        start_index += input_channels_per_depth
                        partial_input = input_list[n]
                        W1 = self.weight.data* W_mask
                        ps = nn.functional.conv2d(partial_input, self.weight.data*W_mask,
                           None, self.stride, self.padding, self.dilation, self.groups) 
                       
                        out = quant_XNORSRAM_4d(ps,levels, edges, len(edges), bitlinenoise,offsetnoise)
                        
                        self.activation_list.append(out)
                        self.partialsum_ideal_list.append(ps)
                        self.partialsum_noisy_list.append(out)
                        
                        self.num_srams += 1
           
                else:
                    num_input_folds = np.int(np.ceil(self.weight.size(1)/SRAM_rows))
                    num_channels_per_fold = np.int(self.weight.size(1)/num_input_folds)
                    
                    for i in range(self.weight.size(2)):
                        for j in range(self.weight.size(3)):
                            W_mask = np.zeros((self.weight.size(0),self.weight.size(1),self.weight.size(2),self.weight.size(3)))
                            W_mask= torch.from_numpy(W_mask).cuda().float() 
                            W_mask[:,:,i,j] =1
                            
                            W1 = self.weight.data* W_mask
                            
                            for k in range(num_input_folds): 
                                start = k*num_channels_per_fold
                                finish = (k+1)*num_channels_per_fold
                                if k == num_input_folds -1:
                                    finish = self.weight.size(1)
                                input_index = range(start,finish)
                                partial_input = input_list[n][:,input_index,:,:]
                                ps = nn.functional.conv2d(partial_input, self.weight.data[:,input_index,:,:]*W_mask[:,input_index,:,:],
                                      None, self.stride, self.padding, self.dilation, self.groups) 
                                
                                out = quant_XNORSRAM_4d(ps,levels, edges, len(edges), bitlinenoise,offsetnoise)
                                
                                self.activation_list.append(out)
                                self.partialsum_ideal_list.append(ps)
                                self.partialsum_noisy_list.append(out)
                                self.num_srams +=1
                                
                if self.num_srams>1:
                    activation += torch.sum(torch.stack(self.activation_list), axis=0) *base * self.alpha
                    
                else:
                    activation += self.activation_list[0] *base *self.alpha
                base/= 2.
                
        else:
             activation = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)
             
        if not self.bias is None:
             self.bias.org=self.bias.data.clone()
             activation += self.bias.view(1, -1, 1, 1).expand_as(activation)
       
        self.out = activation
        return activation
        return out

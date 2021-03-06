"""
Multi-bit Quantizers
"""

import torch
from torch import nn
from torch import autograd

def dec2binary_act(x, n_bits, alpha):
    '''
    assuming x is quantized to [-1, +1]
    '''
    if n_bits == 1:
        return [x]
    base = 2.**(n_bits-1) / (2**n_bits-1.) * alpha
    # print(f'alpha in dec2binary_act = {alpha}')
    y = []
    x_remain = x.clone()
    for i in range(n_bits): # from MSB to LSB
        xb = x * base * 2**(-i)
        xb.data = x_remain.gt(0.75*base*2**(-i)).float()
        y.append(xb)
        x_remain -= xb * base*2**(-i)
        # base /= 2.
    return y


def dec2binary_weight(x, n_bits, alpha):
    '''
    assuming x is quantized to [-1, +1]
    '''
    if n_bits == 1:
        return [x]
    base = 2.**(n_bits-1) / (2**n_bits-1.) * alpha
    y = []
    x_remain = x.clone()
    for i in range(n_bits): # from MSB to LSB
        xb = x * base * 2**(-i)
        xb.data = torch.sign(x_remain)
        y.append(xb)
        x_remain -= xb * base * 2**(-i)
        # base /= 2.
    return y


def get_scale_2bit(input):
    c1, c2 = 3.212, -2.178
    
    std = input.std()
    mean = input.abs().mean()
    
    q_scale = c1 * std + c2 * mean
    
    return q_scale 

def to_tensor(sat_val):
    is_scalar = not isinstance(sat_val, torch.Tensor)
    out = torch.tensor(sat_val) if is_scalar else sat_val.clone().detach()
    if not out.is_floating_point():
        out = out.to(torch.float32)
    if out.dim() == 0:
        out = out.unsqueeze(0)
    return is_scalar, out

def linear_quantize(input, scale, zero_point, inplace=False):
    """
    Linearly quantize the input tensor based on scale and zero point.
    https://pytorch.org/docs/stable/quantization.html
    """
    if inplace:
        input.mul_(scale).sub_(zero_point).round_()
        return input
    return torch.round(input * scale - zero_point)

def linear_dequantize(input, scale, zero_point, inplace=False):
    if inplace:
        input.add_(zero_point).div_(scale)
        return input
    return (input + zero_point) / scale

def quantizer(num_bits, saturation_min, saturation_max,
                                          integral_zero_point=True, signed=False):
    scalar_min, sat_min = to_tensor(saturation_min)
    scalar_max, sat_max = to_tensor(saturation_max)
    is_scalar = scalar_min and scalar_max

    if scalar_max and not scalar_min:
        sat_max = sat_max.to(sat_min.device)
    if scalar_min:
        sat_min = sat_min.to(sat_max.device)

    if any(sat_min > sat_max):
        import pdb; pdb.set_trace()
        raise ValueError('saturation_min must be smaller than saturation_max')

    n = 2 ** num_bits - 1

    # Make sure 0 is in the range
    sat_min = torch.min(sat_min, torch.zeros_like(sat_min))
    sat_max = torch.max(sat_max, torch.zeros_like(sat_max))

    diff = sat_max - sat_min
    diff[diff == 0] = n

    scale = n / diff
    zero_point = scale * sat_min
    if integral_zero_point:
        zero_point = zero_point.round()
    if signed:
        zero_point += 2 ** (num_bits - 1)
    if is_scalar:
        return scale.item(), zero_point.item()
    return scale, zero_point

class SAWB_2bit_Func(torch.autograd.Function):

    def __init__(self, alpha):
        super(SAWB_2bit_Func, self).__init__()
        self.alpha = alpha

    def forward(self, input):
        self.save_for_backward(input)
        
        output = input.clone()
        output[input.ge(self.alpha - self.alpha/3)] = self.alpha
        output[input.lt(-self.alpha + self.alpha/3)] = -self.alpha
        
        output[input.lt(self.alpha - self.alpha/3)*input.ge(0)] = self.alpha.div(3)
        output[input.ge(-self.alpha + self.alpha/3)*input.lt(0)] = -self.alpha.div(3)

        return output
    
    def backward(self, grad_output):
        # print('SAWB backward!') 
        grad_input = grad_output.clone()
        input, = self.saved_tensors
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0

        return grad_input

class STEQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, zero_point, dequantize, inplace):
        if inplace:
            ctx.mark_dirty(input)

        output = linear_quantize(input, scale, zero_point)
        if dequantize:
            output = linear_dequantize(output, scale, zero_point)  
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Straight Through Estimator
        """
        # print(f'PACT Backward! | grad_output size={grad_output.size()}')
        return grad_output, None, None, None, None

class PACT_MultiBit(nn.Module):
    def __init__(self, n_bits, base_a=None):
        super(PACT_MultiBit, self).__init__()
        self.n_bits = n_bits
        self.base_a = base_a

    def forward(self, input):
        """
        From MSB to LSB
        """
        if not self.base_a is None:
            base = self.base_a
            # print(base)
        else:
            base = 2.**(self.n_bits-1) / (2**self.n_bits-1.)

        y = []
        input_remain = input.clone()
        for i in range(self.n_bits):
            xb = input * base
            xb.data = torch.sign(input_remain)
            y.append(xb)
            input_remain -= xb * base
            base /= 2.
        return y

class SAWB_MultiBit(nn.Module):
    def __init__(self, n_bits=2, base_w=None):
        super(SAWB_MultiBit, self).__init__()
        self.n_bits = n_bits
        self.base_w = base_w
    
    def forward(self, input):
        if not self.base_w is None:
            base = self.base_w
        else:
            base = 2.**(self.n_bits-1) / (2**self.n_bits-1.)

        y = []
        input_remain = input.clone()
        for i in range(self.n_bits):
            # print(base)
            xb = input * base
            xb.data = input_remain.ge(base).float()
            y.append(xb)
            input_remain -= xb * base
            base /= 2.
        
        return y


class PACT_Quant(nn.Module):
    def __init__(self, n_bits=2, alpha=10.0, inplace=False, dequantize=True):
        super(PACT_Quant, self).__init__()
        self.n_bits = n_bits
        self.alpha = alpha
        self.inplace = inplace
        self.dequantize = dequantize

    def forward(self, input):
        # input_ = input.clone()
        input[input.ge(self.alpha)] = self.alpha
        input[input.le(0)] = 0

        with torch.no_grad():
            scale, zero_point = quantizer(self.n_bits, 0, self.alpha)
        output = STEQuantizer.apply(input, scale, zero_point, self.dequantize, self.inplace)
        return output

class SAWB_Quant(nn.Module):
    def __init__(self, n_bits=2, alpha_w=1.0):
        super(SAWB_Quant, self).__init__()
        self.n_bits = n_bits
        self.alpha_w = alpha_w
        self.quantizer = SAWB_2bit_Func(alpha_w)

    def update_alpha(self, alpha):
        self.alpha_w = alpha
        self.quantizer = SAWB_2bit_Func(self.alpha_w)

    def forward(self, input):
        output = self.quantizer(input)
        return output


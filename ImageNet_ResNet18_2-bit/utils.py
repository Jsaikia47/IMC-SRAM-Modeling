'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.init as init
from models.quant import QuantizeLinear, QuantizeConv2d, QuantizeActLayer, BatchNorm2d, BatchNorm1d
from models.quant_modules import QuantizedConv2d

import scipy.io as sio

def get_loss_for_H(model, weight_decay=1e-4):
    loss = 0
    for m in model.modules():
        if isinstance(m, QuantizeLinear) or isinstance(m, QuantizeConv2d):
            loss += -1e-5 * torch.log(m.H)
            #loss += -1/2 * weight_decay * (m.H ** 2)
        #elif isinstance(m, QuantizeActLayer):
        #    loss += -1e-5 * torch.log(m.H)
            #loss += -1/2 * weight_decay * (m.H ** 2)
    return loss

def update_sram_depth(model, sram_depth, quant_bound, prob_table, prob_table_9, noise_std, noise_9_std, noise_9_mean):
    if prob_table is not None:
        data = sio.loadmat(prob_table)
        cdf_table = torch.tensor(data['prob']).cuda().cumsum(dim=-1).float()
        if cdf_table.shape[0] == sram_depth + 1: #step_size = 2, intopolate to step_size = 1
            cdf_table_interpolated = torch.zeros((2*sram_depth+1, cdf_table.shape[1]-1)).cuda()
            for i in range(cdf_table.shape[0]):
                cdf_table_interpolated[i*2] = cdf_table[i,0:-1]
            for i in range(cdf_table.shape[0]-1):
                cdf_table_interpolated[i*2+1] = (cdf_table_interpolated[i*2] + cdf_table_interpolated[i*2+2])/2.
        else:
             cdf_table_interpolated = cdf_table[:,0:-1].contiguous()
    if prob_table_9 is not None:
        data = sio.loadmat(prob_table_9)
        cdf_table_9 = torch.tensor(data['prob']).cuda().cumsum(dim=-1).float()
        cdf_table_interpolated_9 = cdf_table_9[:,0:-1].contiguous()

    for m in model.modules():
        if isinstance(m, QuantizeLinear) or isinstance(m, QuantizedConv2d):
            m.sram_depth = sram_depth
            m.quant_bound = quant_bound
            m.noise_std = noise_std
            if isinstance(m, QuantizedConv2d):
                m.noise_9_std = noise_9_std
                m.noise_9_mean = noise_9_mean
            if prob_table is not None:
                m.cdf_table = torch.nn.Embedding.from_pretrained(cdf_table_interpolated)
            if isinstance(m, QuantizedConv2d) and prob_table_9 is not None:
                m.cdf_table_9 = torch.nn.Embedding.from_pretrained(cdf_table_interpolated_9)

class Hook_record_input():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input

    def close(self):
        self.hook.remove()

def add_input_hook_for_2nd_conv_layer(model):
    Hook = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            if module.in_channels != 3:
                Hook = Hook_record_input(module)
                break
    return Hook

def get_input_for_2nd_conv_layer_from_Hook(Hook):
    return Hook.input[0]


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 50.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append(' | Tot: %.1fs' % tot_time)
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

class SGD_binary(torch.optim.SGD):
    '''
    Customized SGD optimizer for binary-weight models
    '''
    def __init__(self, params, lr=0.1, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, binary=True):
        super(SGD_binary, self).__init__(params=params, lr=lr, momentum=momentum,
                 dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        self.binary = binary
    
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                # scale d_p with p.lr_scale
                if self.binary and hasattr(p, 'lr_scale'):
                    d_p = d_p.mul_(p.lr_scale)

                p.data.add_(-group['lr'], d_p)
        return loss

__optimizers = {
    'SGD_binary': SGD_binary,
    'SGD': torch.optim.SGD,
    'ASGD': torch.optim.ASGD,
    'Adam': torch.optim.Adam,
    'Adamax': torch.optim.Adamax,
    'Adagrad': torch.optim.Adagrad,
    'Adadelta': torch.optim.Adadelta,
    'Rprop': torch.optim.Rprop,
    'RMSprop': torch.optim.RMSprop
}


def adjust_optimizer(optimizer, epoch, config):
    """Reconfigures the optimizer according to epoch and config dict"""
    def modify_optimizer(optimizer, setting):
        if 'optimizer' in setting:
            optimizer = __optimizers[setting['optimizer']](
                optimizer.param_groups)
        for param_group in optimizer.param_groups:
            for key in param_group.keys():
                if key in setting:
                    param_group[key] = setting[key]
        return optimizer

    if callable(config):
        optimizer = modify_optimizer(optimizer, config(epoch))
    else:
        for e in range(epoch + 1):  # run over all epochs - sticky setting
            if e in config:
                optimizer = modify_optimizer(optimizer, config[e])

    return optimizer

def simplify_BN_parameters(model):
    ml = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            ml.append(m)
        if isinstance(m, BatchNorm2d) or isinstance(m, BatchNorm1d):
            ml.append(m)
        if isinstance(m, QuantizeActLayer):
            m.mode = 'hardware'

    num_pairs = len(ml) // 2
    biases = {}
    weights = {}
    for p in range(1, num_pairs): # skip the first pair
        if (isinstance(ml[2*p], nn.Conv2d) and isinstance(ml[2*p+1], BatchNorm2d)) or \
           (isinstance(ml[2*p], nn.Linear) and isinstance(ml[2*p+1], BatchNorm1d)):
            ml[2*p].mode = 'hardware'
            ml[2*p+1].mode = 'hardware'
            ml[2*p+1].weight_effective.data = ml[2*p+1].weight / torch.sqrt(ml[2*p+1].running_var + ml[2*p+1].eps)
            ml[2*p+1].bias_effective.data = ml[2*p+1].weight_effective * (ml[2*p].bias - ml[2*p+1].running_mean) + ml[2*p+1].bias
            ml[2*p].bias.data.zero_()
            ml[2*p+1].weight_effective.data *= (ml[2*p].quant_bound / 5.) # 11 levels, uniform-spaced quantization
            # rectify effective weights
            if torch.any(ml[2*p+1].weight_effective < 0):
                IX = torch.where(ml[2*p+1].weight_effective < 0)[0]
                ml[2*p+1].weight_effective.data[IX] *= -1.
                ml[2*p].weight.data[IX] *= -1. # note: this is still not equivalent when max pooling is considered.
            # comment below if it's not -1/+1 activation
            ml[2*p+1].bias_effective.data /= ml[2*p+1].weight_effective
            ml[2*p+1].weight_effective.data.fill_(1.)
            # round bias to int8 (ceiling)
            ml[2*p+1].bias_effective.data.clamp_(-127, 128).ceil_() # -bias_effective is the actual data load to HW
            bias = ml[2*p+1].bias_effective.data.cpu().numpy()
            biases['B%d' % p] = (-bias).astype('int8')
            weights['W%d' % p] = ml[2*p].weight.data.cpu().numpy().astype('int8')
    biases['B%d' % (p+1)] = (ml[2*p+2].bias.data.cpu().numpy() * 5. / ml[2*p+2].quant_bound).round().astype('int8')
    weights['W%d' % (p+1)] = ml[2*p+2].weight.data.cpu().numpy().astype('int8')


    sio.savemat('results/Biases.mat', biases)
    sio.savemat('results/Weights.mat', weights)



    if isinstance(ml[-1], nn.Linear):
        ml[-1].bias.data.div_(ml[-1].quant_bound / 5.)

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch
from torch.nn import init
from .quant import ClippedReLU, int_conv2d, int_linear, ClippedHardTanh, Conv2d_2bit, Linear2bit

__all__ = ['resnet18_quant_tanh']

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, wbit=4, abit=4, alpha_init=10, mode='mean', k=2, ch_group=16, push=False, bitlinenoise = 0., offsetnoise=0.,levels = 15.0, depth = 256, voltage_swing = 0.01):
        super(BasicBlock, self).__init__()
        
        if wbit == 32:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        elif wbit == 2:
            self.conv1 = Conv2d_2bit(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, mode=mode, k=k, bitlinenoise=bitlinenoise, offsetnoise=offsetnoise, levels=levels, depth =depth, voltage_swing = voltage_swing)
        else:
            self.conv1 = int_conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, nbit=wbit, mode=mode, k=k, ch_group=ch_group, push=push)
        
        if abit == 32:
            # self.relu1 = nn.ReLU(inplace=True)
            self.relu1 = nn.Hardtanh(inplace=True)
        else:
            # self.relu1 = ClippedHardTanh(num_bits=abit, inplace=True)    # Clipped ReLU function 4 - bits
            self.relu1 = ClippedHardTanh(num_bits=abit, inplace=True)

        self.bn1 = nn.BatchNorm2d(planes)
        
        if wbit == 32:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        elif wbit == 2:
            self.conv2 = Conv2d_2bit(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, mode=mode, k=k, bitlinenoise=bitlinenoise, offsetnoise=offsetnoise, levels=levels, depth = depth, voltage_swing = voltage_swing)
        elif wbit == 4:
            self.conv2 = int_conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, nbit=wbit, mode=mode, k=k, ch_group=ch_group, push=push)
        
        
        if abit == 32:
            # self.relu2 = nn.ReLU(inplace=True)
            self.relu2 = nn.Hardtanh(inplace=True)
        else:
            # self.relu2 = ClippedHardTanh(num_bits=abit, inplace=True)    # Clipped ReLU function 4 - bits
            self.relu2 = ClippedHardTanh(num_bits=abit, inplace=True)

        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                # int_conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False, nbit=wbit, mode=mode, k=k, ch_group=ch_group, push=False),
                # nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                Conv2d_2bit(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False, mode=mode, k=k, bitlinenoise = bitlinenoise, offsetnoise = offsetnoise, levels = levels, depth =depth, voltage_swing=voltage_swing),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(self.bn1(out))
        
        out = self.conv2(out)
        out = self.bn2(out)
      

        out += self.shortcut(x)
        out = self.relu2(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, wbit=4, abit=4, alpha_init=10, mode='mean', k=2, bitlinenoise = 0. , offsetnoise = 0., levels= 15.0, depth = 256, voltage_swing = 0.01, ch_group=16, push=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # self.conv1 = int_conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False, nbit=wbit, mode=mode, k=k, ch_group=ch_group, push=False)
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = Conv2d_2bit(3, 64, kernel_size=3, stride=1, padding=1, bias=False, mode=mode, k=k, bitlinenoise = bitlinenoise, offsetnoise = offsetnoise, levels=levels, depth = depth, voltage_swing =voltage_swing)
        self.bn1 = nn.BatchNorm2d(64)
        
        if abit == 32:
            # self.relu0 = nn.ReLU(inplace=True)
            self.relu0 = nn.Hardtanh(inplace=True)
        else:
            self.relu0 = ClippedHardTanh(num_bits=abit, inplace=True)    # Clipped ReLU function 4 - bits

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, wbit=wbit, abit=abit, alpha_init=alpha_init, mode=mode, k=k, ch_group=ch_group, push=push, bitlinenoise = bitlinenoise, offsetnoise = offsetnoise, levels =levels, depth = depth, voltage_swing = voltage_swing)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, wbit=wbit, abit=abit, alpha_init=alpha_init, mode=mode, k=k, ch_group=ch_group, push=push, bitlinenoise = bitlinenoise, offsetnoise = offsetnoise, levels=levels, depth = depth, voltage_swing = voltage_swing)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, wbit=wbit, abit=abit, alpha_init=alpha_init, mode=mode, k=k, ch_group=ch_group, push=push, bitlinenoise = bitlinenoise, offsetnoise = offsetnoise, levels=levels, depth = depth, voltage_swing = voltage_swing)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, wbit=wbit, abit=abit, alpha_init=alpha_init, mode=mode, k=k, ch_group=ch_group, push=push, bitlinenoise = bitlinenoise, offsetnoise = offsetnoise, levels =levels, depth = depth, voltage_swing =voltage_swing)
        # self.linear = int_linear(512*block.expansion, num_classes, nbit=wbit, mode=mode, k=k, ch_group=ch_group, push=False)
        # self.linear = nn.Linear(512*block.expansion, num_classes)
        self.linear = Linear2bit(512*block.expansion, num_classes, mode=mode, k=k, bitlinenoise = bitlinenoise, offsetnoise = offsetnoise, voltage_swing = voltage_swing)


    def _make_layer(self, block, planes, num_blocks, stride, wbit=4, abit=4, alpha_init=10, mode='mean', k=2, ch_group=16, push=False, bitlinenoise = 0., offsetnoise = 0., levels=15.0, depth = 256, voltage_swing = 0.01):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, wbit=wbit, abit=abit, alpha_init=alpha_init, mode=mode, k=k, ch_group=ch_group, push=push, bitlinenoise = bitlinenoise, offsetnoise=offsetnoise, levels=levels, depth = depth, voltage_swing = voltage_swing))
            self.in_planes = planes* block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)

        # out = F.relu(self.bn1(out))
        out = self.relu0(self.bn1(out))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

class resnet18_quant_tanh:
    base = ResNet
    args = list()
    kwargs = {'block': BasicBlock, 'num_blocks': [2, 2, 2, 2]}


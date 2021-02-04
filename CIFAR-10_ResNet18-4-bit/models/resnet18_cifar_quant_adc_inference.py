import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch
from torch.nn import init
from .quant import ClippedReLU, int_conv2d, int_linear, Qconv2d_adc

__all__ = ['resnet18_quant_eval']

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, wbit=4, abit=4, alpha_init=10, mode='mean', k=2, col_size=16, ch_group=16, ADCprecision=5, cellBit=2):
        super(BasicBlock, self).__init__()
        # self.conv1 = int_conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, nbit=wbit, mode=mode, k=k, ch_group=ch_group, push=push)
        self.conv1 = Qconv2d_adc(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False,
                    col_size=col_size, group_size=ch_group, wl_input=abit,wl_weight=wbit,inference=1, mode=mode, k=k, cellBit=cellBit,ADCprecision=ADCprecision)

        if abit == 32:
            self.relu1 = nn.ReLU(inplace=True)
        else:
            self.relu1 = ClippedReLU(num_bits=abit, alpha=alpha_init, inplace=True)    # Clipped ReLU function 4 - bits

        self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = int_conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, nbit=wbit, mode=mode, k=k, ch_group=ch_group, push=push)
        self.conv2 = Qconv2d_adc(planes, planes, kernel_size=3, stride=1, padding=1, bias=False,
                                col_size=col_size, group_size=ch_group, wl_input=abit, wl_weight=wbit, inference=1, mode=mode, k=k, cellBit=cellBit,ADCprecision=ADCprecision)  # quantization))
        
        if abit == 32:
            self.relu2 = nn.ReLU(inplace=True)
        else:
            self.relu2 = ClippedReLU(num_bits=abit, alpha=alpha_init, inplace=True)    # Clipped ReLU function 4 - bits

        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                int_conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False, nbit=wbit, mode=mode, k=k, ch_group=ch_group, push=False),
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
    def __init__(self, block, num_blocks, num_classes=10, wbit=4, abit=4, alpha_init=10, mode='mean', k=2, col_size=16, ch_group=16, ADCprecision=5, cellBit=2):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = int_conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False, nbit=wbit, mode=mode, k=k, ch_group=ch_group, push=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        if abit == 32:
            self.relu0 = nn.ReLU(inplace=True)
        else:
            self.relu0 = ClippedReLU(num_bits=abit, alpha=alpha_init, inplace=True)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, wbit=wbit, abit=abit, alpha_init=alpha_init, mode=mode, k=k, col_size=col_size, ch_group=ch_group, ADCprecision=ADCprecision, cellBit=cellBit)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, wbit=wbit, abit=abit, alpha_init=alpha_init, mode=mode, k=k, col_size=col_size, ch_group=ch_group, ADCprecision=ADCprecision, cellBit=cellBit)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, wbit=wbit, abit=abit, alpha_init=alpha_init, mode=mode, k=k, col_size=col_size, ch_group=ch_group, ADCprecision=ADCprecision, cellBit=cellBit)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, wbit=wbit, abit=abit, alpha_init=alpha_init, mode=mode, k=k, col_size=col_size, ch_group=ch_group, ADCprecision=ADCprecision, cellBit=cellBit)
        self.linear = int_linear(512*block.expansion, num_classes, nbit=wbit, mode=mode, k=k, ch_group=ch_group, push=False)


    def _make_layer(self, block, planes, num_blocks, stride, wbit=4, abit=4, alpha_init=10, mode='mean', k=2, col_size=16, ch_group=16, ADCprecision=5, cellBit=2):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, wbit=wbit, abit=abit, alpha_init=alpha_init, mode=mode, k=k, ch_group=ch_group, ADCprecision=ADCprecision, cellBit=cellBit))
            self.in_planes = planes * block.expansion
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

class resnet18_quant_eval:
    base = ResNet
    args = list()
    kwargs = {'block': BasicBlock, 'num_blocks': [2, 2, 2, 2]}


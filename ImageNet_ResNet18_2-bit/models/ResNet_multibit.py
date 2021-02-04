import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch
from collections import OrderedDict
from torch.nn import init
from .quant_modules import QuantizedConv2d, QuantizedLinear

def conv3x3_bl(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv3x3_quan(in_planes, out_planes, stride=1, w_bits=2, a_bits=2):
    return QuantizedConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, w_bits=2, a_bits=2)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, w_bits=2, a_bits=2):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3_quan(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=False)
        
        self.conv2 = conv3x3_quan(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=False)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        # print(f'Conv input activation: {torch.unique(out)}')

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out)
        # print(f'Conv input activation: {torch.unique(out)}')

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, 
                                 padding=0, bias=False)

        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, 
                                 padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1, 
                                 padding=0, bias=False)
        
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu3(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, fp_fl=True, fp_ll=True, w_bits=2, a_bits=2):
        self.inplanes = 64
        super(ResNet, self).__init__()
        if fp_fl:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = QuantizedConv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], w_bits=w_bits, a_bits=a_bits)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, w_bits=w_bits, a_bits=a_bits)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, w_bits=w_bits, a_bits=a_bits)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, w_bits=w_bits, a_bits=a_bits)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = QuantizedLinear(512 * block.expansion, num_classes)
    
        count = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if isinstance(m, QuantizedConv2d):
                    if count == 0:
                        m.input_quant = False
                    count += 1
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, QuantizedLinear):
                m.weight_quant = False

    def _make_layer(self, block, planes, blocks, stride=1, w_bits=2, a_bits=2):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                QuantizedConv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False, w_bits=w_bits, a_bits=a_bits),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, w_bits=w_bits, a_bits=a_bits))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, w_bits=w_bits, a_bits=a_bits))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet18b_quant(num_classes=1000, w_bits=2, a_bits=2):
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, fp_fl=True, fp_ll=True, w_bits=w_bits, a_bits=a_bits)
    return model


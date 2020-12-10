import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from .quant import ClippedReLU, int_conv2d, int_linear

__all__ = ["resnet18_imagenet_quant"]


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv3x3_quant(in_planes, out_planes, stride=1, wbit=4, mode='mean', k=2):
    """3x3 convolution with padding"""
    return int_conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, nbit=wbit, mode=mode, k=k)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, wbit=4, abit=4, alpha_init=10, mode='mean', k=2):
        super(BasicBlock, self).__init__()
        # self.conv1 = conv3x3(inplanes, planes, stride, nbit=wbit, mode=mode, k=k)
        self.conv1 = int_conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, nbit=wbit, mode=mode, k=k)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = ClippedReLU(num_bits=abit, alpha=alpha_init, inplace=True)    # Clipped ReLU function 4 - bits
        
        # self.conv2 = conv3x3(planes, planes, nbit=wbit, mode=mode, k=k)
        self.conv2 = int_conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, nbit=wbit, mode=mode, k=k)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = ClippedReLU(num_bits=abit, alpha=alpha_init, inplace=True)    # Clipped ReLU function 4 - bits
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, wbit=4, abit=4, alpha_init=10, mode='mean', k=2):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = int_conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False, nbit=wbit, mode=mode, k=k)
        self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)
        self.relu0 = ClippedReLU(num_bits=abit, alpha=alpha_init, inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], wbit=wbit, abit=abit, alpha_init=alpha_init, mode=mode, k=k)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, wbit=wbit, abit=abit, alpha_init=alpha_init, mode=mode, k=k)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, wbit=wbit, abit=abit, alpha_init=alpha_init, mode=mode, k=k)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, wbit=wbit, abit=abit, alpha_init=alpha_init, mode=mode, k=k)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc = int_linear(512*block.expansion, num_classes, nbit=wbit, mode=mode, k=k)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, wbit=4, abit=4, alpha_init=10, mode='mean', k=2):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                int_conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False, nbit=wbit, mode=mode, k=k),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, wbit=wbit, abit=abit, alpha_init=alpha_init, mode=mode, k=k))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, wbit=wbit, abit=abit, alpha_init=alpha_init, mode=mode, k=k))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu0(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class resnet18_imagenet_quant:
    base = ResNet
    args = list()
    kwargs = {'block': BasicBlock, 'layers': [2, 2, 2, 2]}
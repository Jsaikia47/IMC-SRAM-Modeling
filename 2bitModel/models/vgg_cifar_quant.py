
import math
import torch.nn as nn
from .quant import ClippedReLU, int_conv2d, int_linear


# __all__ = ['vgg7']


def make_layers(cfg, batch_norm=False):
    layers = list()
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def make_layers_quant(cfg, batch_norm=False, wbit=4, abit=4, alpha_init=10, mode='mean', k=2, ch_group=16, push=False):
    layers = list()
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = int_conv2d(in_channels, v, kernel_size=3, padding=1, bias=False, nbit=wbit, mode=mode, k=k, ch_group=ch_group, push=push)
            # conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), ClippedReLU(num_bits=abit, alpha=alpha_init, inplace=True)]
            else:
                layers += [conv2d, ClippedReLU(num_bits=abit, alpha=alpha_init, inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    7: [128, 128, 'M', 256, 256, 'M', 512, 512, 'M'],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
         512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, num_classes=10, depth=16, batch_norm=False):
        super(VGG, self).__init__()
        self.features = make_layers(cfg[depth], batch_norm)
        if depth == 7:
            self.classifier = nn.Sequential(
                nn.Linear(8192, 1024),
                nn.ReLU(True),
                nn.Linear(1024, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Linear(512, num_classes),
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class VGG_quant(nn.Module):
    def __init__(self, num_classes=10, depth=16, batch_norm=False, wbit=4, abit=4, alpha_init=10, mode='mean', k=2, ch_group=16, push=False):
        super(VGG_quant, self).__init__()
        self.features = make_layers_quant(cfg[depth], batch_norm, wbit=wbit, abit=abit, alpha_init=alpha_init, mode=mode, k=k, ch_group=ch_group, push=push)
        if depth == 7:
            self.classifier = nn.Sequential(
                int_linear(8192, 1024, nbit=wbit, mode=mode, k=k, ch_group=ch_group, push=push),
                # nn.Linear(8192, 1024),
                ClippedReLU(num_bits=abit, alpha=alpha_init, inplace=True),
                int_linear(1024, num_classes, nbit=wbit, mode=mode, k=k, ch_group=ch_group, push=push),
                # nn.Linear(1024, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Linear(512, num_classes),
            )

        for m in self.modules():
            if isinstance(m, int_conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# def vgg7(num_classes=10):
#     model = VGG(num_classes=num_classes, depth=7, batch_norm=True)
#     return model

# def vgg7_quant(num_classes=10):
#     model = VGG_quant(num_classes=num_classes, depth=7, batch_norm=True)
#     return model

class vgg7_quant:
    base = VGG_quant
    args = list()
    kwargs={'depth':7, 'batch_norm':True}
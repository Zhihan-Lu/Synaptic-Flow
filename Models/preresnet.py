from __future__ import absolute_import
import math

from Layers import layers
import torch.nn as nn
import torch.nn.functional as F
from .channel_selection import channel_selection


__all__ = ['resnet']

"""
preactivation resnet with bottleneck design.
"""

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bn1 = layers.BatchNorm2d(inplanes)
        # self.select = channel_selection(inplanes)
        self.conv1 = layers.Conv2d(cfg[0], cfg[1], kernel_size=1, bias=False)
        self.bn2 = layers.BatchNorm2d(cfg[1])
        self.conv2 = layers.Conv2d(cfg[1], cfg[2], kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn3 = layers.BatchNorm2d(cfg[2])
        self.conv3 = layers.Conv2d(cfg[2], planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        # out = self.select(out)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        # out = self.relu(out)
        out = F.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        # out = self.relu(out)
        out = F.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out

class resnet(nn.Module):
    def __init__(self, depth=164, num_classes=100, cfg=None):
        super(resnet, self).__init__()
        assert (depth - 2) % 9 == 0, 'depth should be 9n+2'
        in_channels = 3
        in_size = 32

        n = (depth - 2) // 9
        block = Bottleneck
    # 【16,16,16,64,16,16,64,16,16,64,32,32,128,32,32,128,32,32,128,64,64,256,64,64,256,64,64,256]
    # [5, 6, 5, 20, 5, 4, 20, 4, 8, 20, 6, 5, 20, 2, 5, 20, 6, 3, 20, 6, 2, 20, 5, 4, 20, 6, 2, 20, 3, 6, 20, 5, 8, 20, 10, 8, 39, 7, 9, 39, 11, 7, 39, 7, 10, 39, 9, 11, 39, 10, 9, 39, 5, 9, 39, 8, 9, 39, 13, 16, 39, 11, 10, 39, 12, 11, 39, 12, 11, 77, 13, 21, 77, 18, 22, 77, 18, 23, 77, 18, 23, 77, 23, 19, 77, 22, 30, 77, 21, 12, 77, 21, 17, 77, 24, 16, 77, 26, 13, 77]
        if cfg is None:
            # Construct config variable.
            cfg = [[16, 16, 16], [64, 16, 16]*(n-1), [64, 32, 32], [128, 32, 32]*(n-1), [128, 64, 64], [256, 64, 64]*(n-1), [256]]
            cfg = [item for sub_list in cfg for item in sub_list]

        self.inplanes = 16

        self.conv1 = layers.Conv2d(in_channels, 16, kernel_size=3, padding=1,
                               bias=False)
        self.layer1 = self._make_layer(block, 16, n, cfg = cfg[0:3*n])
        self.layer2 = self._make_layer(block, 32, n, cfg = cfg[3*n:6*n], stride=2)
        self.layer3 = self._make_layer(block, 64, n, cfg = cfg[6*n:9*n], stride=2)
        self.bn1 = layers.BatchNorm2d(64 * block.expansion)
        self.select = channel_selection(64 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = layers.Linear(cfg[-1], num_classes)

        for name,m in self.named_modules():
            if isinstance(m, (layers.Linear, nn.Linear, layers.Conv2d)):
                nn.init.kaiming_normal_(m.weight)        
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, layers.BatchNorm2d):
                nn.init.constant_(m.weight, 0.1)  # choosing a small value here to avoid overflow
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, cfg, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = layers.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False)

        this_layers = []
        this_layers.append(block(self.inplanes, planes, cfg[0:3], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            this_layers.append(block(self.inplanes, planes, cfg[3*i: 3*(i+1)]))

        return nn.Sequential(*this_layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = self.bn1(x)
        # x = self.select(x)
        # x = self.relu(x)
        x = F.relu(x)        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

def preresnet101(input_shape, num_classes, dense_classifier=False, pretrained=False):
    return resnet(depth=101,num_classes=num_classes)
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn


class VGG16(nn.Module):
    def __init__(self, args):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            # 1
            nn.Conv2d(args.num_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 2
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 4
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 5
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 6
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 7
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 8
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 9
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 10
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 11
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 12
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 13
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AvgPool2d(kernel_size=1, stride=1),
        )
        feature_dim = 512 * 1 * 1  # 512, for 32x32 inputs

        self.classifier = nn.Sequential(
            # 14
            nn.Linear(feature_dim, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            # 15
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
        )
        self.fc = nn.Linear(4096, args.num_classes)

    def forward(self, x, start_layer_idx=0, logit=False):
        if start_layer_idx < 0:  #
            return self.mapping(x, start_layer_idx=start_layer_idx, logit=logit)
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        out = self.fc(out)
        result = {}
        result['logit'] = out
        result['output'] = out
        return result

    def mapping(self, z_input, start_layer_idx=-1, logit=True):
        z = z_input
        z = self.fc(z)

        result = {'output': z}
        if logit:
            result['logit'] = z
        return result


def ResNet50(args):
    num_block = [3, 4, 6, 3]
    return ResNet4(args, ImprovedBottleneck, num_block, num_classes=args.num_classes)


class ResNet4(nn.Module):
    def __init__(self, args, block, num_blocks, num_classes=10):
        super(ResNet4, self).__init__()
        # More initial channels
        self.inplanes = 32  # increased from 16 to 32

        # Stronger initial feature extraction
        self.conv1 = nn.Conv2d(args.num_channels, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        # More channels
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)  # 32 channels
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)  # 64 channels
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)  # 128 channels
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)  # 256 channels



        self.bn_final = nn.BatchNorm2d(256 * block.expansion)
        self.relu_final = nn.ReLU(inplace=True)

        # Improved classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Add dropout against overfitting
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(256 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        result = {}

        # Backbone
        x = self.layer1(x)
        result['activation1'] = x

        x = self.layer2(x)
        result['activation2'] = x

        x = self.layer3(x)
        result['activation3'] = x

        x = self.layer4(x)
        result['activation4'] = x

        # Final processing
        x = self.bn_final(x)
        x = self.relu_final(x)

        # Classification head
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        result['representation'] = x
        x = self.dropout(x)
        x = self.fc(x)
        result['output'] = x

        return result


class ImprovedBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ImprovedBottleneck, self).__init__()
        # Bottleneck layer 1: 1x1 convolution (channel reduction)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # Bottleneck layer 2: 3x3 grouped convolution
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=32)
        self.bn2 = nn.BatchNorm2d(planes)

        # Bottleneck layer 3: 1x1 convolution (channel expansion)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)



        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        # 1x1 convolution
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 3x3 grouped convolution
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # 1x1 convolution
        out = self.conv3(out)
        out = self.bn3(out)


        # Residual connection
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

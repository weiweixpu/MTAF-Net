# _*_ coding: utf-8 _*_
# @Author :Menghang Ma
# @Email :mamenghang9@gmail.com
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from backbone.HSP import HSPModule
from network_architecture.research import CA, SELayer3D

class BackBone3D(nn.Module):
    def __init__(self):
        super(BackBone3D, self).__init__()
        net = ResNet3D(ResNetBottleneck, [3, 4, 6, 3], HSPModule, num_classes=2)
        # resnext3d-101 is [3, 4, 23, 3]
        # we use the resnet3d-50 with [3, 4, 6, 3] blocks
        net = list(net.children())
        self.layer0 = net[0]
        # the layer0 contains stem
        self.layer1 = net[1]
        # the layer1 contains the first 3 bottle blocks
        self.layer2 = nn.Sequential(*net[2:4])
        # the layer2 contains the first HSP module the second 4 bottle blocks
        self.layer3 = nn.Sequential(*net[4:6])
        # the layer3 contains the second HSP module the media bottle blocks
        self.layer4 = net[6]
        # the layer4 contains the final 3 bottle blocks
        # according the backbone the next is avg-pooling and dense with num classes uints
        # but we don't use the final two layers in backbone networks

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return layer0, layer1, layer2, layer3, layer4


class ResNet3D(nn.Module):

    def __init__(self, block, layers, Downsample, cardinality=32, num_classes=2):

        super(ResNet3D, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(3, 128, kernel_size=4, stride=4),
            nn.BatchNorm3d(128), nn.PReLU())
        self.layer1 = self._make_layer(block, 64, layers[0],  cardinality)
        self.downlayer1 = self._downsample_layers(Downsample, 128, 16)
        self.layer2 = self._make_layer(block, 128, layers[1],  cardinality)
        self.downlayer2 = self._downsample_layers(Downsample, 256, 16)
        self.layer3 = self._make_layer(block, 256, layers[2],  cardinality)
        self.layer4 = self._make_layer(SENetDilatedBottleneck, 256, layers[3], cardinality)

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, cardinality):
        layers = []
        self.inplanes = planes * block.expansion
        for _ in range(0, blocks):
            layers.append(block(self.inplanes, planes, cardinality))
        return nn.Sequential(*layers)

    def _downsample_layers(self, Downsample, planes, reduction):
        down_sample = Downsample(planes, reduction)
        return nn.Sequential(down_sample)

    def forward(self, x):
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNetBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1,  reduction=16):
        super(ResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(
            planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.PReLU()
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
        out += residual
        out = self.relu(out)

        return out


class SENetDilatedBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1,
                 downsample=None):
        super(SENetDilatedBottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=2,
            dilation=2,
            groups=cardinality,
            bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(
            planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.PReLU()
        self.downsample = downsample
        self.se = SELayer3D(planes * self.expansion, reduction=16)
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
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetDilatedBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1,  reduction=16):
        super(ResNetDilatedBottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=2,
            dilation=2,
            groups=cardinality,
            bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(
            planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.PReLU()
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
        out += residual
        out = self.relu(out)

        return out


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out

def senet3d10(**kwargs):
    """Constructs a SENet3D-10 model."""
    model = ResNet3D(ResNetBottleneck, [1, 1, 1, 1], **kwargs)
    return model


def senet3d18(**kwargs):
    """Constructs a SENet3D-18 model."""
    model = ResNet3D(ResNetBottleneck, [2, 2, 2, 2], **kwargs)
    return model


def senet3d34(**kwargs):
    """Constructs a SENet3D-34 model."""
    model =ResNet3D(ResNetBottleneck, [3, 4, 6, 3], **kwargs)
    return model


def senet3d50(**kwargs):
    """Constructs a SENet3D-50 model."""
    model =ResNet3D(ResNetBottleneck, [3, 4, 6, 3], **kwargs)
    return model


def senet3d101(**kwargs):
    """Constructs a SENet3D-101 model."""
    model = ResNet3D(ResNetBottleneck, [3, 4, 23, 3], **kwargs)
    return model


def senet3d152(**kwargs):
    """Constructs a SENet3D-152 model."""
    model = ResNet3D(ResNetBottleneck, [3, 8, 36, 3], **kwargs)
    return model


def senet3d200(**kwargs):
    """Constructs a SENet3D-200 model."""
    model = ResNet3D(ResNetBottleneck, [3, 24, 36, 3], **kwargs)
    return model
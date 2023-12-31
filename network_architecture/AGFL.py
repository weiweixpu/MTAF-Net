# _*_ coding: utf-8 _*_
# @Author :Menghang Ma
# @Email :mamenghang9@gmail.com
import torch
from torch import nn
from network_architecture.research import CA


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def add_conv3D(in_ch, out_ch, ksize, stride):

    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv3d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('norm', nn.GroupNorm(32,out_ch))
    stage.add_module('relu', nn.GELU())
    return stage


class AGFLblock(nn.Module):
    
    def __init__(self, num_channels, level):
        super(AGFLblock, self).__init__()
        self.level = level
        if self.level != 1:
            self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
            self.w_relu = nn.GELU()
            self.act = Swish()
            self.epsilon = 1e-4

        self.down = add_conv3D(num_channels, num_channels, 3, (2, 2, 2))
        self.cv = nn.Sequential(nn.Conv3d(num_channels, num_channels, kernel_size=1),
                                nn.GroupNorm(32, num_channels),
                                nn.GELU())

        self.branch1 = nn.Sequential(nn.Conv3d(num_channels, num_channels, kernel_size=3,  stride=1, padding=1, dilation=1),
                                     nn.GroupNorm(32, num_channels), nn.GELU())
        self.branch2 = nn.Sequential(nn.Conv3d(num_channels, num_channels, kernel_size=3, stride=1, padding=3, dilation=3),
                                     nn.GroupNorm(32, num_channels), nn.GELU())
        self.branch3 = nn.Sequential(nn.Conv3d(num_channels, num_channels, kernel_size=3, stride=1, padding=5, dilation=5),
                                     nn.GroupNorm(32, num_channels), nn.GELU())
        self.fusion1 = nn.Sequential(nn.Conv3d(num_channels*3, num_channels, kernel_size=1, stride=1),
                                     nn.GroupNorm(32, num_channels))

        self.cv1 = nn.Sequential(nn.Conv3d(num_channels, num_channels, kernel_size=1, stride=1),
                                 nn.GroupNorm(32, num_channels))

        self.relu = nn.GELU()
        self.ca = CA(num_channels, num_channels)

    def forward(self, x, x1):
        if self.level == 1:
            x_1 = self.branch1(x1)
            x_2 = self.branch2(x1)
            x_3 = self.branch3(x1)
            out = self.fusion1(torch.cat((x_1, x_2, x_3), 1))
            feat_ca = self.ca(out)
            x = self.cv1(x)
            out = x + feat_ca
            out = self.relu(out)

        elif self.level == 4:

            w = self.w_relu(self.w)
            weight = w / (torch.sum(w, dim=0) + self.epsilon)
            x1 = self.cv(self.act(weight[0] * x + weight[1] * x1))
            x_1 = self.branch1(x1)
            x_2 = self.branch2(x1)
            x_3 = self.branch3(x1)
            out = self.fusion1(torch.cat((x_1, x_2, x_3), 1))
            feat_ca = self.ca(out)
            x = self.cv1(x)
            out = x + feat_ca
            out = self.relu(out)

        else:
            x1 = self.down(x1)
            w = self.w_relu(self.w)
            weight = w / (torch.sum(w, dim=0) + self.epsilon)
            x1 = self.cv(self.act(weight[0] * x + weight[1] * x1))
            x_1 = self.branch1(x1)
            x_2 = self.branch2(x1)
            x_3 = self.branch3(x1)
            out = self.fusion1(torch.cat((x_1, x_2, x_3), 1))
            feat_ca = self.ca(out)
            x = self.cv1(x)
            out = x + feat_ca
            out = self.relu(out)

        return out
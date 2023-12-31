# _*_ coding: utf-8 _*_
# @Author :Menghang Ma
# @Email :mamenghang9@gmail.com
import torch
import torch.nn as nn
from network_architecture.research import SELayer3D

class HSPModule(nn.Module):

    def __init__(self, planes, reduction=16):
        super(HSPModule, self).__init__()
        # mid_planes = int(planes / 2)
        self.m = nn.MaxPool3d(kernel_size=2, stride=2)
        self.cv1 = nn.Conv3d(planes, planes, kernel_size=1, bias=False)
        self.norm1 = nn.BatchNorm3d(planes)

        self.a = nn.AvgPool3d(kernel_size=2, stride=2)
        self.cv2 = nn.Conv3d(planes, planes, kernel_size=1, bias=False)
        self.norm2 = nn.BatchNorm3d(planes)

        self.cv3 = nn.Conv3d(planes, planes, kernel_size=2, stride=2, bias=False)
        self.norm3 = nn.BatchNorm3d(planes)

        self.cv4 = nn.Conv3d(planes*3, planes*2, kernel_size=1, stride=1, bias=False)
        self.norm4 = nn.BatchNorm3d(planes*2)

        self.relu = nn.PReLU()
        self.se = SELayer3D(planes*2, reduction)

    def forward(self, x):
        max_m = self.cv1(self.m(x))
        max_m = self.norm1(max_m)
        max_m = self.relu(max_m)

        avg = self.cv2(self.a(x))
        avg = self.norm2(avg)
        avg = self.relu(avg)

        conv = self.cv3(x)
        conv = self.norm3(conv)
        conv = self.relu(conv)

        out = torch.cat([max_m, avg, conv], dim=1)
        out = self.cv4(out)
        out = self.norm4(out)
        out = self.se(out)

        return out
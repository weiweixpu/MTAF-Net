# _*_ coding: utf-8 _*_
# @Author :Menghang Ma
# @Email :mamenghang9@gmail.com
import torch
from torch import nn
import torch.nn.functional as F

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    
class SEB(nn.Module):
    def __init__(self, in_channels, out_channels, onnx_export=False):
        super(SEB, self).__init__()

        self.stage = nn.Sequential(nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                   nn.GroupNorm(32, out_channels))

        self.fus = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                 nn.GroupNorm(32, out_channels))

        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w_relu = nn.GELU()
        self.act = Swish()
        self.epsilon = 1e-4
        self.cv = nn.Sequential(nn.Conv3d(out_channels, out_channels, kernel_size=1),
                                nn.GroupNorm(32, out_channels),
                                nn.GELU()
                                )

    def forward(self, x1, x2):
        x1 = self.stage(x1)
        x2 = self.fus(x2)
        x2 = F.interpolate(x2, size=x1.size()[2:], mode='trilinear', align_corners=True)
        w = self.w_relu(self.w)
        weight = w / (torch.sum(w, dim=0) + self.epsilon)

        out = self.cv(self.act(weight[0] * x1 + weight[1] * x2))
        return out
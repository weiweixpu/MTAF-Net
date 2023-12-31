# _*_ coding: utf-8 _*_
# @Author :Menghang Ma
# @Email :mamenghang9@gmail.com
import torch
from torch import nn
import torch.nn.functional as F
from backbone.BackBone import BackBone3D
from network_architecture.WSEB import SEB
from network_architecture.AGFL import AGFLblock
from utils.tools import model_info

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result


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
    stage.add_module('norm', nn.GroupNorm(32, out_ch))
    stage.add_module('relu', nn.GELU())

    return stage


class TaskAttention(nn.Module):
    def __init__(self,
                 in_channels,
                 channel_in,
                 out_channels,):
        super(TaskAttention, self).__init__()

        self.gap = nn.AdaptiveAvgPool3d(1)

        self.stage_encoder = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.GELU())

        self.fus_encoder = nn.Sequential(
            nn.Conv3d(channel_in, out_channels, 1),
            nn.GroupNorm(32, out_channels),
            nn.GELU()
        )

        self.fus_reencoder = nn.Sequential(
            nn.Conv3d(channel_in, out_channels, 1),
            nn.GroupNorm(32, out_channels),
            nn.GELU()
        )

        self.normalizer = nn.Sigmoid()

    def forward(self, stage_feature, fus_features):
        b, c, _, _, _ = stage_feature.size()

        stage_feats = self.gap(stage_feature).view(b, c)
        stage_feats = self.stage_encoder(stage_feats).view(b, c, 1, 1, 1)

        fus_feat = self.fus_encoder(fus_features)
        relations = self.normalizer((fus_feat * stage_feats).sum(dim=1, keepdim=True))

        p_feats = self.fus_reencoder(fus_features)

        refined_feats = relations * p_feats

        return refined_feats


class MTAF3D(nn.Module):
    def __init__(self, vis=False):
        super(MTAF3D, self).__init__()

        self.backbone = BackBone3D()

        self.conv_l1_down_channel = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=1),
            nn.GroupNorm(32, 64), nn.GELU()
        )
        self.conv_l2_down_channel = nn.Sequential(
            nn.Conv3d(256, 64, kernel_size=1),
            nn.GroupNorm(32, 64), nn.GELU()
        )
        self.conv_l3_down_channel = nn.Sequential(
            nn.Conv3d(512, 64, kernel_size=1),
            nn.GroupNorm(32, 64), nn.GELU()
        )
        self.conv_l4_down_channel = nn.Sequential(
            nn.Conv3d(512, 64, kernel_size=1),
            nn.GroupNorm(32, 64), nn.GELU()
        )

        self.seb3 = SEB(64, 64)
        self.seb2 = SEB(128, 64)
        self.seb1 = SEB(192, 64)

        self.AGFL1 = AGFLblock(64, level=1)
        self.AGFL2 = AGFLblock(64, level=2)
        self.AGFL3 = AGFLblock(64, level=3)
        self.AGFL4 = AGFLblock(64, level=4)

        ### segmentation branch
        self.attention0 = TaskAttention(64, 192, 64)
        self.conv0 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.GELU()
        )

        self.attention1 = TaskAttention(64, 128, 64)
        self.conv1 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.GELU()
        )

        self.predict = nn.Conv3d(64, 1, kernel_size=1)

        ### classification branch
        self.pool0 = add_conv3D(64, 64, 3, 2)
        self.attention2 = TaskAttention(64, 128, 64)
        self.conv2 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.GELU()
        )

        self.pool1 = add_conv3D(64, 64, 3, 2)
        self.attention3 = TaskAttention(64, 128, 64)
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.GELU()
        )

        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(128, 64)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x0 = self.backbone.layer0(x)    # [8, 64, 64, 64, 64]
        x1 = self.backbone.layer1(x0)  # [8, 128, 32, 32, 32]
        x2 = self.backbone.layer2(x1)  # [8, 256, 16, 16, 16]
        x3 = self.backbone.layer3(x2)   # [8, 512, 8, 8, 8]
        x4 = self.backbone.layer4(x3)   # [8, 512, 8, 8, 8]

        x1 = self.conv_l1_down_channel(x1)
        x2 = self.conv_l2_down_channel(x2)
        x3 = self.conv_l3_down_channel(x3)
        x4 = self.conv_l4_down_channel(x4)

        s3 = self.seb3(x3, x4)

        x4_3 = F.interpolate(x4, size=x3.size()[2:], mode='trilinear', align_corners=True)
        s2 = self.seb2(x2, torch.cat((x3, x4_3), dim=1))

        x4_2 = F.interpolate(x4, size=x2.size()[2:], mode='trilinear', align_corners=True)
        x3_2 = F.interpolate(x3, size=x2.size()[2:], mode='trilinear', align_corners=True)

        s1 = self.seb1(x1, torch.cat((x2, x4_2, x3_2), dim=1))

        Scale1A = self.AGFL1(s1, s1)
        Scale2A = self.AGFL2(s2, Scale1A)
        Scale3A = self.AGFL3(s3, Scale2A)
        Scale4A = self.AGFL4(x4, Scale3A)

        ### segmentation branch
        out_F3_0 = torch.cat((Scale4A, Scale3A), 1)
        out_F3_1 = F.interpolate(out_F3_0, size=Scale2A.size()[2:], mode='trilinear', align_corners=True)

        out_F2_0 = torch.cat((out_F3_1, Scale2A), 1)
        out_F2_1 = self.conv0(self.attention0(Scale2A, out_F2_0))
        out_F2_2 = F.interpolate(out_F2_1, size=Scale1A.size()[2:], mode='trilinear', align_corners=True)

        out_F1_0 = torch.cat((out_F2_2, Scale1A), 1)
        out_F1_1 = self.conv1(self.attention1(Scale1A, out_F1_0))

        out_F1_2 = F.interpolate(out_F1_1, size=x.size()[2:], mode='trilinear', align_corners=True)

        ### classificication branch
        out_F10 = self.pool0(Scale1A)

        out_F20 = torch.cat((out_F10, Scale2A), 1)
        out_F21 = self.conv2(self.attention2(Scale2A, out_F20))
        out_F22 = self.pool1(out_F21)

        out_F30 = torch.cat((out_F22, Scale3A), 1)
        out_F31 = self.conv3(self.attention3(Scale3A, out_F30))

        out_F40 = torch.cat((out_F31, Scale4A), 1)


        seg_predict = self.predict(out_F1_2)
        class_predict1 = self.gap(out_F40)
        class_predict1 = class_predict1.view(class_predict1.size(0), -1)

        class_predict = self.fc(self.fc1(class_predict1))
        return seg_predict, class_predict


if __name__ == '__main__':
    a = MTAF3D()
    print(list(a.children()))
    model_info(a, verbose=True, img_size=128)
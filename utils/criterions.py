# _*_ coding: utf-8 _*_
# @Author :Menghang Ma
# @Email :mamenghang9@gmail.com
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from utils.miou import SegmentationMetric

class AutomaticWeightedLoss1(nn.Module):
    """automatically weighted multi-task loss
    """
    def __init__(self, task_num, loss_fn):
        super(AutomaticWeightedLoss1, self).__init__()
        self.task_num = task_num
        self.loss_fn = loss_fn
        params = torch.ones(self.task_num, requires_grad=True)
        self.params = nn.Parameter(params)

    def forward(self, outputs, targets, weights):
        std_1 = torch.log(1+self.params[0]**2)
        std_2 = torch.log(1+self.params[1]**2)

        seg_loss, dice, miou = self.loss_fn[0](outputs[0], targets[0], weights[0])

        seg_loss_1 = torch.sum(0.5 / (self.params[0]**2) * seg_loss + std_1, -1)

        idh_loss = self.loss_fn[1](outputs[1], targets[1], weights[1])

        idh_loss_1 = torch.sum(0.5 / (self.params[1]**2) * idh_loss + std_2, -1)

        loss = seg_loss_1 + idh_loss_1

        return loss, seg_loss, idh_loss, dice, miou, std_1, std_2

class AutomaticWeightedLoss2(nn.Module):
    def __init__(self, task_num, loss_fn):
        super(AutomaticWeightedLoss2, self).__init__()
        self.task_num = task_num
        self.loss_fn = loss_fn
        params = torch.tensor((0.0,0.0), requires_grad=True)
        self.log_vars = nn.Parameter(params) #1.0, 6.0

    def forward(self, outputs,targets,weights):
        std_1 = torch.exp(self.log_vars[0]) ** 0.5
        std_2 = torch.exp(self.log_vars[1]) ** 0.5

        seg_loss, dice, miou = self.loss_fn[0](outputs[0], targets[0], weights[0])

        seg_loss_1 = torch.sum(0.5 * torch.exp(-self.log_vars[0]) * seg_loss + self.log_vars[0],-1) #

        idh_loss = self.loss_fn[1](outputs[1], targets[1], weights[1])

        idh_loss_1 = torch.sum(0.5 * torch.exp(-self.log_vars[1]) * idh_loss + self.log_vars[1],-1)

        loss = torch.mean(seg_loss_1+idh_loss_1)

        return loss, seg_loss, idh_loss, dice, miou, std_1, std_2
    

class AutomaticWeightedLoss3(nn.Module):
    def __init__(self, task_num, loss_fn):
        super(AutomaticWeightedLoss3, self).__init__()
        self.task_num = task_num
        self.loss_fn = loss_fn

    def forward(self, outputs, targets, weights):
        seg_loss, dice, miou = self.loss_fn[0](outputs[0], targets[0], weights[0])
        idh_loss = self.loss_fn[1](outputs[1], targets[1], weights[1])
        loss = idh_loss**2 / (idh_loss+seg_loss) + seg_loss**2 / (idh_loss+seg_loss)
        std_1 = seg_loss / (idh_loss+seg_loss)
        std_2 = idh_loss / (idh_loss+seg_loss)
        return loss, seg_loss, idh_loss, dice, miou, std_1, std_2
    

class AutomaticWeightedLoss4(nn.Module):
    """equal weighted multi-task loss
    """
    def __init__(self, task_num, loss_fn):
        super(AutomaticWeightedLoss4, self).__init__()
        self.task_num = task_num
        self.loss_fn = loss_fn

    def forward(self, outputs, targets, weights):
        seg_loss, dice, miou = self.loss_fn[0](outputs[0], targets[0], weights[0])
        idh_loss = self.loss_fn[1](outputs[1], targets[1], weights[1])
        loss = 0.5 * seg_loss + 0.5 * idh_loss

        std_1 = 0.5
        std_2 = 0.5

        return loss, seg_loss, idh_loss, dice, miou, std_1, std_2
    

def ce_loss(input, target, weight):
    loss = nn.CrossEntropyLoss(weight, reduction='mean')
    return loss(input, target)

def structure_loss(y_pred, y_true, weights):
    dice_loss = HybridLoss(weights, smooth=1e-5)
    l2, dice, miou = dice_loss(y_pred, y_true)
    return l2, dice, miou


class Single_dice(_Loss):
    '''
    Soft_Dice = 2*|dot(A, B)| / (|dot(A, A)| + |dot(B, B)| + eps)
    eps is a small constant to avoid zero division,
    '''
    def __init__(self, weight=None):
        super(Single_dice, self).__init__()
        self.weight = weight

    def forward(self, y_pred, y_true, eps): 
        y_pred = y_pred[:, 1:, ...]
        intersection = torch.sum(torch.mul(y_pred, y_true), dim=(-3, -2, -1)) + eps / 2
        union = torch.sum(y_pred, dim=(-3, -2, -1)) + torch.sum(y_true, dim=(-3, -2, -1)) + eps
        dice = 2 * intersection / union

        return 1-dice.mean(dim=0), dice.mean(dim=0)


def MiouMetrics(imgPredict, imgLabel):
    imgPredict = imgPredict[:, 1:, ...]
    metric = SegmentationMetric(2)
    metric.addBatch(imgPredict, imgLabel)
    mIoU = metric.meanIntersectionOverUnion()
    return mIoU


class HybridLoss(nn.Module):
    def __init__(self, weight=None, smooth=1e-5):

        "Segmentation Hybrid loss:BCE+IOU "

        super(HybridLoss, self).__init__()
        self.weight = weight
        self.smooth = smooth

    def forward(self, input, target):

        weit = 1 + 5 * torch.abs(F.avg_pool3d(target, kernel_size=31, stride=1, padding=15) - target)
        wbce = F.binary_cross_entropy_with_logits(input, target, reduce=None)
        wbce = (weit * wbce).sum(dim=(2, 3, 4)) / weit.sum(dim=(2, 3, 4))

        pred = torch.sigmoid(input)
        inter = ((pred * target) * weit).sum(dim=(2, 3, 4))
        union = ((pred + target) * weit).sum(dim=(2, 3, 4))
        wiou = 1 - (inter + 1) / (union - inter + 1)

        inter = (pred * target).sum(dim=(2, 3, 4)) + self.smooth / 2  
        union = (pred + target).sum(dim=(2, 3, 4)) + self.smooth
  
        dice = 2 * inter / union      # dice
        iou = inter / (union - inter)        # iou

        total_loss = (wbce+wiou).mean()

        return total_loss, dice.mean(dim=0), iou.mean(dim=0)

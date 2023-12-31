# _*_ coding: utf-8 _*_
# @Author :马梦航
# @Email :mamenghang9@gmail.com
"""
refer to https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/metrics.py
"""
import torch
import torch.nn.functional as F

__all__ = ['SegmentationMetric']

"""
confusionMetric  # Note: Here, horizontal represents predicted values, and vertical represents true values
P\L     P    N
P      TP    FP
N      FN    TN
"""


class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = torch.zeros((self.numClass,) * 2).cuda()

    def meanIntersectionOverUnion(self):

        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = torch.diag(self.confusionMatrix)  
        union = torch.sum(self.confusionMatrix, axis=1) + torch.sum(self.confusionMatrix, axis=0) - torch.diag(
            self.confusionMatrix)  
        IoU = intersection / union + 1e-7  
        # print('IOU', IoU)
        mIoU = torch.mean(IoU)
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        label = torch.round(label)
        count = torch.bincount(label.long(), minlength=self.numClass ** 2)
        confusionMatrix = count.view(self.numClass, self.numClass)
        return confusionMatrix

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = torch.zeros((self.numClass, self.numClass))


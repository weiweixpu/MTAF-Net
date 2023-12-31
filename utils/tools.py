# _*_ coding: utf-8 _*_
# @Author :Menghang Ma
# @Email :mamenghang9@gmail.com
import numpy as np
import torch
import os
from utils.logger import log
from copy import deepcopy

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path, patience=10, verbose=False, delta=0):
        """
        Args:
            save_path : Model Save Folder
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.val_loss_min = np.Inf
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss, epoch, model, optimizer):

        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, epoch, model, optimizer)
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            log.info('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, epoch, model, optimizer)
            self.counter = 0

    def save_checkpoint(self, val_loss, epoch, model, optimizer):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            log.info('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'
                     .format(self.val_loss_min, val_loss))
        path = os.path.join(self.save_path, 'best_network.pth.tar')
        torch.save({
            'epoch': epoch,
            'en_state_dict': model['en'].state_dict(),
            'mtl_state_dict': model['mtl'].state_dict(),
            'optimizer': optimizer.state_dict()},
            path)
        self.val_loss_min = val_loss


class MultiEarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path, patience=10, verbose=False, delta=0):
        """
        Args:
            save_path : Model Save Folder
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.max_score = np.Inf
        self.early_stop = False
        self.delta = delta

    def __call__(self, score, epoch, model, optimizer):

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, epoch, model, optimizer)
        elif score < self.best_score + self.delta:
            self.counter += 1
            log.info('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, epoch, model, optimizer)
            self.counter = 0

    def save_checkpoint(self, score, epoch, model, optimizer):
        '''Saves model when validation score crease.'''
        if self.verbose:
            log.info('Validation score creased ({:.6f} --> {:.6f}).  Saving model ...'
                     .format(self.max_score, score))
        path = os.path.join(self.save_path, 'best_network.pth.tar')
        torch.save({
            'epoch': epoch,
            'en_state_dict': model['en'].state_dict(),
            'mtl_state_dict': model['mtl'].state_dict(),
            'optimizer': optimizer.state_dict()},
            path)
        self.max_score = score


def model_info(model, verbose=False, img_size=128):
    # Model information. img_size may be int or list, i.e. img_size=128 or img_size=[128, 128]
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print(f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}")
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    try:  # FLOPs
        from thop import profile
        stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32
        img = torch.zeros((1, 3, stride, stride, stride), device=next(model.parameters()).device)  # input
        flops = profile(deepcopy(model), inputs=(img,), verbose=False)[0] / 1E9 * 2  # stride GFLOPs
        img_size = img_size if isinstance(img_size, list) else [img_size, img_size, img_size]  # expand if int/float
        fs = ', %.1f GFLOPs' % (flops * img_size[0] / stride * img_size[1] / stride * img_size[2] / stride)  # 640x640 GFLOPs
    except (ImportError, Exception):
        fs = ''

    log.info(f"Model Summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}")



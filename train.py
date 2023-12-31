# _*_ coding: utf-8 _*_
# @Author :Menghang Ma
# @Email :mamenghang9@gmail.com
import os
import math
import random
import time
import argparse
import numpy as np
import torch
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from torch.cuda import amp
from torch.utils.data import DataLoader
import torch.multiprocessing
from utils.logger import log, log_args
from utils.tools import EarlyStopping
from data.BraTS_IDH import PklIDH
from utils import criterions
from utils.criterions import AutomaticWeightedLoss1, AutomaticWeightedLoss2, AutomaticWeightedLoss3, AutomaticWeightedLoss4
from network_architecture.MTAF import MTAF3D
from sklearn.metrics import roc_auc_score, accuracy_score

torch.multiprocessing.set_sharing_strategy('file_system')
local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('--train_root', default='/data/mmhang/multitask/train_multitask', type=str, help='Root directory')

parser.add_argument('--valid_root', default='/data/mmhang/multitask/valid_multitask', type=str, help='Root directory')

parser.add_argument('--experiment', default='MTAF', type=str, help='experiment name')

parser.add_argument('--date', default=local_time.split(' ')[0], type=str, help='Today date')

parser.add_argument('--Bsize', default=8, type=int, help='Batch Size')

parser.add_argument('--learning_rate', default=0.0001, type=float, help='initial learning rate')

parser.add_argument('--lrf', default=0.000001, type=float, help='final OneCycleLR learning rate (learning_rate * lrf)')

parser.add_argument('--momentum', default=0.9, type=float, help='The values of momentum')

parser.add_argument('--warmup_momentum', default=0.8, type=float, help='The value of momentum for initial training')

parser.add_argument('--weight_decay', default=5e-3, type=float, help='optimizer weight decay')

parser.add_argument('--amsgrad', default=True, type=str2bool,help='Whether to use amsgrad')

parser.add_argument('--seg_criterion', default='structure_loss', type=str,help='Segmentation loss')

parser.add_argument('--class_criterion', default='ce_loss', type=str, help='IDH genotyping loss')

parser.add_argument('--start_epoch', default=0, type=int, help='initial epoch')

parser.add_argument('--end_epochs', default=300, type=int, help='Number of total epochs to run')

parser.add_argument('--load', default=False, type=str2bool,help='Whether to load resume model')

parser.add_argument('--resume_path', default='', type=str, help='Path for resume model')

parser.add_argument('--save_intervals', default=50, type=int, help='Interation for saving model')

parser.add_argument('--nosave', default=False, action='store_true', help='only save final checkpoint')

parser.add_argument('--patience', type=int, default=50, help='EarlyStopping')

parser.add_argument('--use_early', default=True, type=str2bool,  help='Whether to use early stop')

parser.add_argument('--seed', default=42, type=int, help='Manually set random seed')

parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

parser.add_argument('--num_workers', default=8, type=int, help='Number of jobs')

parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")

args = parser.parse_args()
args.amp = not args.noamp

args.save_folder = "./trails/{}/{}"\
    .format(args.experiment,str(local_time.split(' ')[0]))


def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2, cos0->lrf
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def train():

    log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'log', args.experiment + args.date)
    log_file = log_dir + '.txt'
    log_args(log_file)

    log.info('--------------------------------------This is all argsurations----------------------------------')
    for arg in vars(args):
        log.info('{}={}'.format(arg, getattr(args, arg)))
    log.info('----------------------------------------This is a halving line----------------------------------')
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model = MTAF3D()
    seg_criterion = getattr(criterions, args.seg_criterion)
    idh_criterion = getattr(criterions, args.class_criterion)
    MTL = AutomaticWeightedLoss1(2, loss_fn=[seg_criterion, idh_criterion])
    nets = {
        'en': model.cuda(),
        'mtl': MTL.cuda()
    }

    # model parameters
    param = [p for v in nets.values() for p in list(v.parameters())]
    n_p = sum([p.numel() for v in nets.values() for p in list(v.parameters())])
    n_g = sum([p.numel() for v in nets.values() for p in list(v.parameters()) if p.requires_grad])
    log.info(f"Model Summary:  {n_p} parameters, {n_g} gradients")

    params = [{'params': param, 'lr': args.learning_rate, 'betas': (args.momentum, 0.999),
              'weight_decay': args.weight_decay, 'amsgrad': args.amsgrad}]
    optimizer = torch.optim.Adam(params)

    # learning rate decay
    lf = one_cycle(1, args.lrf, args.end_epochs)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    train_dataset = PklIDH(args.train_root, phase='train')
    train_data_loader = DataLoader(train_dataset, batch_size=args.Bsize, shuffle=True,
                                   num_workers=args.num_workers, pin_memory=True)
    test_dataset = PklIDH(args.valid_root, phase='valid')
    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True)

    batches_train_epoch, batches_test_epoch = len(train_data_loader), len(test_data_loader)
    log.info('{} epochs in total, {} train batches per epoch, {} test batches per epoch'.\
             format(args.end_epochs, batches_train_epoch, batches_test_epoch))

    multi_model = {
        'en': nets['en'],
        'mtl': nets['mtl'],
    }

    # If training is interrupted, continue training 
    resume = args.resume_path
    if os.path.isfile(resume) and args.load:
        log.info('loading checkpoint {}'.format(resume))
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        multi_model['en'].load_state_dict(checkpoint['en_state_dict'])
        multi_model['mtl'].load_state_dict(checkpoint['mtl_state_dict'])
        optimizer.param_groups['lr'].load_state_dict(checkpoint['optimizer'])
        log.info('Successfully loading checkpoint {} and training from epoch: {}'
                     .format(resume, args.start_epoch))
    else:
        log.info('re-training!!!')

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    early_stopping = EarlyStopping(save_path=args.save_folder, patience=args.patience, verbose=True)
    
    # warmup epochs
    nw = max(round(3 * batches_train_epoch), 200)

    # Automatic Mixed Precision Training
    if args.amp:
        scaler = amp.GradScaler(enabled=args.amp)
    else:
        scaler = None

    start_time = time.time()
    for epoch in range(args.start_epoch, args.end_epochs):
        print('-' * 100)
        log.info('Start epoch {}, lr = {}'.format(epoch+1, optimizer.param_groups[0]['lr']))
        epoch_train_loss, epoch_train_dice, epoch_train_miou = 0, 0, 0
        train_id = []
        train_label, train_predicted, train_forecast = [], [], []
        start_epoch = time.time()

        multi_model['en'].train()
        multi_model['mtl'].train()
        for batch_id, batch_data_train in enumerate(train_data_loader):
            batch_id_sp = epoch * batches_train_epoch
            ni = batch_id + epoch * batches_train_epoch  # number integrated batches (since train start)
            optimizer.zero_grad()
            # Warm-up phase for learning rate
            if ni <= nw:
                xi = [0, nw]  # x interp
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [args.warmup_momentum, args.momentum])

            image_train, mask_train, labels_train, patient_ID_train = batch_data_train.values()

            image_train, mask_train = image_train.cuda(), mask_train.cuda()
            labels_train = labels_train.cuda()

            with amp.autocast(enabled=args.amp):
                seg_train, class_ptrain = multi_model['en'](image_train)
                loss, seg_loss, idh_loss, train_dice, train_miou, seg_std, idh_std = \
                    multi_model['mtl']([seg_train, class_ptrain], [mask_train, labels_train], [None, None])

            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            log.info('Epoch: {}_Iter:{}  train_loss: {:.5f} seg_loss: {:.5f} idh_loss: {:.5f} || '
                     '{:.4f} | {:.4f} || seg_std:{:.4f} idh_std:{:.4f}'.
                     format(epoch + 1, batch_id + 1, loss.item(), seg_loss.item(), idh_loss.item(),
                     train_dice.item(), train_miou.item(), seg_std.item(), idh_std.item()))

            # save intervals model
            if batch_id == 0 and batch_id_sp != 0 and epoch % args.save_intervals == 0:
                model_save_file = 'epoch_{}_batch_{}_intervals.pth.tar'.format(epoch, batch_id)
                model_save_path = os.path.join(args.save_folder, model_save_file)
                model_save_dir = os.path.dirname(model_save_path)
                if not os.path.exists(model_save_dir):
                    os.makedirs(model_save_dir)
                log.info('Save checkpoints: epoch = {}, batch_id = {}'.format(epoch, batch_id))
                torch.save({
                    'epoch': epoch,
                    'en_state_dict': multi_model['en'].state_dict(),
                    'mtl_state_dict': multi_model['mtl'].state_dict(),
                    'optimizer': optimizer.state_dict()},
                    model_save_path)

            predicted_train = F.softmax(class_ptrain, 1)
            train_IDH = torch.argmax(predicted_train, dim=1)
            epoch_train_loss += loss.item()
            epoch_train_dice += train_dice.item()
            epoch_train_miou += train_miou.item()
            train_probability = np.array(predicted_train[:, -1].detach().cpu())
            label_train = np.array(labels_train.cpu())

            train_id.extend(patient_ID_train)
            train_label.extend(label_train)
            train_predicted.extend(train_probability)
            train_forecast.extend(np.array(train_IDH.detach().cpu()))

        scheduler.step()

        avg_train_loss = epoch_train_loss / batches_train_epoch
        avg_train_dice = epoch_train_dice / batches_train_epoch
        avg_train_miou = epoch_train_miou / batches_train_epoch
        avg_train_auc = roc_auc_score(train_label, train_predicted)
        epoch_train_loss, epoch_train_dice = 0, 0
        avg_train_acc = accuracy_score(train_label, train_forecast)

        with torch.no_grad():

            multi_model['en'].eval()
            multi_model['mtl'].eval()

            epoch_test_loss, epoch_test_dice, epoch_test_miou = 0, 0, 0

            test_id = []
            test_label, test_predicted, test_forecast = [], [], []
            for i_batch_test, batch_data_test in enumerate(test_data_loader):
                image_test, mask_test, labels_test, patient_ID_test = batch_data_test.values()

                image_test, mask_test = image_test.cuda(), mask_test.cuda()
                labels_test = labels_test.cuda()

                with amp.autocast(enabled=args.amp):
                    seg_test, class_test = multi_model['en'](image_test)

                    test_loss, seg_loss, idh_loss, test_dice, test_miou, seg_std, idh_std = \
                        multi_model['mtl']([seg_test, class_test], [mask_test, labels_test], [None, None])

                log.info('Epoch: {}_Iter:{}  test_loss: {:.5f} seg_loss: {:.5f} idh_loss: {:.5f} || '
                         '{:.4f} | {:.4f} || seg_std:{:.4f} idh_std:{:.4f}'.
                         format(epoch + 1, i_batch_test + 1, test_loss.item(), seg_loss.item(), idh_loss.item(),
                                test_dice.item(), test_miou.item(), seg_std.item(), idh_std.item()))
                predicted_test = F.softmax(class_test, 1)
                test_IDH = torch.argmax(predicted_test, dim=1)
                epoch_test_loss += test_loss.item()
                epoch_test_dice += test_dice.item()
                epoch_test_miou += test_miou.item()

                test_probability = np.array(predicted_test[:, -1].detach().cpu())
                label_test = np.array(labels_test.cpu())

                test_id.extend(patient_ID_test)
                test_label.extend(label_test)
                test_predicted.extend(test_probability)
                test_forecast.extend(np.array(test_IDH.detach().cpu()))

        end_epoch = time.time()
        epoch_time_minute = (end_epoch - start_epoch) / 60
        remaining_time_hour = (args.end_epochs - epoch - 1) * epoch_time_minute / 60

        avg_test_loss = epoch_test_loss / batches_test_epoch
        avg_test_dice = epoch_test_dice / batches_test_epoch
        avg_test_miou = epoch_test_miou / batches_test_epoch
        avg_test_auc = roc_auc_score(test_label, test_predicted)
        avg_test_acc = accuracy_score(test_label, test_forecast)

        log.info(
            'Epoch: {}-{}, train_loss = {:.3f}, train_dice = {:.3f}, train_miou = {:.3f},'
            'train_AUC = {:.3f},train_ACC = {:.3f}, epoch_time = {:.2f} minutes'
            .format(epoch+1, args.end_epochs, avg_train_loss, avg_train_dice, avg_train_miou,
                    avg_train_auc, avg_train_acc, epoch_time_minute))

        log.info(
            'Epoch: {}-{}, test_loss = {:.3f}, test_dice = {:.3f}, test_miou = {:.3f},'
            'test_AUC = {:.3f}, test_ACC = {:.3f}, epoch_time = {:.2f} minutes'.
            format(epoch+1, args.end_epochs, avg_test_loss, avg_test_dice, avg_test_miou,
                   avg_test_auc, avg_test_acc, epoch_time_minute))

        log.info('Estimated remaining training time: {:.2f} hours!'.format(remaining_time_hour))

        # save last model
        final_epoch = epoch + 1 == args.end_epochs
        if (not args.nosave) or final_epoch:  # if save
            model_last_file = os.path.join(args.save_folder, 'last_network.pth.tar')
            torch.save({
                'epoch': epoch,
                'en_state_dict': multi_model['en'].state_dict(),
                'mtl_state_dict': multi_model['mtl'].state_dict(),
                'optimizer': optimizer.state_dict()},
                model_last_file)

        # save best model
        if args.use_early:
            early_stopping(avg_test_loss, epoch, multi_model, optimizer)
            if early_stopping.early_stop:
                log.info('Early stopping And Save checkpoints: epoch = {}'.format(epoch))
                break

    end_time = time.time()
    total_time = (end_time-start_time)/3600
    log.info('The total training time is {:.2f} hours'.format(total_time))
    log.info('----------------------------------The training process finished!-----------------------------------')


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    train()

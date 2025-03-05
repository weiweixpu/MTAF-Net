# _*_ coding: utf-8 _*_
# @Author :Menghang Ma
# @Email :mamenghang9@gmail.com
import argparse
import os
import time
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torch.optim
from torch.utils.data import DataLoader
from data.BraTS_IDH import PklIDH
from predict import validate_softmax
from network_architecture.MTAF import MTAF3D

parser = argparse.ArgumentParser()

parser.add_argument('--test_root', default=r'/data/mmhang/multitask/test_multitask', type=str, help='Root directory')

parser.add_argument('--experiment', default='MTAF', type=str, help='experiment name')   

parser.add_argument('--test_date', default='2023-08-20', type=str, help='experiment data')

parser.add_argument('--test_file', default='best_network.pth.tar', type=str, help='model directory name')

parser.add_argument('--use_TTA', default=True, type=bool, help='Whether to use augmentation during testing')

parser.add_argument('--post_process', default=True, type=bool, help='Whether to use post-processing')

parser.add_argument('--save_csv', default='test.csv', choices=['train.csv', 'valid.csv', 'test.csv'], type=str, help='evaluation metric saving file name')

parser.add_argument('--output_dir', default='', type=str, help='segmentation results saving file name')

parser.add_argument('--submission', default='submission', type=str, help='the path to save nii')

parser.add_argument('--visual', default='visualization', type=str, help='the path to save visualization')

parser.add_argument('--save_format', default='nii', choices=['npy', 'nii'], type=str,help='segmentation result storage file type')

parser.add_argument('--seed', default=42, type=int, help='Manually set random seed')

parser.add_argument('--device', default='0', type=str, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

parser.add_argument('--num_workers', default=8, type=int, help='Number of jobs')

args = parser.parse_args()


def main():

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    model = MTAF3D().cuda()
    dict_model = {'en': model}
    load_file = os.path.join("./trails", args.test_file)

    if os.path.exists(load_file):
        checkpoint = torch.load(load_file)
        dict_model['en'].load_state_dict(checkpoint['en_state_dict'])
        print('Successfully load checkpoint {}'.format(load_file))
    else:
        print('There is no resume file to load!')


    valid_set = PklIDH(args.test_root, phase='test')
    print('Samples for valid = {}'.format(len(valid_set)))

    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    submission = os.path.join('/data/', args.output_dir,
                               args.experiment+args.test_date, args.submission)
    visual = os.path.join('/data/', args.output_dir,
                           args.experiment+args.test_date, args.visual)

    if not os.path.exists(submission):
        os.makedirs(submission)
    if not os.path.exists(visual):
        os.makedirs(visual)

    start_time = time.time()
    with torch.no_grad():
        validate_softmax(valid_loader=valid_loader,
                         model=dict_model,
                         savepath=submission,
                         use_TTA=args.use_TTA,
                         save_format=args.save_format,
                         snapshot=True,
                         visual=visual,
                         postprocess=args.post_process,
                         save_csv=args.save_csv)

    end_time = time.time()
    full_test_time = (end_time-start_time)/60
    average_time = full_test_time/len(valid_set)
    print('{:.2f} minutes!'.format(average_time))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    assert torch.cuda.is_available() # Currently, we only support CUDA version
    main()



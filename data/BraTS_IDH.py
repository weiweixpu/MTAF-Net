# _*_ coding: utf-8 _*_
# @Author :Menghang Ma
# @Email :mamenghang9@gmail.com
import os
import random
import math
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from scipy import ndimage
from torch.utils.data.sampler import Sampler
import torch.distributed as dist

def pkload(fname):
    # load PKL file
    with open(fname, 'rb') as f:
        return pickle.load(f)
    
class DistributedWeightedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, replacement=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.replacement = replacement
        self.shuffle = True

    def calculate_weights(self, targets):
        class_sample_count = torch.tensor(
            [(targets == t).sum() for t in torch.unique(targets, sorted=True)])
        weight = 1. / class_sample_count.double()
        samples_weight = torch.tensor([weight[t] for t in targets])
        return samples_weight

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        # get targets (you can alternatively pass them in __init__, if this op is expensive)
        targets = self.dataset.targets
        targets = targets[self.rank:self.total_size:self.num_replicas]
        assert len(targets) == self.num_samples
        weights = self.calculate_weights(targets)
        print("weights:",weights)

        return iter(torch.multinomial(weights, self.num_samples, self.replacement).tollist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def cc2weight(cc, w_min: float = 1., w_max: float = 2e5):
    weight = torch.zeros_like(cc, dtype=torch.float32)
    cc_items = torch.unique(cc)
    K = len(cc_items) - 1
    N = torch.prod(torch.tensor(cc.shape))
    for i in cc_items:
        weight[cc == i] = N / ((K + 1) * torch.sum(cc == i))
    return torch.clip(weight, w_min, w_max)


class Random_Flip(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        if random.random() < 0.5:
            image = np.flip(image, 0)
            label = np.flip(label, 0)
        if random.random() < 0.5:
            image = np.flip(image, 1)
            label = np.flip(label, 1)
        if random.random() < 0.5:
            image = np.flip(image, 2)
            label = np.flip(label, 2)

        return {'image': image, 'label': label}


class Random_Crop(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        H = random.randint(0, 182 - 128)
        W = random.randint(0, 218 - 128)
        D = random.randint(0, 182 - 128)
        image = image[H: H + 128, W: W + 128, D: D + 128, ...]
        label = label[..., H: H + 128, W: W + 128, D: D + 128]

        return {'image': image, 'label': label}


class Random_intencity_shift(object):
    def __call__(self, sample, factor=0.1):
        image = sample['image']
        label = sample['label']

        scale_factor = np.random.uniform(1.0-factor, 1.0+factor, size=[1, image.shape[1], 1, image.shape[-1]])
        shift_factor = np.random.uniform(-factor, factor, size=[1, image.shape[1], 1, image.shape[-1]])

        image = image*scale_factor+shift_factor

        return {'image': image, 'label': label}


class Random_rotate(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        angle = round(np.random.uniform(-10, 10), 2)
        image = ndimage.rotate(image, angle, axes=(0, 1), reshape=False)
        label = ndimage.rotate(label, angle, axes=(0, 1), reshape=False)

        return {'image': image, 'label': label}


class BrightnessTransform(object):

    def __init__(self, mu, sigma, per_channel=True, p_per_sample=1., p_per_channel=1.):
        """
        Augments the brightness of data. Additive brightness is sampled from Gaussian distribution with mu and sigma
        :param mu: mean of the Gaussian distribution to sample the added brightness from
        :param sigma: standard deviation of the Gaussian distribution to sample the added brightness from
        :param per_channel: whether to use the same brightness modifier for all color channels or a separate one for
        each channel
        :param data_key:
        :param p_per_sample:
        """
        self.p_per_sample = p_per_sample
        self.mu = mu
        self.sigma = sigma
        self.per_channel = per_channel
        self.p_per_channel = p_per_channel

    def __call__(self, sample):
        data, label = sample['image'], sample['label']

        for b in range(data.shape[0]):
            if np.random.uniform() < self.p_per_sample:
                data[b] = augment_brightness_additive(data[b], self.mu, self.sigma, self.per_channel,
                                                      p_per_channel=self.p_per_channel)

        return {'image': data, 'label': label}


def augment_brightness_additive(data_sample, mu:float, sigma:float , per_channel:bool=True, p_per_channel:float=1.):
    """
    data_sample must have shape (c, x, y(, z)))
    :param data_sample:
    :param mu:
    :param sigma:
    :param per_channel:
    :param p_per_channel:
    :return:
    """
    if not per_channel:
        rnd_nb = np.random.normal(mu, sigma)
        for c in range(data_sample.shape[0]):
            if np.random.uniform() <= p_per_channel:
                data_sample[c] += rnd_nb
    else:
        for c in range(data_sample.shape[0]):
            if np.random.uniform() <= p_per_channel:
                rnd_nb = np.random.normal(mu, sigma)
                data_sample[c] += rnd_nb
    return data_sample


def augment_gaussian_noise(data_sample, noise_variance=(0, 0.1)):
    if noise_variance[0] == noise_variance[1]:
        variance = noise_variance[0]
    else:
        variance = random.uniform(noise_variance[0], noise_variance[1])
    data_sample = data_sample + np.random.normal(0.0, variance, size=data_sample.shape)
    return data_sample


class GaussianNoise(object):
    def __init__(self, noise_variance=(0, 0.1), p=0.5):
        self.prob = p
        self.noise_variance = noise_variance

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        if np.random.uniform() < self.prob:
            image = augment_gaussian_noise(image, self.noise_variance)
        return {'image': image, 'label': label}


class ContrastAugmentationTransform(object):
    def __init__(self, contrast_range=(0.75, 1.25), preserve_range=True, per_channel=True, p_per_sample=1.):
        """
        Augments the contrast of data
        :param contrast_range: range from which to sample a random contrast that is applied to the data. If
        one value is smaller and one is larger than 1, half of the contrast modifiers will be >1 and the other half <1
        (in the inverval that was specified)
        :param preserve_range: if True then the intensity values after contrast augmentation will be cropped to min and
        max values of the data before augmentation.
        :param per_channel: whether to use the same contrast modifier for all color channels or a separate one for each
        channel
        :param data_key:
        :param p_per_sample:
        """
        self.p_per_sample = p_per_sample
        self.contrast_range = contrast_range
        self.preserve_range = preserve_range
        self.per_channel = per_channel

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        for b in range(len(image)):
            if np.random.uniform() < self.p_per_sample:
                image[b] = augment_contrast(image[b], contrast_range=self.contrast_range,
                                            preserve_range=self.preserve_range, per_channel=self.per_channel)
        return {'image': image, 'label': label}


def augment_contrast(data_sample, contrast_range=(0.75, 1.25), preserve_range=True, per_channel=True):
    if not per_channel:
        mn = data_sample.mean()
        if preserve_range:
            minm = data_sample.min()
            maxm = data_sample.max()
        if np.random.random() < 0.5 and contrast_range[0] < 1:
            factor = np.random.uniform(contrast_range[0], 1)
        else:
            factor = np.random.uniform(max(contrast_range[0], 1), contrast_range[1])
        data_sample = (data_sample - mn) * factor + mn
        if preserve_range:
            data_sample[data_sample < minm] = minm
            data_sample[data_sample > maxm] = maxm
    else:
        for c in range(data_sample.shape[0]):
            mn = data_sample[c].mean()
            if preserve_range:
                minm = data_sample[c].min()
                maxm = data_sample[c].max()
            if np.random.random() < 0.5 and contrast_range[0] < 1:
                factor = np.random.uniform(contrast_range[0], 1)
            else:
                factor = np.random.uniform(max(contrast_range[0], 1), contrast_range[1])
            data_sample[c] = (data_sample[c] - mn) * factor + mn
            if preserve_range:
                data_sample[c][data_sample[c] < minm] = minm
                data_sample[c][data_sample[c] > maxm] = maxm
    return data_sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image = sample['image']
        image = np.ascontiguousarray(image.transpose(3, 0, 1, 2))
        label = sample['label']
        label = np.ascontiguousarray(label)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
        return {'image': image, 'label': label}
    

def transform(sample):
    trans = transforms.Compose([
        Random_Crop(),
        Random_rotate(),
        Random_Flip(),
        BrightnessTransform(0, 0.1, True, 0.15, 0.5),
        ContrastAugmentationTransform(p_per_sample=0.15),
        Random_intencity_shift(),
        GaussianNoise(p=0.1),
        ToTensor()
    ])

    return trans(sample)


def transform_valid(sample):
    trans = transforms.Compose([
        Random_Crop(),
        ToTensor()
    ])

    return trans(sample)


def transform_test(sample):
    trans = transforms.Compose([
        ToTensor()
    ])

    return trans(sample)



class PklIDH(Dataset):

    def __init__(self, root_dir, phase='train'):  
        super(PklIDH, self).__init__()
        self.root_dir = root_dir 
        list_ID = os.listdir(root_dir)
        list_ID.sort()
        self.data_list = list_ID
        self.phase = phase

    def __len__(self): 
        return len(self.data_list)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]

    def __getitem__(self, idx):
        patient = self.data_list[idx]
        img_path = os.path.join(self.root_dir, patient)
        data = pkload(img_path)
        patient_ID = patient[: -9]
        if self.phase == 'train':
            image, label, IDH = data[0], data[1], data[2]
            sample = {'image': image, 'label': label}
            sample = transform(sample)
            IDH = torch.tensor(IDH).type(torch.LongTensor)
            
            mask_array = sample['label'].unsqueeze(0)
            return {'image': sample['image'], 'mask': mask_array, 'label': IDH, 'patient_ID': patient_ID}

        if self.phase == 'valid':

            image, label, IDH = data[0], data[1], data[2]
            sample = {'image': image, 'label': label}
            sample = transform_valid(sample)
            IDH = torch.tensor(IDH).type(torch.LongTensor)

            mask_array = sample['label'].unsqueeze(0)
            return {'image': sample['image'], 'mask': mask_array, 'label': IDH, 'patient_ID': patient_ID}

        if self.phase == 'test':

            image, label, IDH = data[0], data[1], data[2]
            sample = {'image': image, 'label': label}
            sample = transform_test(sample)
            IDH = torch.tensor(IDH).type(torch.LongTensor)

            mask_array = sample['label'].unsqueeze(0)
            return {'image': sample['image'], 'mask': mask_array, 'label': IDH, 'patient_ID': patient_ID}

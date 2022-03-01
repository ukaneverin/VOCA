import numpy as np
import random
from scipy.misc import toimage
from pdb import set_trace


class CorlorJitter(object):
    def __call__(self, sample):
        from torchvision.transforms import ColorJitter as tvColorJitter
        ColorJitter_transform = tvColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
        sample['image'] = ColorJitter_transform(sample['image'])
        return sample


class ToTensor(object):
    def __call__(self, sample):
        from torchvision.transforms import ToTensor as tvToTensor
        totensor_transform = tvToTensor()
        sample['image'] = totensor_transform(sample['image'])
        return sample


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        from torchvision.transforms import Normalize as tvNormalize
        normalize_transform = tvNormalize(self.mean, self.std)
        sample['image'] = normalize_transform(sample['image'])
        return sample


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        n = int(sample['task_map'].shape[-1] / 4)  # n is the number of classes
        if random.random() < 0.5:
            sample['image'] = np.fliplr(sample['image'])
            sample['gt_map'] = np.fliplr(sample['gt_map']).copy()
            sample['task_map'] = np.fliplr(sample['task_map']).copy()
            sample['task_map'][:, :, n:2 * n] *= -1  # x should be reversed

        sample['image'] = toimage(sample['image'], mode='RGB')
        return sample


class RandomVerticalFlip(object):
    def __call__(self, sample):
        n = int(sample['task_map'].shape[-1] / 4)  # n is the number of classes
        if random.random() < 0.5:
            sample['image'] = np.flipud(sample['image'])
            sample['gt_map'] = np.flipud(sample['gt_map']).copy()
            sample['task_map'] = np.flipud(sample['task_map']).copy()
            sample['task_map'][:, :, 2 * n:3 * n] *= -1  # y should be reversed

        sample['image'] = toimage(sample['image'], mode='RGB')
        return sample


class RandomRotate(object):
    def __call__(self, sample):
        n = int(sample['task_map'].shape[-1] / 4)  # n is the number of classes

        k = np.random.randint(0, 4)
        sample['image'] = np.rot90(sample['image'], k)
        sample['gt_map'] = np.rot90(sample['gt_map'], k).copy()
        sample['task_map'] = np.rot90(sample['task_map'], k).copy()
        if k == 1:
            """swap x and y map"""
            x_map = sample['task_map'][:, :, n:2 * n].copy()
            y_map = sample['task_map'][:, :, 2 * n:3 * n].copy()
            sample['task_map'][:, :, n:2 * n] = y_map
            sample['task_map'][:, :, 2 * n:3 * n] = x_map
            sample['task_map'][:, :, 2 * n:3 * n] *= -1
        if k == 2:
            sample['task_map'][:, :, n:3 * n] *= -1
        if k == 3:
            """swap x and y map"""
            x_map = sample['task_map'][:, :, n:2 * n].copy()
            y_map = sample['task_map'][:, :, 2 * n:3 * n].copy()
            sample['task_map'][:, :, n:2 * n] = y_map
            sample['task_map'][:, :, 2 * n:3 * n] = x_map
            sample['task_map'][:, :, n:2 * n] *= -1

        sample['image'] = toimage(sample['image'], mode='RGB')
        return sample

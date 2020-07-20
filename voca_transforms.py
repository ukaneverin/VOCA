import numpy as np
import random
from scipy.misc import toimage
import sys

class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, sample):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        from torchvision.transforms import ToTensor as tvToTensor
        totensor_transform = tvToTensor()
        sample['image'] = totensor_transform(sample['image'])
        return sample


class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        from torchvision.transforms import Normalize as tvNormalize
        normalize_transform = tvNormalize(self.mean, self.std)
        sample['image'] = normalize_transform(sample['image'])
        return sample

class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a probability of 0.5."""

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        n = int(sample['task_map'].shape[-1] / 4) # n is the number of classes
        if random.random() < 0.5:
            sample['image'] = np.fliplr(sample['image'])
            sample['gt_map'] = np.fliplr(sample['gt_map']).copy()
            sample['task_map'] = np.fliplr(sample['task_map']).copy()
            sample['task_map'][:,:,n:2*n] *= -1 # x should be reversed

        sample['image'] = toimage(sample['image'], mode='RGB')
        return sample


class RandomVerticalFlip(object):
    """Vertically flip the given PIL Image randomly with a probability of 0.5."""

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        n = int(sample['task_map'].shape[-1] / 4)  # n is the number of classes
        if random.random() < 0.5:
            sample['image'] = np.flipud(sample['image'])
            sample['gt_map'] = np.flipud(sample['gt_map']).copy()
            sample['task_map'] = np.flipud(sample['task_map']).copy()
            sample['task_map'][:,:,2*n:3*n] *= -1 # y should be reversed

        sample['image'] = toimage(sample['image'], mode='RGB')
        return sample

class RandomRotate(object):
    """Rotate the given PIL Image counterclockwise with a probability of 0.25 for each angle: [90, 180, 270]."""

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        n = int(sample['task_map'].shape[-1] / 4)  # n is the number of classes

        k = np.random.randint(0,4)
        sample['image'] = np.rot90(sample['image'], k)
        sample['gt_map'] = np.rot90(sample['gt_map'], k).copy()
        sample['task_map'] = np.rot90(sample['task_map'], k).copy()
        if k == 1:
            """swap x and y map"""
            x_map = sample['task_map'][:,:,n:2*n].copy()
            y_map = sample['task_map'][:,:,2*n:3*n].copy()
            sample['task_map'][:,:,n:2*n] = y_map
            sample['task_map'][:,:,2*n:3*n] = x_map
            sample['task_map'][:,:,2*n:3*n] *= -1
        if k == 2:
            sample['task_map'][:,:,n:3*n] *= -1
        if k == 3:
            """swap x and y map"""
            x_map = sample['task_map'][:,:,n:2*n].copy()
            y_map = sample['task_map'][:,:,2*n:3*n].copy()
            sample['task_map'][:,:,n:2*n] = y_map
            sample['task_map'][:,:,2*n:3*n] = x_map
            sample['task_map'][:,:,n:2*n] *= -1

        sample['image'] = toimage(sample['image'], mode='RGB')
        return sample

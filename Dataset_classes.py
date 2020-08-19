from torch.utils.data import Dataset
import os
import itertools
from imageio import imread
from scipy.misc import toimage
import numpy as np

class TestImageDataset(Dataset):
    """
    The dataset to load grid patches of ONE test image
    """
    def __init__(self, image, grid_size=127, crop_size=127, transform=None):
        """
        Args:
            image (numpy array): input image for cell detection
            grid_size: the stride of patches
            crop_size: the size of patches
            transform (callable, optional): Optional transform to be applied
                on patches.
        """

        x_grid = list(range(0, image.shape[1]-crop_size, grid_size)) + [image.shape[1]-crop_size]
        y_grid = list(range(0, image.shape[0]-crop_size, grid_size)) + [image.shape[0]-crop_size]

        coords_list = list(itertools.product(x_grid, y_grid))

        self.grid_size = grid_size
        self.image = image
        self.coords_list = coords_list
        self.crop_size = crop_size
        self.transform = transform

    def __len__(self):
        return len(self.coords_list)

    def __getitem__(self, idx):
        coord_x = self.coords_list[idx][0]
        coord_y = self.coords_list[idx][1]
        image_crop = self.image[coord_y:coord_y+self.crop_size, coord_x:coord_x+self.crop_size, :]
        image_crop = toimage(image_crop, mode='RGB')
        sample = {'image_crop': image_crop, 'coords': np.asarray([coord_x, coord_y], dtype=float)}

        if self.transform:
            sample['image_crop'] = self.transform(sample['image_crop']) #to float tensor and more

        return sample

class TrainImagesDataset(Dataset):
    """
    The dataset to load grid patches of train images
    """
    def __init__(self, image_folder, map_folder, gt_folder, split_folder, cv, crop_size, r, phase, transform=None):
        with open(os.path.join(split_folder, 'CV_%s_%s.csv' % (cv, phase)), 'r') as f:
            train_image_list = [line.strip() for line in f]
        orignal_images = {}; task_maps = {}; gt_maps = {}
        for image_name in train_image_list:
            image_path = os.path.join(image_folder, image_name)
            image = imread(image_path)
            image = image[:,:,:3]
            exec("orignal_images['%s'] = image" % image_name)
            task_map = np.load(os.path.join(map_folder+'%s_r%s.npy' % (image_name, r)))
            exec("task_maps['%s'] = task_map" % image_name)
            gt_map = np.load(os.path.join(gt_folder+'%s.npy' % image_name))
            exec("gt_maps['%s'] = gt_map" % image_name)

        with open(os.path.join(split_folder, 'library_%s_%s.csv' % (cv, phase)), 'r') as crops_file:
            crops_list = [line.strip().split(',') for line in crops_file]

        self.orignal_images = orignal_images
        self.task_maps = task_maps
        self.gt_maps = gt_maps
        self.crops_list = crops_list
        self.crop_size = crop_size
        self.transform = transform

    def __len__(self):
        return len(self.crops_list)

    def __getitem__(self, idx):
        crops_info = self.crops_list[idx]
        img_name = crops_info[0]
        coords = np.asarray([int(crops_info[1]), int(crops_info[2])])

        image_crop = self.orignal_images[img_name][coords[1]:coords[1]+self.crop_size, coords[0]:coords[0]+self.crop_size, :]
        task_map = self.task_maps[img_name][coords[1]:coords[1]+self.crop_size, coords[0]:coords[0]+self.crop_size, :]
        gt_map = self.gt_maps[img_name][coords[1]:coords[1] + self.crop_size, coords[0]:coords[0] + self.crop_size,:]

        sample = {'image': image_crop, 'image_name': img_name, 'coords': coords, 'task_map': task_map, 'gt_map': gt_map}

        if self.transform:
            '''
            self defined transforms. since we also need to do same transforms on the task maps
            '''
            sample = self.transform(sample)

        return sample

from torch.utils.data import Dataset
import os
import itertools
from imageio import imread
from scipy.misc import toimage
import numpy as np
import glob
import json
from utils import non_max_suppression, remove_non_assigned, get_mean_and_std
import torch
from abc import ABC, abstractmethod
from voca_transforms import ToTensor
import argparse
from pdb import set_trace
import random
from typing import Callable


class VocaData(ABC):
    def __init__(self, root: str, args: argparse.Namespace, cell_type_dict: dict):
        """
        A class to handle and process VOCA data.
        :param root: root directory where all train-validation files will be saved under
        :param args: the parsed parameters from running scripts
        :param cell_type_dict: a dictionary indexing the cell types, e.g. {0: 'tumor', 1: 'lymphocyte'}
        """
        self.root = root
        self.image_folder = args.image_folder
        self.n_cv = args.n_cv
        self.input_size = args.input_size
        self.num_classes = args.num_classes
        self.r = args.r
        self.cell_type_dict = cell_type_dict

        self.status = {}
        self.check_status()

        if not os.path.exists(self.root):
            os.makedirs(self.root)

        self.image_list = os.listdir(self.image_folder)

        # make sub-directories
        self.annotation_path = os.path.join(self.root, 'annotations/')
        if not os.path.exists(self.annotation_path):
            os.mkdir(self.annotation_path)

        self.task_map_path = os.path.join(root, 'task_map/')
        if not os.path.exists(self.task_map_path):
            os.mkdir(self.task_map_path)

        self.gt_map_path = os.path.join(root, 'gt_map/')
        if not os.path.exists(self.gt_map_path):
            os.mkdir(self.gt_map_path)

        self.library_path = os.path.join(root, 'library/')
        if not os.path.exists(self.library_path):
            os.mkdir(self.library_path)

        self.model_directory = os.path.join(root, 'trained_models/')
        if not os.path.exists(self.model_directory):
            os.mkdir(self.model_directory)

        self.val_path = os.path.join(root, 'val_results/')
        if not os.path.exists(self.val_path):
            os.mkdir(self.val_path)

    def process(self) -> None:
        self.check_status()
        if not self.status['annotation']:
            self.get_annotation()
            open(os.path.join(self.root, 'annotation'), 'w+').close()
        if not self.status['split']:
            self.split()
            open(os.path.join(self.root, 'split'), 'w+').close()
        if not self.status['map']:
            self.generate_task_map()
            open(os.path.join(self.root, 'map'), 'w+').close()

    @abstractmethod
    def get_annotation(self):
        """
        Customized method to get the annotation of nuclei positions:
        The annotation files must be named as "image_file_name.json",
        which is a list of dictionaries with keys {'x', 'y', 'class'}
        Each element in the list represent one cell.
        e.g. [{"project_id":"1","image_id":"my_image_1.png","label_type":"nucleus","x":"108","y":"567","class":"1"},
        {"project_id":"1","image_id":"my_image_1.png","label_type":"nucleus","x":"116","y":"578","class":"1"}]
        """
        ...

    def split(self, n_cv: int = None, input_size: int = None) -> None:
        """
        Split the dataset for cross-validation.
        :param n_cv: number of folds in cross-validation.
        :param input_size: the input size for the VOCA model
        :return: saved patch library (coordinates of crops) of each subsample
        """
        if n_cv is not None:
            self.n_cv = n_cv
        if input_size is not None:
            self.input_size = input_size
        for i in range(self.n_cv):
            val_list = self.image_list[
                       i * int(len(self.image_list) / self.n_cv): (i + 1) * int(len(self.image_list) / self.n_cv)]
            train_list = list(set(self.image_list) - set(val_list))
            # split dataset
            with open(os.path.join(self.library_path, 'CV_%s_train.csv' % i), 'w+') as train_file:
                for img in train_list:
                    train_file.write(img + '\n')
            with open(os.path.join(self.library_path, 'CV_%s_val.csv' % i), 'w+') as val_file:
                for img in val_list:
                    val_file.write(img + '\n')
            # generate coordinate libraries
            patch_library_train = open(os.path.join(self.library_path, 'library_%s_train.csv' % i), 'w+')
            for image_name in train_list:
                image = imread(os.path.join(self.image_folder, image_name))
                for crop_x in range(0, image.shape[1] - self.input_size, 17):
                    for crop_y in range(0, image.shape[0] - self.input_size, 17):
                        patch_library_train.write('%s,%s,%s\n' % (image_name, crop_x, crop_y))
            patch_library_train.close()

            patch_library_val = open(os.path.join(self.library_path, 'library_%s_val.csv' % i), 'w+')
            for image_name in val_list:
                image = imread(os.path.join(self.image_folder, image_name))
                for crop_x in range(0, image.shape[1] - self.input_size, 17):
                    for crop_y in range(0, image.shape[0] - self.input_size, 17):
                        patch_library_val.write('%s,%s,%s\n' % (image_name, crop_x, crop_y))
            patch_library_val.close()

    def generate_task_map(self, num_classes: int = None, r: int = None) -> None:
        """
        Generate the task maps (cls_map, vector_map, wt_map) for supervision.
        :param num_classes: number of cell types
        :param r: radius of the disk around gt
        :return: saved task maps
        """
        self.check_status()
        if not self.status['annotation']:
            raise Exception("Task maps cannot be generated without downloading annotations")
        if num_classes is not None:
            self.num_classes = num_classes
        if r is not None:
            self.r = r
        # generate the task maps for VOCA (saving to disk for training speed)
        for image_name in self.image_list:
            image_path = os.path.join(self.image_folder, image_name)
            image = imread(image_path)
            image = image[:, :, :3]
            # annotations must be store in .json: a list of dicts with keys {'x', 'y', 'class'}
            image_label_file = os.path.join(self.annotation_path, image_name + '.json')
            with open(image_label_file, 'r') as f:
                coords_data = json.load(f)

            # create ground truth map as a matrix os shape: (image_shape[0], image.shape[1], c)
            # c channels, each channel represent one cell type; entries: {0: background; 1: cell}
            gt_map = np.zeros((image.shape[0], image.shape[1], self.num_classes))
            dist_map = np.ones((image.shape[0], image.shape[1], self.num_classes)) * \
                       ((image.shape[0]) ** 2 + (image.shape[1]) ** 2)
            # task_map: prediction targets for VOCA; channels: [cls_map, x_map, y_map, wt_map]
            task_map = np.zeros((image.shape[0], image.shape[1], self.num_classes * 4))
            print("Generating task_map for image: %s" % image_name)
            for p in coords_data:
                x = max(min(int(p['x']), image.shape[1] - 1), 0)  # some times annotation is off image lmao
                y = max(min(int(p['y']), image.shape[0] - 1), 0)
                c = int(p['class'])
                # create the point map
                gt_map[y, x, c] = 1  # y is axis 0, x is axis 1; y = vertical, x = horizontal; (0,0) is top left

                # create the task maps------------------------
                # cls_map
                y_disk, x_disk = np.ogrid[-y: image.shape[0] - y, -x: image.shape[1] - x]
                disk_mask = y_disk ** 2 + x_disk ** 2 <= self.r ** 2
                task_map[disk_mask, c] = 1  # update cls_map

                # x, y map; recording a dist_map to know whether should update to current ground truth
                x_coord_matrix = np.array([np.arange(image.shape[1]), ] * image.shape[0])
                y_coord_matrix = np.array([np.arange(image.shape[0]), ] * image.shape[1]).transpose()
                x_map = x - x_coord_matrix
                y_map = y - y_coord_matrix
                # update mask is where in disk mask the distance is smaller than previous smallest dist
                update_mask = ((x_map ** 2 + y_map ** 2) < dist_map[:, :, c]) * disk_mask
                # now we update the dist_map to the smaller dist
                dist_map[disk_mask, c] = np.minimum((x_map ** 2 + y_map ** 2)[disk_mask], dist_map[disk_mask, c])
                task_map[update_mask, self.num_classes + c] = x_map[update_mask]  # update x_map
                task_map[update_mask, 2 * self.num_classes + c] = y_map[update_mask]  # update y_map

                # update wt map
                task_map[disk_mask, 3 * self.num_classes + c] += 1

            task_map_save_file = os.path.join(self.task_map_path, '%s_r%s.npy' % (image_name, self.r))
            np.save(task_map_save_file, task_map)
            gt_map_save_file = os.path.join(self.gt_map_path, '%s.npy' % image_name)
            np.save(gt_map_save_file, gt_map)

    def get_mean_std(self, cv: int):
        """
        Calculate the normalization parameters
        :param cv: which cross-validation subsample
        :return: normalization parameters: [mean, std, pos_ratio]
        """
        self.check_status()
        if not all(self.status.values()):
            raise Exception("Normalization can only be calculated after dataset is fully processed")
        try:
            mean = torch.FloatTensor(np.load(os.path.join(self.library_path, 'CV_%s_mean.npy' % cv)))
            std = torch.FloatTensor(np.load(os.path.join(self.library_path, 'CV_%s_std.npy' % cv)))
            sample_sizes = torch.FloatTensor(np.load(os.path.join(self.library_path, 'CV_%s_samplesizes_r%s.npy' % (cv, self.r))))
        except:
            mean, std, sample_sizes = get_mean_and_std(128 * 8, TrainImagesDataset('train', ToTensor(), cv, **self.__dict__), self.num_classes)
            np.save(os.path.join(self.library_path, 'CV_%s_mean.npy' % cv), mean)
            np.save(os.path.join(self.library_path, 'CV_%s_std.npy' % cv), std)
            np.save(os.path.join(self.library_path, 'CV_%s_samplesizes_r%s.npy' % (cv, self.r)), sample_sizes)
        return mean, std, sample_sizes

    def check_status(self) -> None:
        """
        Check the status of the dataset for all check points
        :return: dictionary of boolean values indicating the status of each check point
        """
        for p in ['annotation', 'map', 'split']:
            self.status[p] = os.path.isfile(os.path.join(self.root, p))


class TrainImagesDataset(Dataset):
    """
    The dataset to load grid patches of train images
    """
    def __init__(self, phase, transform, cv, subsample, **data_attr):
        with open(os.path.join(data_attr['library_path'], 'CV_%s_%s.csv' % (cv, phase)), 'r') as f:
            train_image_list = [line.strip() for line in f]
        orignal_images = {}
        task_maps = {}
        gt_maps = {}
        for image_name in train_image_list:
            image_path = os.path.join(data_attr['image_folder'], image_name)
            image = imread(image_path)
            image = image[:, :, :3]
            orignal_images[image_name] = image
            task_maps[image_name] = np.load(os.path.join(data_attr['task_map_path'], '%s_r%s.npy' % (image_name, data_attr['r'])))
            gt_maps[image_name] = np.load(os.path.join(data_attr['gt_map_path'], '%s.npy' % image_name))

        with open(os.path.join(data_attr['library_path'], 'library_%s_%s.csv' % (cv, phase)), 'r') as crops_file:
            crops_list = [line.strip().split(',') for line in crops_file]

        crops_list = random.sample(crops_list, int(subsample*len(crops_list)))

        self.orignal_images = orignal_images
        self.task_maps = task_maps
        self.gt_maps = gt_maps
        self.crops_list = crops_list
        self.crop_size = data_attr['input_size']
        self.transform = transform

    def __len__(self):
        return len(self.crops_list)

    def __getitem__(self, idx):
        crops_info = self.crops_list[idx]
        img_name = crops_info[0]
        coords = np.asarray([int(crops_info[1]), int(crops_info[2])])

        image_crop = self.orignal_images[img_name][coords[1]:coords[1] + self.crop_size,
                     coords[0]:coords[0] + self.crop_size, :]
        task_map = self.task_maps[img_name][coords[1]:coords[1] + self.crop_size, coords[0]:coords[0] + self.crop_size, :]
        gt_map = self.gt_maps[img_name][coords[1]:coords[1] + self.crop_size, coords[0]:coords[0] + self.crop_size, :]

        sample = {'image': image_crop, 'image_name': img_name, 'coords': coords, 'task_map': task_map, 'gt_map': gt_map}

        if self.transform:
            '''
            self defined transforms. since we also need to do same transforms on the task maps
            '''
            sample = self.transform(sample)

        return sample


class TestImageDataset(Dataset):
    """
    The dataset to load grid patches of ONE test image
    """
    def __init__(self, image, crop_size=127, transform=None):
        """
        Args:
            image (numpy array): input image for cell detection
            grid_size: the stride of patches
            crop_size: the size of patches
            transform (callable, optional): Optional transform to be applied
                on patches.
        """

        x_grid = list(range(0, image.shape[1] - crop_size, crop_size)) + [image.shape[1] - crop_size]
        y_grid = list(range(0, image.shape[0] - crop_size, crop_size)) + [image.shape[0] - crop_size]

        coords_list = list(itertools.product(x_grid, y_grid))

        self.image = image
        self.coords_list = coords_list
        self.crop_size = crop_size
        self.transform = transform

    def __len__(self):
        return len(self.coords_list)

    def __getitem__(self, idx):
        coord_x = self.coords_list[idx][0]
        coord_y = self.coords_list[idx][1]
        image_crop = self.image[coord_y:coord_y + self.crop_size, coord_x:coord_x + self.crop_size, :]
        image_crop = toimage(image_crop, mode='RGB')
        sample = {'image_crop': image_crop, 'coords': np.asarray([coord_x, coord_y], dtype=float)}

        if self.transform:
            sample['image_crop'] = self.transform(sample['image_crop'])  # to float tensor and more

        return sample

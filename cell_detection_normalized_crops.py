import numpy as np
import glob
from scipy.misc import imread,imsave,toimage
from PIL import Image
import sys
import os
from models import * #models defined
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
import time
from scipy.optimize import linear_sum_assignment as lsa
from scipy.spatial.distance import cdist
from Dataset_classes import LymphocytesTestImage
from skimage.util import invert

import openslide
from SlideTileExtractor import extract_tissue

option = sys.argv[1] # jointweighted: the training option
dis_thresh = sys.argv[2] # r for cls_map
config = sys.argv[3]  # which model
train_set = sys.argv[4] # b and l: breast and lung data
disThresh = int(sys.argv[5]) # NMS threshold
slide_id = sys.argv[6]

data_directory = '/lila/data/fuchs/xiec/results/auto_tri/'
model_directory = data_directory + 'trained_models/'

mean = np.load(data_directory+'crops_train_patch_mean_%s.npy' % train_set).tolist()
std = np.load(data_directory+'crops_train_patch_std_%s.npy' % train_set).tolist()

data_transforms = transforms.Compose([
                  transforms.ToTensor(),
                  transforms.Normalize(mean, std)
                  ])

use_gpu = torch.cuda.is_available()

model_ft = ResNet18()
model_ft.load_state_dict(torch.load(model_directory+'trained_cell_vgg11_%s_%s_%s_%s.pth' % (option, config, dis_thresh, train_set)))


if use_gpu:
    model_ft = model_ft.cuda()

model_ft.train(False)

def non_max_suppression(points, disThresh):
    # if there are no points, return an empty list
    if len(points) == 0:
        return np.asarray([]).reshape(-1,3)

    if points.dtype.kind == 'i':
        points = points.astype('float')
    np.random.shuffle(points)

    pick = []

    x = points[:,0]
    y = points[:,1]
    score = points[:,2]

    idxs = np.argsort(score)

    while len(idxs) > 0:

        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        x_dis = x[idxs[:last]] -  x[i]
        y_dis = y[idxs[:last]] -  y[i]

        dis = (x_dis**2+y_dis**2)**0.5 # threshold is distance instead of IoU

        idxs = np.delete(idxs, np.concatenate(([last],
        	np.where(dis < disThresh)[0])))

    return points[pick]


crop_size = 32
input_size = 127
crop_radius = int(crop_size/2)
grid_size = 127
tile_size = 508

"""
make grid for slides
This dataset is 20x .5 mpp. Only detecting the lymphocytes; Be VERY cafeful about the resolution, coords, etc...
"""

image_path = '/lila/data/fuchs/projects/lung/nuclei_vs_mask_test_images/%s.png' % slide_id

cell_file = open('/lila/data/fuchs/projects/lung/nuclei_vs_mask_test_images/coords/%s_lymphocytes_0.2.csv' % slide_id, 'w+')


since = time.time()
image = imread(image_path)
image = image[:,:,:3]
image = image.astype(float)

image_datasets = LymphocytesTestImage(image, grid_size, input_size, data_transforms)

dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=int((tile_size/input_size)**2), shuffle=True, num_workers=4)
dataset_sizes = len(image_datasets)


confidence_mask = np.zeros((image.shape[:-1]))
confidence_n = np.zeros((image.shape[:-1]))

for sample in dataloaders:

    inputs = sample['image_crop']
    coords = sample['coords'].numpy()

    # wrap them in Variable
    if use_gpu:
        inputs = Variable(inputs.cuda())
    else:
        inputs = Variable(inputs)
    # forward
    class_probs, trans_preds, num_preds = model_ft(inputs)
    if use_gpu:
        class_probs = class_probs.data.cpu().numpy()
        trans_preds = trans_preds.data.cpu().numpy()
        num_preds = num_preds.data.cpu().numpy()
    else:
        class_probs = class_probs.data.numpy()
        trans_preds = trans_preds.data.numpy()
        num_preds = num_preds.data.numpy()

    for coord_i in range(coords.shape[0]):
        x = int(coords[coord_i, 0])
        y = int(coords[coord_i, 1])
        for i in range(input_size):
            for j in range(input_size):
                map_i = x + i
                map_j = y + j

                if confidence_n[map_i,map_j] == 0:
                    x_trans = trans_preds[coord_i, 0, i, j]
                    y_trans = trans_preds[coord_i, 1, i, j]
                    num_contribution = num_preds[coord_i, 0, i, j]
                    confidence_x = int(round(map_i+x_trans))
                    confidence_y = int(round(map_j+y_trans))
                    if confidence_x>=0 and confidence_x<image.shape[0] and confidence_y>=0 and confidence_y<image.shape[1]:
                        confidence_mask[confidence_x, confidence_y] += class_probs[coord_i, 0, i, j] * num_contribution
                confidence_n[map_i,map_j] += 1


detected_coords = []
for x in range(confidence_mask.shape[0]):
    for y in range(confidence_mask.shape[1]):
        if confidence_mask[x,y] > 0:
            detected_coords.append([x,y,confidence_mask[x,y]])
detected_coords = np.asarray(detected_coords, dtype=float).reshape(-1,3)
detected_coords = non_max_suppression(detected_coords, disThresh) #non_max_suppression


suppression_mask = np.zeros((image.shape[:-1]))
for coords in detected_coords:
    suppression_mask[int(coords[0]), int(coords[1])] = confidence_mask[max(int(coords[0])-2, 0):min(int(coords[0])+3, image.shape[0]), \
                                                                       max(int(coords[1])-2, 0):min(int(coords[1])+3, image.shape[1])].sum() #real accumulated confidence map (-3 to +3 pixels)

#normalize suppression_mask
minval = suppression_mask.min()
maxval = suppression_mask.max()
suppression_mask -= minval
suppression_mask = suppression_mask/(maxval-minval)


plot_coords = np.transpose(np.nonzero(suppression_mask > 0.1))
for dc in plot_coords:
    cell_file.write('%s,%s,%s,%s\n' % (slide_id, dc[1], dc[0], suppression_mask[dc[0],dc[1]]))
time_elapsed = time.time() - since

cell_file.close()

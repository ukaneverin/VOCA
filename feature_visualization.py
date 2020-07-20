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
option = sys.argv[1]
dis_thresh = sys.argv[2]
config = sys.argv[3]
train_set = sys.argv[4]
test_set = sys.argv[5]


class_number = [1.0,2.0]
data_directory = '/lila/data/fuchs/xiec/TIL_detection/'
model_directory = data_directory + 'trained_models/'


learned_hyper_file = open(data_directory+'hyper_learned/hyper_file_%s_%s_%s_%s_%s.txt' % (option, config, dis_thresh, train_set, test_set), 'r')
learned_hyper = [line.strip().split(',') for line in learned_hyper_file][-1]
overlapThresh = float(learned_hyper[1])

mean = np.load(data_directory+'crops_train_patch_mean_%s.npy' % train_set).tolist()
std = np.load(data_directory+'crops_train_patch_std_%s.npy' % train_set).tolist()

data_transforms = transforms.Compose([
                  transforms.ToTensor(),
                  transforms.Normalize(mean, std)
                  ])

use_gpu = torch.cuda.is_available()

if option == 'cls':
    model_ft = cell_vgg(config, num_classes = 2)
    model_ft.load_state_dict(torch.load(model_directory+'trained_cell_vgg11_cls_%s.pth' % dis_thresh))
else:
    if config == 'vgg':
        model_ft = cell_vgg_cls_reg('config2')
    elif config == 'vggsplit':
        model_ft = cell_vgg_cls_reg_split('config2')
    elif config == 'vggsub':
        model_ft = cell_vgg_cls_reg_sub('config2')
    elif config == 'vggskip':
        model_ft = cell_vgg_cls_skip('config2')
    elif config == 'res':
        model_ft = ResNet18()
    elif config == 'dense':
        model_ft = DenseNet121()
    model_ft.load_state_dict(torch.load(model_directory+'trained_cell_vgg11_%s_%s_%s_%s.pth' % (option, config, dis_thresh, train_set)))


if use_gpu:
    model_ft = model_ft.cuda()

model_ft.train(False)

def non_max_suppression(points, overlapThresh, crop_size):
    crop_size = float(crop_size)
    crop_radius = crop_size/2
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

        xx1 = np.maximum(x[i], x[idxs[:last]]) - crop_radius
        yy1 = np.maximum(y[i], y[idxs[:last]]) - crop_radius
        xx2 = np.minimum(x[i], x[idxs[:last]]) + crop_radius
        yy2 = np.minimum(y[i], y[idxs[:last]]) + crop_radius

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / (crop_size*crop_size)

        idxs = np.delete(idxs, np.concatenate(([last],
        	np.where(overlap > overlapThresh)[0])))

    return points[pick]

def calc_error(image_name, detected_coords, probs_thresh_grid):
    detected_coords = detected_coords.astype(float)

    image_label_file =  data_directory+'Crops_Labels_Anne_All/'+image_name+'_SVG_nucleus.csv'
    image_label =  open(image_label_file, 'r')
    image_label_list = [line.strip().split(';') for line in image_label][1:]
    image_label_list = np.asarray(image_label_list).reshape(-1,4)
    image_label_list = image_label_list[:, :3].astype(float)
    image_label_list = image_label_list[np.in1d(image_label_list[:,2], class_number)]
    label_coords = image_label_list[:,:2]
    label_coords = np.fliplr(label_coords)
    if len(label_coords) >=1: #if thre are lymphocytes in the image
        true_positive = np.zeros(len(probs_thresh_grid))
        false_positive = np.zeros(len(probs_thresh_grid))
        false_negative = np.zeros(len(probs_thresh_grid))
        thresh_i = 0
        for probs_thresh in probs_thresh_grid:
            if detected_coords.size > 0:
                detected_coords_over_thresh = detected_coords[np.where(detected_coords[:,2] > probs_thresh)]

                detected_coords_xy = detected_coords_over_thresh[:,:2]

                label_x_detected = cdist(label_coords, detected_coords_xy) #distance matrix

                label_ind, detected_ind =  lsa(label_x_detected)

                false_positive[thresh_i] = max(detected_coords_xy.shape[0]-detected_ind.size, 0)
                false_negative[thresh_i] = max(label_coords.shape[0]-label_ind.size, 0)
                for i in range(label_ind.size):
                    dist = np.linalg.norm(label_coords[label_ind[i], :] - detected_coords_xy[detected_ind[i], :])
                    if dist <= 10: #fixed number (not a hyper parameter)
                        true_positive[thresh_i] += 1
                    else:
                        false_negative[thresh_i] += 1
                        false_positive[thresh_i] += 1
            else:
                false_negative[thresh_i] = label_coords.shape[0]
            thresh_i += 1
        return true_positive, false_positive, false_negative

    else:
        true_positive = np.zeros(len(probs_thresh_grid))
        false_positive = np.zeros(len(probs_thresh_grid))
        false_negative = np.zeros(len(probs_thresh_grid))
        thresh_i = 0
        for probs_thresh in probs_thresh_grid:
            detected_coords_over_thresh = detected_coords[np.where(detected_coords[:,2] > probs_thresh)[0]]
            false_positive[thresh_i] = detected_coords_over_thresh.shape[0]
            thresh_i += 1
        return true_positive, false_positive, false_negative


if not os.path.exists(data_directory+'Crops_Anne_test_activation_%s' % dis_thresh):
    os.makedirs(data_directory+'Crops_Anne_test_activation_%s' % dis_thresh)


crop_size = 32
input_size = 127
crop_radius = int(crop_size/2)
grid_size = 127
min_val = 100; max_val=0
suppression_mask_all = []
image_name_all = []

speed_file = open('speed.csv', 'w+')

test_list = np.load(data_directory + 'Crops_Anne_test/test_set_%s.npy' % test_set)
for image_file in test_list:


    image =  imread(image_file)
    image = image[:,:,:3]
    image_name =  image_file.split('/')[-1]
    image = image.astype(float)

    image_datasets = LymphocytesTestImage(image, grid_size, input_size, data_transforms)

    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=6, shuffle=True, num_workers=6)
    dataset_sizes = len(image_datasets)

    for sample in dataloaders:

        inputs = sample['image_crop']
        coords = sample['coords'].numpy()

        # wrap them in Variable
        if use_gpu:
            inputs = Variable(inputs.cuda())
        else:
            inputs = Variable(inputs)
        # forward
        #class_probs, trans_preds, num_preds, out = model_ft(inputs)
        _, _, _, out = model_ft(inputs)

        if use_gpu:
            out = out.data.cpu().numpy()
        else:
            out = out.data.numpy()

        for k in range(out.shape[1]):
            feature_mask = np.zeros((image.shape[:-1]))
            multiplied_mask = np.zeros((image.shape))
            thresholded_mask = np.zeros((image.shape))
            confidence_n = np.zeros((image.shape[:-1]))
            for coord_i in range(coords.shape[0]):
                x = int(coords[coord_i, 0])
                y = int(coords[coord_i, 1])
                for i in range(input_size):
                    for j in range(input_size):
                        map_i = x + i
                        map_j = y + j
                        if confidence_n[map_i,map_j] == 0:
                            feature_mask[map_i, map_j] = max(0,out[coord_i, k, i, j])
                            multiplied_mask[map_i, map_j, :] = max(0,out[coord_i, k, i, j])*image[map_i,map_j, :] #* num_contribution
                            thresholded_mask[map_i, map_j, :] = float(out[coord_i, k, i, j]>0)*image[map_i,map_j, :]
                        confidence_n[map_i,map_j] += 1
            imsave(data_directory+'Crops_Anne_test_activation_%s/%s_feature_%s_%s_%s_%s.png' % (dis_thresh, image_name, option, config, train_set, k), feature_mask)
            imsave(data_directory+'Crops_Anne_test_activation_%s/%s_multiplied_%s_%s_%s_%s.png' % (dis_thresh, image_name, option, config, train_set, k), multiplied_mask)
            imsave(data_directory+'Crops_Anne_test_activation_%s/%s_thresholded_%s_%s_%s_%s.png' % (dis_thresh, image_name, option, config, train_set, k), thresholded_mask)

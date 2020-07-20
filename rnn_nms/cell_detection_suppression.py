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
class_number = [1.0,2.0]
data_directory = '/lila/data/fuchs/xiec/auto_nms/'
model_directory = data_directory + 'trained_models/'


mean = np.load(data_directory+'crops_train_patch_mean.npy').tolist()
std = np.load(data_directory+'crops_train_patch_std.npy').tolist()

data_transforms = transforms.Compose([
                  transforms.ToTensor(),
                  transforms.Normalize(mean, std)
                  ])

use_gpu = torch.cuda.is_available()

if option == 'cls':
    model_ft = cell_vgg(config, num_classes = 2)
    model_ft.load_state_dict(torch.load(model_directory+'trained_cell_vgg11_cls_%s.pth' % dis_thresh))
else:
    if config[:3] == 'vgg':
        model_ft = cell_vgg_cls_reg('config2')
    elif config == 'res':
        model_ft = ResNet18()
    elif config == 'dense':
        model_ft = DenseNet121()
    model_ft.load_state_dict(torch.load(model_directory+'trained_cell_vgg11_%s_%s_%s_doublefcn.pth' % (option, config, dis_thresh)))


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


if not os.path.exists(data_directory+'Crops_Anne_test_mask_suppression_%s' % dis_thresh):
    os.makedirs(data_directory+'Crops_Anne_test_mask_suppression_%s' % dis_thresh)
if not os.path.exists(data_directory+'Crops_Anne_test_all_peak_suppression_%s' % dis_thresh):
    os.makedirs(data_directory+'Crops_Anne_test_all_peak_suppression_%s' % dis_thresh)

prediction_time_file = open(data_directory+'prediction_time_suppression_%s.csv' % option, 'w+')
prediction_time_file.write('image_name,detection_time\n')


crop_size = 32
input_size = 127
crop_radius = int(crop_size/2)
grid_size = 127
min_val = 100; max_val=0
suppression_mask_all = []
image_name_all = []

speed_file = open('speed.csv', 'w+')
for image_file in glob.glob(data_directory+'Crops_Anne_test/*.png'):


    image =  imread(image_file)
    image = image[:,:,:3]
    image_name =  image_file.split('/')[-1]
    image = image.astype(float)

    image_datasets = LymphocytesTestImage(image, grid_size, input_size, data_transforms)

    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=36, shuffle=True, num_workers=6)
    dataset_sizes = len(image_datasets)


    confidence_mask = np.zeros((image.shape[:-1]))
    confidence_n = np.zeros((image.shape[:-1]))
    since = time.time()
    for sample in dataloaders:

        inputs = sample['image_crop']
        coords = sample['coords'].numpy()

        # wrap them in Variable
        if use_gpu:
            inputs = Variable(inputs.cuda())
        else:
            inputs = Variable(inputs)
        # forward
        class_probs, trans_preds, num_preds, peak_preds, weak_peak_preds = model_ft(inputs)
        class_preds = class_probs > 0.5
        if use_gpu:
            class_probs = class_probs.data.cpu().numpy()
            class_preds = class_preds.data.cpu().numpy()
            trans_preds = trans_preds.data.cpu().numpy()
            num_preds = num_preds.data.cpu().numpy()
            peak_preds = peak_preds.data.cpu().numpy()
            weak_peak_preds = weak_peak_preds.data.cpu().numpy()
        else:
            class_probs = class_probs.data.numpy()
            class_preds = class_preds.data.numpy()
            trans_preds = trans_preds.data.numpy()
            num_preds = num_preds.data.numpy()
            peak_preds = peak_preds.data.numpy()
            weak_peak_preds = weak_peak_preds.data.numpy()

        if config == 'vggweak' or config == 'vggweakalt':
            peak_map = weak_peak_preds
        else:
            peak_map = peak_preds
        for coord_i in range(coords.shape[0]):
            x = int(coords[coord_i, 0])
            y = int(coords[coord_i, 1])
            for i in range(input_size):
                for j in range(input_size):
                    map_i = x + i
                    map_j = y + j
                    if peak_map[coord_i, 0, i, j] > 0 and confidence_n[map_i,map_j] == 0:
                        confidence_mask[map_i, map_j] = peak_map[coord_i, 0, i, j]
                    confidence_n[map_i,map_j] += 1
    time_elapsed_1 = time.time() - since

    #inverted_mask = invert(confidence_mask)\
    confidence_mask = confidence_mask > 0.5
    imsave(data_directory+'Crops_Anne_test_mask_suppression_%s/%s_mask_%s_%s.png' % (dis_thresh, image_name, option, config), confidence_mask)
#     since = time.time()
#     detected_coords = []
#     for x in range(confidence_mask.shape[0]):
#         for y in range(confidence_mask.shape[1]):
#             if confidence_mask[x,y] > 0:
#                 detected_coords.append([x,y,confidence_mask[x,y]])
#     detected_coords = np.asarray(detected_coords, dtype=float).reshape(-1,3)
#     detected_coords = non_max_suppression(detected_coords, overlapThresh, crop_size) #non_max_suppression
#     time_elapsed_2 = time.time() - since
#     print(time_elapsed_1, time_elapsed_2)
#     speed_file.write('%s,%s,\n' % (time_elapsed_1, time_elapsed_2))
#
#     suppression_mask = np.zeros((image.shape[:-1]))
#     for coords in detected_coords:
#         suppression_mask[int(coords[0]), int(coords[1])] = confidence_mask[max(int(coords[0])-2, 0):min(int(coords[0])+3, image.shape[0]), \
#                                                                            max(int(coords[1])-2, 0):min(int(coords[1])+3, image.shape[1])].sum() #real accumulated confidence map (-3 to +3 pixels)
#     inverted_peak = invert(suppression_mask)
#     imsave(data_directory+'Crops_Anne_test_all_peak_suppression_%s/%s_all_peak_%s_%s.png' % (dis_thresh, image_name, option, config), inverted_peak)
#     suppression_mask_all.append(suppression_mask)
#     image_name_all.append(image_name)
#     if suppression_mask.max() >= max_val:
#         max_val = suppression_mask.max()
#     if suppression_mask.min() <= min_val:
#         min_val = suppression_mask.min()
#
#     time_elapsed = time.time() - since
#     prediction_time_file.write(str(time_elapsed)+'\n')
# speed_file.close()
#
# probs_thresh_grid = np.linspace(0,255,101)[:-1]
# true_positive = np.zeros(len(probs_thresh_grid))
# false_positive = np.zeros(len(probs_thresh_grid))
# false_negative = np.zeros(len(probs_thresh_grid))
# for i in range(len(suppression_mask_all)):
#     suppression_mask = suppression_mask_all[i]
#     suppression_mask -= min_val
#     suppression_mask *= (255.0/(max_val-min_val))
#     detected_coords = []
#     for x in range(suppression_mask.shape[0]):
#         for y in range(suppression_mask.shape[1]):
#             if suppression_mask[x,y] > 0:
#                 detected_coords.append([x,y,suppression_mask[x,y]])
#     detected_coords = np.asarray(detected_coords, dtype=float).reshape(-1,3)
#
#     true_positive_image, false_positive_image, false_negative_image = calc_error(image_name_all[i], detected_coords, probs_thresh_grid)
#     true_positive += true_positive_image
#     false_positive +=  false_positive_image
#     false_negative += false_negative_image
#
#
# precision = np.divide(true_positive,  (true_positive + false_positive), out=np.zeros(true_positive.shape), where=(true_positive + false_positive)!=0)
# recall = np.divide(true_positive,  (true_positive + false_negative), out=np.zeros(true_positive.shape), where=(true_positive + false_negative)!=0)
# fscore = 2 * np.divide((precision * recall),  (precision + recall), out=np.zeros(true_positive.shape), where=(precision + recall)!=0)
#
# if not os.path.exists(data_directory+'detection_score'):
#     os.makedirs(data_directory+'detection_score')
# np.save(data_directory + 'detection_score/test_precision_%s_%s_%s.npy' % (option, config, dis_thresh), precision)
# np.save(data_directory + 'detection_score/test_recall_%s_%s_%s.npy' % (option, config, dis_thresh), recall)
# np.save(data_directory + 'detection_score/test_fscore_%s_%s_%s.npy' % (option, config, dis_thresh), fscore)

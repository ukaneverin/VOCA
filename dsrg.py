from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import transforms
import time
import os
from Dataset_classes import * #dataset classes defined
from models import * #models defined
import sys
import torch.nn.functional as F
from cell_transforms import *
from shutil import copyfile
from scipy.misc import imread,imsave,toimage
from PIL import Image
from scipy.ndimage import label as cc_label

option = sys.argv[1]
dis_thresh = sys.argv[2]
loss_option = sys.argv[3]
train_set = sys.argv[4]

data_directory = '/lila/data/fuchs/xiec/dsrg/'
model_directory = data_directory + 'trained_models/'

num_classes = 2 #including background class

def get_mean_and_std(batch_size, dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    mean = torch.zeros(3).cuda()
    std = torch.zeros(3).cuda()
    image_dimension_x = 0.0
    image_dimension_y = 0.0
    print('==> Computing mean and std..')
    for sample in dataloader:
        inputs = sample['image']
        inputs = inputs.cuda()
        image_dimension_x = inputs.shape[2]
        image_dimension_y = inputs.shape[3]

        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean(1).mean(1).sum()
            non_reduced_mean = inputs[:,i,:,:].mean(1,keepdim=True).mean(2,keepdim=True).expand_as(inputs[:,i,:,:])
            std[i] += (inputs[:,i,:,:] - non_reduced_mean).pow(2).sum()

    std.div_(image_dimension_x*image_dimension_y*len(dataset)-1).pow_(0.5)
    mean.div_(len(dataset))
    print(mean.cpu(), std.cpu())
    return mean.cpu(), std.cpu()
#function to calculate the mean and ste

def calc_sample_size(batch_size, dataset):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    sample_sizes = torch.zeros(num_classes)
    for sample in dataloader:
        task_map = sample['task_map']
        sample_sizes[0] += (task_map==0).sum().item()
        sample_sizes[1] += (task_map!=0).sum().item()
    return sample_sizes


if option == 'calc_ms':
    mean, std = get_mean_and_std(128*8, LymphocytesTrainImage(data_directory+'Crops_Anne_train/train_label_file_%s.csv' % train_set,
                                        data_directory+'Crops_Anne_train',
                                        data_directory+'Crops_Anne_train/dis_thresh_%s' % dis_thresh,
                                        ToTensor()))
    np.save(data_directory+'crops_train_patch_mean_%s.npy' % train_set, mean)
    np.save(data_directory+'crops_train_patch_std_%s.npy' % train_set, std)
    sys.exit()
else:
    mean = torch.FloatTensor(np.load(data_directory+'crops_train_patch_mean_%s.npy' % train_set))
    std = torch.FloatTensor(np.load(data_directory+'crops_train_patch_std_%s.npy' % train_set))
    #sample_sizes = torch.FloatTensor(np.load(data_directory+'crops_train_patch_sizes_%s_%s.npy' % (dis_thresh, train_set)))
    print(mean)
    print(std)
    #print(sample_sizes)




data_transforms = {
    'train': transforms.Compose([
        #can add more augmentations here for training data
        RandomVerticalFlip(),
        RandomHorizontalFlip(),
        RandomRotate(),
        ToTensor(),
        Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        RandomVerticalFlip(),
        RandomHorizontalFlip(),
        RandomRotate(),
        ToTensor(),
        Normalize(mean, std)
    ]),
    'test': transforms.Compose([
        RandomVerticalFlip(),
        RandomHorizontalFlip(),
        RandomRotate(),
        ToTensor(),
        Normalize(mean, std)
    ]),
}

use_gpu = torch.cuda.is_available()

######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.

# def joint_loss(labels, class_preds, weighted=True):
#     out = (labels+0.3)*(class_preds-labels)**2
#     #out = (input-target)**2
#     loss = out.sum() # or sum over whatever dimensions
#     loss = loss/(class_preds.data.shape[0]*class_preds.data.shape[1])
#     return loss

def joint_loss(labels, class_preds, sample_sizes, weighted=True):
    """
    labels: size(batch_size, 1)
    trans_targets: size(batch_size, 2)
    """
    pos = labels > 0  # [N,#anchors]

    neg = labels == 0

    if weighted == False:
        return F.binary_cross_entropy(class_preds, labels)
    else:
        cls_weights = sample_sizes.sum()/sample_sizes
        ce_weights = torch.zeros(labels.data.size())
        ce_weights[neg] = cls_weights[0]
        ce_weights[pos] = cls_weights[1]
        if use_gpu:
            ce_weights = Variable(ce_weights).cuda()
        else:
            ce_weights = Variable(ce_weights)

        return F.binary_cross_entropy(class_preds, labels, weight=ce_weights)


def focal_loss(labels, class_preds, trans_targets, trans_preds, sample_sizes):
    """
    labels: size(batch_size, 1)
    trans_targets: size(batch_size, 2)
    """
    """
    Regression smooth_l1_loss
    """
    pos = labels > 0  # [N,#anchors]
    num_pos = pos.data.long().sum()

    neg = labels == 0
    trans_preds[neg] = trans_targets[neg]

    """
    Classification focal loss
    """
    #class_weights =
    cls_weights = sample_sizes.sum()/sample_sizes
    ce_weights = torch.zeros(labels.data.size())
    if use_gpu:
        ce_weights = Variable(ce_weights).cuda()
    else:
        ce_weights = Variable(ce_weights)
    ce_weights[neg] = cls_weights[0]
    ce_weights[pos] = cls_weights[1]

    gamma = int(loss_option[-1])

    pt = class_preds*labels + (1-class_preds)*(1-labels)         # pt = p if t = 1 else 1-p
    w = ce_weights * (1-pt).pow(gamma)

    if num_pos > 0:
        return num_pos, F.binary_cross_entropy(class_preds, labels, w), F.smooth_l1_loss(trans_preds, trans_targets, size_average=False)/num_pos
    else:
        return num_pos, F.binary_cross_entropy(class_preds, labels, w), F.smooth_l1_loss(trans_preds, trans_targets, size_average=False)



def train_model(model, optimizer, epoch):
    """
    Load the current training maps
    """
    if epoch == 0:
        image_datasets = {x: LymphocytesTrainImage(data_directory+'Crops_Anne_%s/%s_label_file_%s.csv' % (x, x, train_set),
                                                data_directory+'Crops_Anne_%s' % x,
                                                data_directory+'Crops_Anne_%s/dis_thresh_%s' % (x, dis_thresh),
                                                data_transforms[x])
                          for x in ['train']}

        dataloders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8,
                                                     shuffle=True, num_workers=8)
                      for x in ['train']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}
        '''
        Calculate sample_sizes
        '''
        sample_sizes = calc_sample_size(128*8, LymphocytesTrainImage(data_directory+'Crops_Anne_train/train_label_file_%s.csv' % train_set,
                                            data_directory+'Crops_Anne_train',
                                            data_directory+'Crops_Anne_train/dis_thresh_%s' % dis_thresh,
                                            ToTensor()))
    else:
        image_datasets = {x: LymphocytesTrainImage(data_directory+'Crops_Anne_%s/%s_label_file_%s.csv' % (x, x, train_set),
                                                data_directory+'Crops_Anne_%s' % x,
                                                data_directory+'Crops_Anne_%s/dis_thresh_%s/epoch_%s/mask' % (x, dis_thresh, epoch),
                                                data_transforms[x])
                          for x in ['train']}

        dataloders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8,
                                                     shuffle=True, num_workers=8)
                      for x in ['train']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}
        '''
        Calculate sample_sizes
        '''
        sample_sizes = calc_sample_size(128*8, LymphocytesTrainImage(data_directory+'Crops_Anne_train/train_label_file_%s.csv' % train_set,
                                            data_directory+'Crops_Anne_train',
                                            data_directory+'Crops_Anne_train/dis_thresh_%s/epoch_%s/mask' % (dis_thresh, epoch),
                                            ToTensor()))
    # Each epoch has a training and validation phase
    for phase in ['train']:
        if phase == 'train':
            #scheduler.step()
            model.train(True)  # Set model to training mode
        else:
            model.train(False)  # Set model to evaluate mode

        running_loss = 0.0

        # Iterate over data.
        for sample in dataloders[phase]:
            inputs = sample['image']
            task_map = sample['task_map']
            label_map =  torch.unsqueeze(task_map, -1).permute(0,3,1,2).type(torch.FloatTensor)
            # wrap them in Variable
            if use_gpu:
                inputs = Variable(inputs.cuda())
                label_map = Variable(label_map.cuda())
            else:
                inputs, label_map = Variable(inputs), Variable(label_map)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            class_preds = model(inputs)
            if loss_option == 'focal2' or loss_option == 'focal5':
                num_pos, cls_loss, reg_loss = focal_loss(label_map, class_preds, trans_map, trans_preds, sample_sizes)
            elif loss_option == 'weighted':
                cls_loss = joint_loss(label_map, class_preds, sample_sizes, weighted=True)
            else:
                cls_loss = joint_loss(label_map, class_preds, sample_sizes, weighted=False)
            # backward + optimize only if in training phase
            loss = cls_loss
            if phase == 'train':
                loss.backward()
                optimizer.step()

            # statistics
            # balanced accuracy
            running_loss += loss.item()*128
            if phase == 'train':
                print(loss.item())

        epoch_loss = running_loss / dataset_sizes[phase]
        print('{} Loss: {:.4f} '.format(
            phase, epoch_loss))
    print()

    return model


def expansion(model_ft, epoch, supervision_thresh = 0.9):
    data_transforms_prediction = transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean, std)
                                 ])

    use_gpu = torch.cuda.is_available()

    model_ft.train(False)

    crop_size = 32
    input_size = 127
    crop_radius = int(crop_size/2)
    grid_size = 127

    train_list = np.load(data_directory + 'Crops_Anne_train/train_set_%s.npy' % train_set)

    epoch += 1
    if not os.path.exists(data_directory+'Crops_Anne_train/dis_thresh_%s/epoch_%s/mask' % (dis_thresh, epoch)):
        os.makedirs(data_directory+'Crops_Anne_train/dis_thresh_%s/epoch_%s/mask' % (dis_thresh, epoch))
    if not os.path.exists(data_directory+'Crops_Anne_train/dis_thresh_%s/epoch_%s/mask_show' % (dis_thresh, epoch)):
        os.makedirs(data_directory+'Crops_Anne_train/dis_thresh_%s/epoch_%s/mask_show' % (dis_thresh, epoch))

    for image_file in train_list:

        image =  imread(image_file)
        image = image[:,:,:3]
        image_name =  image_file.split('/')[-1]
        image = image.astype(float)

        image_datasets = LymphocytesTestImage(image, grid_size, input_size, data_transforms_prediction)

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
            class_probs = model_ft(inputs)

            if use_gpu:
                class_probs = class_probs.data.cpu().numpy()

            else:
                class_probs = class_probs.data.numpy()

            for coord_i in range(coords.shape[0]):
                x = int(coords[coord_i, 0])
                y = int(coords[coord_i, 1])
                for i in range(input_size):
                    for j in range(input_size):
                        map_i = x + i
                        map_j = y + j
                        if confidence_n[map_i,map_j] == 0:
                            confidence_mask[map_i, map_j] = class_probs[coord_i, 0, i, j]
                        confidence_n[map_i,map_j] += 1

        '''
        Find the connected components that contains the seeds
        '''
        confidence_mask = confidence_mask > supervision_thresh
        labeled_array, num_features = cc_label(confidence_mask)
        labeled_points = np.load('/lila/data/fuchs/xiec/dsrg/Crops_Anne_train/dis_thresh_0.0/%s_task_map.npy' % image_name)

        for n in range(num_features): # for each connected component
            component_label = n+1
            component_coords_x, component_coords_y =  np.where(labeled_array == component_label)
            is_foreground = False
            for x,y in zip(component_coords_x, component_coords_y):
                if labeled_points[x,y] == 1:
                    is_foreground =  True
            if not is_foreground:
                for x,y in zip(component_coords_x, component_coords_y):
                    confidence_mask[x,y] = 0
        confidence_mask = np.maximum(confidence_mask,labeled_points)

        np.save(data_directory+'Crops_Anne_train/dis_thresh_%s/epoch_%s/mask/%s_task_map.npy' % (dis_thresh, epoch, image_name), confidence_mask)
        imsave(data_directory+'Crops_Anne_train/dis_thresh_%s/epoch_%s/mask_show/%s_epoch_%s.png' % (dis_thresh, epoch, image_name, epoch), confidence_mask)


#model_ft = ResNet18()
model_ft = cell_vgg_cls_reg('VGG11')

if option == 'train':

    if use_gpu:
        model_ft = model_ft.cuda()

    optimizer_ft = optim.SGD(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=0.001, momentum=0.9)
    num_epochs = 10
    for epoch in range(num_epochs):
        for i in range(3):
            model_ft = train_model(model_ft, optimizer_ft, epoch)
        expansion(model_ft, epoch, supervision_thresh = 0.9)

    torch.save(model_ft.state_dict(), model_directory+'trained_cell_vgg11_joint%s_%s_%s.pth' % (loss_option, dis_thresh, train_set))

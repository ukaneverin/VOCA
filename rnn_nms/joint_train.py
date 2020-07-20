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
from optimize import *
import sys
import torch.nn.functional as F
from cell_transforms import *
from shutil import copyfile
from scipy.optimize import linear_sum_assignment as lsa
from scipy.spatial.distance import cdist

option = sys.argv[1]
dis_thresh = sys.argv[2]
loss_option = sys.argv[3]
config = sys.argv[4]

data_directory = '/lila/data/fuchs/xiec/auto_nms/'
model_directory = data_directory + 'trained_models/'

num_classes = 2 #including background class

def get_mean_and_std(batch_size, dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    sample_sizes = torch.zeros(num_classes)
    mean = torch.zeros(3).cuda()
    std = torch.zeros(3).cuda()
    print('==> Computing mean and std..')
    for sample in dataloader:
        inputs = sample['image']
        inputs = inputs.cuda()
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean(1).mean(1).sum()
            non_reduced_mean = inputs[:,i,:,:].mean(1,keepdim=True).mean(2,keepdim=True).expand_as(inputs[:,i,:,:])
            std[i] += ((inputs[:,i,:,:] - non_reduced_mean).pow(2).sum(1).sum(1)).div(inputs.shape[2]*inputs.shape[3]-1).pow(0.5).sum()

        task_map = sample['task_map']
        for n in range(num_classes):
            sample_sizes[n] += (task_map[:,:,:,0]==n).sum()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean.cpu(), std.cpu(), sample_sizes
#function to calculate the mean and ste


if option == 'calc_ms':
    mean, std, sample_sizes = get_mean_and_std(128*8, LymphocytesTrainImage(data_directory+'Crops_Anne_train/train_label_file.csv',
                                        data_directory+'Crops_Anne_train',
                                        data_directory+'Crops_Anne_train/dis_thresh_%s' % dis_thresh,
                                        ToTensor()))
    np.save(data_directory+'crops_train_patch_mean.npy', mean)
    np.save(data_directory+'crops_train_patch_std.npy', std)
    np.save(data_directory+'crops_train_patch_sizes_%s.npy' % dis_thresh, sample_sizes)
    sys.exit()
else:
    mean = np.load(data_directory+'crops_train_patch_mean.npy').tolist()
    std = np.load(data_directory+'crops_train_patch_std.npy').tolist()
    sample_sizes = np.load(data_directory+'crops_train_patch_sizes_%s.npy' % dis_thresh).tolist()
    print(mean)
    print(std)
    print(sample_sizes)




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

# image_datasets = {x: LymphocytesDataset(data_directory+'relabeled/%s_label_file_%s.csv' % (x, dis_thresh),
#                                         data_directory+'crops_%s_patch/' % x,
#                                         data_transforms[x])
#                   for x in ['train', 'val', 'test']}

image_datasets = {x: LymphocytesTrainImage(data_directory+'Crops_Anne_%s/%s_label_file.csv' % (x, x),
                                        data_directory+'Crops_Anne_%s' % x,
                                        data_directory+'Crops_Anne_%s/dis_thresh_%s' % (x, dis_thresh),
                                        data_transforms[x])
                  for x in ['train', 'val', 'test']}

dataloders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8,
                                             shuffle=True, num_workers=8)
              for x in ['train', 'val', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

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
def joint_loss(labels, class_preds, trans_targets, trans_preds, cell_number_targets, cell_number_preds, peak_targets, peak_preds, boost_mask, balanced=False):
    """
    labels: size(batch_size, 1)
    trans_targets: size(batch_size, 2)
    """
    pos = labels > 0  # [N,#anchors]
    num_pos = pos.sum().item()

    neg = labels == 0
    trans_preds[neg.expand_as(trans_preds)] = trans_targets[neg.expand_as(trans_preds)]
    cell_number_preds[neg] = cell_number_targets[neg]

    cls_weights = sample_sizes.sum()/sample_sizes
    ce_weights = torch.zeros(labels.data.size())
    ce_weights[neg] = cls_weights[0]
    ce_weights[pos] = cls_weights[1]
    if use_gpu:
        ce_weights = Variable(ce_weights).cuda()
        boost_mask = Variable(boost_mask).cuda()
    else:
        ce_weights = Variable(ce_weights)
        boost_mask = Variable(boost_mask)


    #reg loss weighted by Num_Loss
    reg_loss = F.smooth_l1_loss(trans_preds, trans_targets, size_average=False, reduce=False)
    if balanced:
        reg_loss = reg_loss * cell_number_targets
    reg_loss = reg_loss.sum()

    #hungarian peak loss
    target_points = []; pred_points = []
    assignment_map = np.zeros(peak_targets.data.size())
    for i in range(peak_targets.data.size()[0]):
        target_plane = peak_targets.data[i,0,:,:]
        target_points = []
        for x in range(target_plane.shape[0]):
            for y in range(target_plane.shape[1]):
                target_points.append([x,y,target_plane[x,y]])
        target_points = np.asarray(target_points)
        target_points = target_points[np.where(target_points[:,-1] == 1)]

        pred_plane = peak_preds.data[i,0,:,:]
        pred_points = []
        for x in range(pred_plane.shape[0]):
            for y in range(pred_plane.shape[1]):
                pred_points.append([x,y,pred_plane[x,y]])
        pred_points = np.asarray(pred_points)
        pred_points = pred_points[np.where(pred_points[:,-1] > 0.5)] #maybe perceptrons are better


        label_x_detected = cdist(target_points[:,:2], pred_points[:,:2]) #distance matrix
        long_dist_mask = label_x_detected > 7
        short_dist_mask = label_x_detected <= 7
        prob_weight = 1 - np.tile(pred_points[:,-1], ( label_x_detected.shape[0], 1))
        prob_weight = np.ma.array(prob_weight, mask = long_dist_mask)
        label_x_detected = label_x_detected * prob_weight
        label_x_detected = label_x_detected.data
        target_ind,pred_ind =  lsa(label_x_detected)

        assignment_plane = target_plane
        for j in range(target_ind.size):
            dist = np.linalg.norm(target_points[target_ind[j], :2] - pred_points[pred_ind[j], :2])
            if dist <= 7: #fixed number (not a hyper parameter)
                assignment_plane[int(pred_points[pred_ind[j], 0]), int(pred_points[pred_ind[j], 1])] = 1
                assignment_plane[int(target_points[target_ind[j], 0]), int(target_points[target_ind[j], 1])] = 0

        assignment_map[i,0,:,:] = assignment_plane

    assignment_map = torch.FloatTensor(assignment_map)
    if use_gpu:
        assignment_map = Variable(assignment_map).cuda()
    else:
        assignment_map = Variable(assignment_map)

    hungarian_loss = F.binary_cross_entropy(peak_preds, assignment_map)

    if num_pos > 0:
        ce_weights = ce_weights * boost_mask

        return num_pos, F.binary_cross_entropy(class_preds, labels, weight=ce_weights), reg_loss/num_pos, F.smooth_l1_loss(cell_number_preds, cell_number_targets, size_average=False)/num_pos, hungarian_loss

    else:
        ce_weights = ce_weights * boost_mask
        return num_pos, F.binary_cross_entropy(class_preds, labels, weight=ce_weights), reg_loss, F.smooth_l1_loss(cell_number_preds, cell_number_targets, size_average=False), hungarian_loss

def focal_loss(labels, class_preds, trans_targets, trans_preds):
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
    cell_number_preds[neg] = cell_number_targets[neg]

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
        return num_pos, F.binary_cross_entropy(class_preds, labels, w), F.smooth_l1_loss(trans_preds, trans_targets, size_average=False)/num_pos, F.smooth_l1_loss(cell_number_preds, cell_number_targets, size_average=False)/num_pos
    else:
        return num_pos, F.binary_cross_entropy(class_preds, labels, w), F.smooth_l1_loss(trans_preds, trans_targets, size_average=False), F.smooth_l1_loss(cell_number_preds, cell_number_targets, size_average=False)



def train_model(model, optimizer, scheduler, num_epochs=10):
    since = time.time()

    best_model_wts = model.state_dict()
    best_loss = 0.0
    best_reg_loss = 0.0
    best_num_loss = 0.0
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_cls_loss = 0.0
            running_reg_loss = 0.0
            running_num_loss = 0.0
            running_corrects = np.zeros(num_classes)
            running_sizes = np.zeros(num_classes)

            # Iterate over data.
            num_pos_total = 0
            for sample in dataloders[phase]:
                inputs = sample['image']
                task_map = sample['task_map']
                peak_map = sample['peak_map']
                label_map =  torch.unsqueeze(task_map[:,:,:,0], -1).permute(0,3,1,2).type(torch.FloatTensor)
                trans_map =  task_map[:,:,:,1:-1].permute(0,3,1,2).type(torch.FloatTensor)
                cell_number_map =  torch.unsqueeze(task_map[:,:,:,-1], -1).permute(0,3,1,2).type(torch.FloatTensor)
                peak_map = torch.unsqueeze(peak_map, -1).permute(0,3,1,2).type(torch.FloatTensor)
                if phase == 'train' and option[:-1] == 'boost':
                    boost_mask = sample['boost_mask']
                    boost_mask = torch.unsqueeze(boost_mask,-1).permute(0,3,1,2).type(torch.FloatTensor)
                else:
                    boost_mask = torch.ones(label_map.size())
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    label_map = Variable(label_map.cuda())
                    trans_map = Variable(trans_map.cuda())
                    cell_number_map = Variable(cell_number_map.cuda())
                    peak_map = Variable(peak_map.cuda())
                else:
                    inputs, label_map, trans_map, cell_number_map, peak_map = Variable(inputs), Variable(label_map), Variable(trans_map), Variable(cell_number_map), Variable(peak_map)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                class_preds, trans_preds, num_preds, peak_preds, weak_peak_preds = model(inputs)
                if loss_option == 'focal2' or loss_option == 'focal5':
                    num_pos, cls_loss, reg_loss, num_loss = focal_loss(label_map, class_preds, trans_map, trans_preds, cell_number_map, num_preds)
                elif loss_option == 'weighted' or loss_option == 'balanced':
                    if config == 'vgg' or config == 'vggalt':
                        if loss_option == 'balanced':
                            num_pos, cls_loss, reg_loss, num_loss, peak_loss = joint_loss(label_map, class_preds, trans_map, trans_preds, cell_number_map, num_preds, peak_map, peak_preds, boost_mask, balanced=True)
                        else:
                            num_pos, cls_loss, reg_loss, num_loss, peak_loss = joint_loss(label_map, class_preds, trans_map, trans_preds, cell_number_map, num_preds, peak_map, peak_preds, boost_mask, balanced=False)
                    elif config == 'vggweak' or config == 'vggweakalt':
                        if loss_option == 'balanced':
                            num_pos, cls_loss, reg_loss, num_loss, peak_loss = joint_loss(label_map, class_preds, trans_map, trans_preds, cell_number_map, num_preds, peak_map, weak_peak_preds, boost_mask, balanced=True)
                        else:
                            num_pos, cls_loss, reg_loss, num_loss, peak_loss = joint_loss(label_map, class_preds, trans_map, trans_preds, cell_number_map, num_preds, peak_map, weak_peak_preds, boost_mask, balanced=False)
                else:
                    num_pos, cls_loss, reg_loss, num_loss = joint_loss(label_map, class_preds, trans_map, trans_preds, cell_number_map, num_preds, weighted=False)
                num_pos_total += num_pos
                # backward + optimize only if in training phase

                loss = peak_loss


                if phase == 'train':
                    if config[-3:] != 'alt':
                        loss.backward()
                        optimizer.step()
                    else:
                        # #training low
                        optimizer_ft_low = optim.SGD([
                            {'params': model.features.parameters()},
                            {'params': model.classifier.parameters()},
                            {'params': model.regressor.parameters()},
                            {'params': model.quantifier.parameters()}
                        ], lr=0.001, momentum=0.9)
                        optimizer_ft_low.zero_grad()

                        #trainning high
                        if config == 'vggalt':
                            optimizer_ft_high = optim.SGD([
                                {'params': model.peaker.parameters()}
                            ], lr=0.001, momentum=0.9)
                        elif config == 'vggweakalt':
                            optimizer_ft_high = optim.SGD([
                                {'params': model.weak_peaker.parameters()}
                            ], lr=0.001, momentum=0.9)
                        optimizer_ft_high.zero_grad()

                        loss.backward()
                        optimizer_ft_low.step()
                        optimizer_ft_high.step()



                # statistics
                label_pred = class_preds.data >= 0.5
                # balanced accuracy

                for c in range(num_classes): #need modification for multi-class
                    running_sizes[c] += torch.sum(label_map.data==c)
                    running_corrects[c] += torch.sum(label_pred[label_map.data==c] == c)

                running_reg_loss += reg_loss.item()*num_pos
                running_num_loss += num_loss.item()*num_pos
                running_cls_loss += cls_loss.item()*128
                if phase == 'train':
                    print(cls_loss.item(), reg_loss.item(), num_loss.item(), peak_loss.item())

            epoch_cls_loss = running_cls_loss / dataset_sizes[phase]
            epoch_reg_loss = running_reg_loss / num_pos_total
            epoch_num_loss = running_num_loss / num_pos_total
            epoch_acc = (running_corrects / running_sizes).mean()
            epoch_loss = 1 - epoch_acc + epoch_reg_loss + epoch_num_loss
            print('{} Cls_Loss: {:.4f} Reg_Loss: {:.4f} Num_Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_cls_loss, epoch_reg_loss, epoch_num_loss, epoch_acc))

            # deep copy the model
            if phase == 'val':
                if epoch == 0:
                    best_loss = epoch_loss
                    best_reg_loss = epoch_reg_loss
                    best_num_loss = epoch_num_loss
                    best_model_wts = model.state_dict()
                    best_acc = epoch_acc
                else:
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        best_reg_loss = epoch_reg_loss
                        best_num_loss = epoch_num_loss
                        best_model_wts = model.state_dict()
                        best_acc = epoch_acc
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val reg_Loss: {:4f}'.format(best_reg_loss))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if config == 'vgg' or config== 'vggalt' or config == 'vggweak' or config == 'vggweakalt':
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

if option == 'train':

    if use_gpu:
        model_ft = model_ft.cuda()


    optimizer_ft = optim.SGD(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)

    model_ft = train_model(model_ft, optimizer_ft, exp_lr_scheduler,
                           num_epochs=10)
    torch.save(model_ft.state_dict(), model_directory+'trained_cell_vgg11_joint%s_%s_%s.pth' % (loss_option, config, dis_thresh))


if option[:-1] == 'boost':

    if option == 'boost1':
        model_ft.load_state_dict(torch.load(model_directory+'trained_cell_vgg11_joint%s_%s.pth' % (loss_option, dis_thresh)))
    else:
        model_ft.load_state_dict(torch.load(model_directory+'trained_cell_vgg11_joint%sboosted%s_%s.pth' % (loss_option, int(option[-1])-1, dis_thresh)))

    if use_gpu:
        model_ft = model_ft.cuda()

    model_ft.train(False)


    grid_size = 127
    input_size = 127

    data_transforms_test = transforms.Compose([
                      transforms.ToTensor(),
                      transforms.Normalize(mean, std)
                      ])

    for image_file in glob.glob(data_directory+'Crops_Anne_train/*.png'):
        print('Testing %s' % image_file)
        image =  imread(image_file)
        image = image[:,:,:3]
        image_name =  image_file.split('/')[-1]
        image = image.astype(float)

        image_datasets = LymphocytesTestImage(image, grid_size, input_size, data_transforms_test)

        dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=6, shuffle=True, num_workers=6)


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
            class_preds = class_probs > 0.5
            if use_gpu:
                class_preds = class_preds.data.cpu().numpy()
            else:
                class_preds = class_preds.data.numpy()

            input_size = 127
            for coord_i in range(coords.shape[0]):
                x = int(coords[coord_i, 0])
                y = int(coords[coord_i, 1])
                for i in range(input_size):
                    for j in range(input_size):
                        map_i = x + i
                        map_j = y + j
                        if class_preds[coord_i, 0, i, j] == 1 and confidence_n[map_i,map_j] == 0:
                            if map_i>=0 and map_i<700 and map_j>=0 and map_j<700:
                                confidence_mask[map_i, map_j] = 1
                        confidence_n[map_i,map_j] += 1

        task_map = np.load(data_directory + 'Crops_Anne_train/dis_thresh_%s/%s_task_map.npy' % (dis_thresh, image_name))
        wrong_mask = confidence_mask != task_map[:,:,0]
        print('wrong pixels: %s' % wrong_mask.sum())
        wrong_mask = 2.0 + wrong_mask.astype(float)
        np.save(data_directory+'Crops_Anne_train/dis_thresh_%s/%s_boost_mask.npy' % (dis_thresh, image_name), wrong_mask)

    model_ft.train(True)

    image_datasets = {x: LymphocytesTrainImage(data_directory+'Crops_Anne_%s/%s_label_file.csv' % (x, x),
                                            data_directory+'Crops_Anne_%s' % x,
                                            data_directory+'Crops_Anne_%s/dis_thresh_%s' % (x, dis_thresh),
                                            data_transforms[x])
                      for x in ['val', 'test']}

    image_datasets['train'] = LymphocytesBoostImage(data_directory+'Crops_Anne_train/train_label_file.csv',
                                            data_directory+'Crops_Anne_train',
                                            data_directory+'Crops_Anne_train/dis_thresh_%s' % dis_thresh,
                                            data_transforms['train'])

    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8,
                                                 shuffle=True, num_workers=8)
                  for x in ['train', 'val', 'test']}
    optimizer_ft = optim.SGD(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)

    model_ft = train_model(model_ft, optimizer_ft, exp_lr_scheduler,
                           num_epochs=10)
    torch.save(model_ft.state_dict(), model_directory+'trained_cell_vgg11_joint%sboosted%s_%s.pth' % (loss_option, option[-1], dis_thresh))

if option == 'test':

    model_ft.load_state_dict(torch.load(model_directory+'trained_cell_vgg11_joint_%s.pth' % dis_thresh))

    if use_gpu:
        model_ft = model_ft.cuda()
    model_ft.train(False)
    running_cls_loss = 0.0
    running_reg_loss = 0.0
    running_corrects = 0
    num_pos_total = 0.0
    for sample in dataloders['test']:
        inputs = sample['image']
        labels = sample['class_label']
        trans_targets = sample['trans_targets'].type(torch.FloatTensor)

        # wrap them in Variable
        if use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
            trans_targets = Variable(trans_targets.cuda())
        else:
            inputs, labels, trans_targets = Variable(inputs), Variable(labels), Variable(trans_targets)

        class_preds, trans_preds = model_ft(inputs)
        num_pos, cls_loss, reg_loss = joint_loss(labels, class_preds, trans_targets, trans_preds)
        num_pos_total += num_pos
        # statistics
        running_corrects += torch.sum(class_preds == labels.data)
        running_reg_loss += reg_loss.data[0]*num_pos
        running_cls_loss += cls_loss

    epoch_cls_loss = running_cls_loss / dataset_sizes[phase]
    epoch_reg_loss = running_reg_loss / num_pos_total
    epoch_acc = running_corrects / dataset_sizes[phase]
    epoch_loss = epoch_cls_loss + epoch_reg_loss
    print('test:: Cls_Loss: %s Reg_Loss: %s Acc: %s TotalLoss: %s' % (epoch_cls_loss, epoch_reg_loss, epoch_acc, epoch_loss))

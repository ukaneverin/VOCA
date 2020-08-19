from __future__ import print_function, division
import math
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torchvision import transforms
import time
import os
from Dataset_classes import *
from models.res import VOCA_Res
import torch.nn.functional as F
from voca_transforms import *
from pdb import set_trace
from utils import non_max_suppression, remove_non_assigned, get_mean_and_std
import argparse

parser = argparse.ArgumentParser(description='VOCA training')
parser.add_argument('--dataset', default='lung_tt', type=str)
parser.add_argument('--cv', default=0, type=int, help='which cross validation fold')
parser.add_argument('--output_path', default='/lila/data/fuchs/xiec/results/VOCA', type=str, help='working directory')

parser.add_argument('--num_classes', default=2, type=int, help='how many cell types')
parser.add_argument('--input_size', default=127, type=int, help='input image size to model')
parser.add_argument('--r', default=12, type=int, help='disk size')
parser.add_argument('--num_workers', default=8, type=int, help='how many workers for dataloader')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--batch_size', default=8, type=int, help='bactch size')
parser.add_argument('--n_epochs', default=10, type=int)
parser.add_argument('--lambda1', default=1.0, type=float, help='weight for loc_loss')
parser.add_argument('--lambda2', default=1.0, type=float, help='weight for wt_loss')
parser.add_argument('--lambda3', default=1.0, type=float, help='weight for thresh_loss')
parser.add_argument('--metric', default='acc', type=str,
                    help='acc, f1; this tells whether to compute the \
                    f1 score for every batch, and save best model based on which metric')
parser.add_argument('--nms_r', default=6, type=int, help='the distance threshold for non-max suppresion')

def main():
    global args
    global device
    args = parser.parse_args()

    model_directory = os.path.join(args.output_path, args.dataset, 'trained_models/')
    if not os.path.exists(model_directory):
        os.mkdir(model_directory)

    # load or calculate the mean and standard deviation of rgb channels of training image crops,
    # and ratio of positive/negative pixels
    try:
        mean = torch.FloatTensor(np.load(os.path.join(args.output_path, args.dataset, 'split/CV_%s_mean.npy' % args.cv)))
        std = torch.FloatTensor(np.load(os.path.join(args.output_path, args.dataset, 'split/CV_%s_std.npy' % args.cv)))
        sample_sizes = torch.FloatTensor(
            np.load(os.path.join(args.output_path, args.dataset, 'split/CV_%s_samplesizes_r%s.npy' % (args.cv, args.r))))
        print(mean)
        print(std)
        print(sample_sizes)
    except:
        mean, std, sample_sizes = get_mean_and_std(128 * 8, TrainImagesDataset(
            os.path.join(args.output_path, args.dataset, 'images/'), #image_folder
            os.path.join(args.output_path, args.dataset, 'task_map/'), #map_folder
            os.path.join(args.output_path, args.dataset, 'gt_map/'),  # gt_folder
            os.path.join(args.output_path, args.dataset, 'split/'), #split_folder
            args.cv, # cv
            args.input_size, # crop_size
            args.r,
            'train',
            ToTensor()), args.num_classes)
        np.save(os.path.join(args.output_path, args.dataset, 'split/CV_%s_mean.npy' % args.cv), mean)
        np.save(os.path.join(args.output_path, args.dataset, 'split/CV_%s_std.npy' % args.cv), std)
        np.save(os.path.join(args.output_path, args.dataset, 'split/CV_%s_samplesizes_r%s.npy' % (args.cv, args.r)),  sample_sizes)

    data_transforms = {
        'train': transforms.Compose([
            #can add more augmentations here for training data
            #but need to modify the voca_transforms script. augmentation sometimes need to apply on the task maps
            RandomVerticalFlip(),
            RandomHorizontalFlip(),
            RandomRotate(),
            ToTensor(),
            Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            ToTensor(),
            Normalize(mean, std)
        ])
    }

    image_datasets = {phase: TrainImagesDataset(
                            os.path.join(args.output_path, args.dataset, 'images/'), #image_folder
                            os.path.join(args.output_path, args.dataset, 'task_map/'), #map_folder
                            os.path.join(args.output_path, args.dataset, 'gt_map/'),  # gt_folder
                            os.path.join(args.output_path, args.dataset, 'split/'), #split_folder
                            args.cv, # cv
                            args.input_size, # crop_size
                            args.r,
                            phase,
                            data_transforms[phase])
                        for phase in ['train', 'val']}

    dataloaders = {phase: torch.utils.data.DataLoader(image_datasets[phase], batch_size=args.batch_size,
                                                 shuffle=True, num_workers=args.num_workers)
                  for phase in ['train', 'val']}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = VOCA_Res().to(device)

    optimizer_ft = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=args.momentum)

    # Decay LR by a factor of 0.1 every 3 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=int((args.n_epochs - 1)/3), gamma=0.1)

    model = train_model(model, optimizer_ft, exp_lr_scheduler, dataloaders, sample_sizes,
                        num_epochs=args.n_epochs)
    torch.save(model.state_dict(), model_directory+'%s_lr%s_%s_cv%s.pth' % (args.r, args.lr, args.metric, args.cv))

def train_model(model, optimizer, scheduler, dataloaders, sample_sizes, num_epochs=10):
    since = time.time()

    best_model_wts = model.state_dict()
    best_loss = 0.0
    best_reg_loss = 0.0
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs), 'lr: {}'.format(optimizer.param_groups[0]['lr']))
        print('-' * 10)
        print('cls_loss\t'
              'reg_loss\t'
              'num_loss\t'
              'f1_best\t\t'
              'f1_pred\t\t'
              'thresh_loss\t'
              'acc\t')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            f1_meter = AverageMeter()
            f1_pred_meter = AverageMeter()
            cls_loss_meter = AverageMeter()
            reg_loss_meter = AverageMeter()
            num_loss_meter = AverageMeter()
            cls_acc_meter = AccuracyMeter()
            thresh_loss_meter = AverageMeter()

            torch.cuda.empty_cache()

            for sample in dataloaders[phase]:
                inputs = sample['image']
                task_map = sample['task_map']
                gt_map = sample['gt_map']
                gt_map = gt_map.permute(0, 3, 1, 2).type(torch.FloatTensor)
                if args.num_classes == 1:
                    label_map = torch.unsqueeze(task_map[:,:,:,:args.num_classes], -1).permute(0,3,1,2).type(torch.FloatTensor)
                    trans_map = task_map[:,:,:,args.num_classes:3*args.num_classes].permute(0,3,1,2).type(torch.FloatTensor)
                    cell_number_map = torch.unsqueeze(task_map[:,:,:,3*args.num_classes:], -1).permute(0,3,1,2).type(torch.FloatTensor)
                else:
                    label_map = task_map[:, :, :, :args.num_classes].permute(0, 3, 1, 2).type(torch.FloatTensor)
                    trans_map = task_map[:, :, :, args.num_classes:3 * args.num_classes].permute(0, 3, 1, 2).type(torch.FloatTensor)
                    cell_number_map = task_map[:, :, :, 3 * args.num_classes:].permute(0, 3, 1, 2).type(torch.FloatTensor)

                inputs = inputs.to(device)
                label_map = label_map.to(device)
                trans_map = trans_map.to(device)
                cell_number_map = cell_number_map.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                torch.cuda.empty_cache()
                if phase == 'train':
                    class_preds, trans_preds, num_preds, t_preds = model(inputs)
                elif phase == 'val':
                    with torch.no_grad():
                        class_preds, trans_preds, num_preds, t_preds = model(inputs)

                """
                If args.metrics is 'f1', voca needs to learn the best threshold and save the best model in terms of f1 score.
                Calculate best batch f1 based on various threholds and choose the best as the target to learn.
                """
                if args.metric == 'f1':
                    # get the accumulated confidence map
                    voca_map = np.zeros(label_map.shape)

                    weighted_confidence_map = class_preds * num_preds
                    x_coord_matrix = np.array([np.arange(label_map.shape[-1]), ] * label_map.shape[-2])
                    y_coord_matrix = np.array([np.arange(label_map.shape[-2]), ] * label_map.shape[-1]).transpose()
                    x_map = np.around(trans_preds[:,:args.num_classes,:,:].detach().cpu().numpy() + x_coord_matrix).astype(int)
                    y_map = np.around(trans_preds[:,args.num_classes:2 * args.num_classes,:,:].detach().cpu().numpy() + y_coord_matrix).astype(int)
                    for c in range(args.num_classes): # for each cell class
                        for n in range(label_map.shape[0]): # batch size
                            x = x_map[n,c].reshape(-1)
                            y = y_map[n,c].reshape(-1)
                            valid_location_index = (x > 0) * (x < voca_map.shape[-1]) * (y > 0) * (y < voca_map.shape[-2])
                            x = x[valid_location_index]
                            y = y[valid_location_index]
                            np.add.at(voca_map[n,c], (y,x), weighted_confidence_map[n, c, y, x].detach().cpu().numpy())

                    # nms on the map
                    suppression_mask = torch.zeros(voca_map.shape)
                    for c in range(args.num_classes):  # for each cell class
                        for n in range(voca_map.shape[0]):  # batch size
                            detected_coords = [x.tolist() + [voca_map[n, c, x[0], x[1]]]
                                               for x in np.asarray(np.where(voca_map[n,c,:,:]>0)).transpose()]
                            detected_coords = np.asarray(detected_coords, dtype=float).reshape(-1, 3)
                            detected_coords = non_max_suppression(detected_coords, args.nms_r)  # non_max_suppression
                            for coords in detected_coords:
                                # real accumulated confidence map (-3 to +3 pixels)
                                suppression_mask[n, c, int(coords[0]), int(coords[1])] = voca_map[n, c,
                                                               max(int(coords[0]) - 2, 0):min(
                                                                   int(coords[0]) + 3, voca_map.shape[-2]), \
                                                               max(int(coords[1]) - 2, 0):min(
                                                                   int(coords[1]) + 3, voca_map.shape[-1])].sum()
                    #calculate the f1 scores
                    probs_thresh_grid = np.linspace(0.1, 0.85, 16)[:-1]
                    minval = 0.0
                    maxval = math.pi * args.r * args.r
                    suppression_mask -= minval
                    suppression_mask = suppression_mask / (maxval - minval)

                    true_positive, pred_num, gt_num, pair_distances = calc_f1_batch(gt_map,
                                                                              suppression_mask,
                                                                                  probs_thresh_grid, r=6)
                    precision = np.divide(true_positive, pred_num, out=np.zeros(true_positive.shape),
                                          where=(pred_num) != 0)
                    recall = np.divide(true_positive, gt_num, out=np.zeros(true_positive.shape), where=(gt_num) != 0)
                    fscore = 2 * np.divide((precision * recall), (precision + recall),
                                           out=np.zeros(true_positive.shape), where=(precision + recall) != 0)
                    best_batch_fscore = fscore.max(axis = -1)
                    #select the threhold that gives the mean best f-score for each image crop in the batch.
                    t_best = torch.Tensor(probs_thresh_grid[np.argmax(fscore, -1)]).to(device)
                    f1_meter.update(best_batch_fscore.mean(), best_batch_fscore.shape[0]*best_batch_fscore.shape[1])
                    #loss for best threshold regression
                    thresh_loss = F.smooth_l1_loss(t_preds, t_best)
                    thresh_loss_meter.update(thresh_loss, t_preds.shape[0]*t_preds.shape[1])
                    #calculate the predicted f1 score based on t_preds
                    t_preds = t_preds.detach().cpu().unsqueeze(-1).expand(t_preds.shape[0],
                                                                          t_preds.shape[1],
                                                                          probs_thresh_grid.shape[0])
                    probs_thresh_grid = torch.Tensor(probs_thresh_grid).unsqueeze(0).unsqueeze(0).expand_as(t_preds)
                    pred_f1_index = np.argmax(t_preds < probs_thresh_grid, -1).numpy()
                    f1_preds = pred_f1_index.choose(np.rollaxis(fscore, 2, 0))
                    f1_pred_meter.update(f1_preds.mean(), f1_preds.shape[0]*f1_preds.shape[1])

                num_pos, cls_loss, reg_loss, num_loss = joint_loss(label_map, class_preds, trans_map,
                                                                   trans_preds, cell_number_map, num_preds,
                                                                   sample_sizes,
                                                                   balanced=False)

                # backward + optimize only if in training phase
                loss = cls_loss + args.lambda1 * reg_loss + args.lambda2 * num_loss

                if args.metric == 'f1':
                    loss += args.lambda3 * thresh_loss

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                label_pred = (class_preds.data >= 0.5).detach().cpu().numpy()
                label_map = label_map.detach().cpu().numpy()
                # balanced accuracy
                cell_corrects = (label_map * label_pred).sum((0,2,3))
                bg_corrects = ((1 - label_map) * (1 - label_pred)).sum((0,2,3))
                cell_total = label_map.sum((0,2,3))
                bg_total = (1 - label_map).sum((0,2,3))
                cls_acc_meter.update(cell_corrects, cell_total,
                                     bg_corrects, bg_total)

                reg_loss_meter.update(reg_loss.item(), num_pos)
                num_loss_meter.update(num_loss.item(), num_pos)
                cls_loss_meter.update(cls_loss.item(), label_map.shape[0])

                if phase == 'train':
                    print('{:.4f} ({:.4f})\t' '{:.4f} ({:.4f})\t' '{:.4f} ({:.4f})\t' 
                          '{:.4f} ({:.4f})\t' '{:.4f} ({:.4f})\t' '{:.4f} ({:.4f})\t' '{:.4f}\t'.format(
                        cls_loss_meter.val, cls_loss_meter.avg,
                        reg_loss_meter.val, reg_loss_meter.avg,
                        num_loss_meter.val, num_loss_meter.avg,
                        f1_meter.val, f1_meter.avg,
                        f1_pred_meter.val, f1_pred_meter.avg,
                        thresh_loss_meter.val, thresh_loss_meter.avg,
                        (cls_acc_meter.acc).mean()))
                torch.cuda.empty_cache()

            epoch_acc = sum(cls_acc_meter.acc)/args.num_classes
            epoch_loss = 1 - epoch_acc + reg_loss_meter.avg + num_loss_meter.avg
            epoch_thresh_loss = thresh_loss_meter.avg
            epoch_f1 = f1_meter.avg
            epoch_f1_pred = f1_pred_meter.avg
            print('{} Cls_Loss: {:.4f} Reg_Loss: {:.4f} Num_Loss: {:.4f} Acc: {:.4f} Thresh_Loss: {:.4f} f1_best: {:.4f} f1_pred: {:.4f}'.format(
                phase, cls_loss_meter.avg, reg_loss_meter.avg, num_loss_meter.avg, epoch_acc, epoch_thresh_loss, epoch_f1, epoch_f1_pred))

            # deep copy the model
            if phase == 'val':
                if epoch == 0:
                    best_loss = epoch_loss
                    best_reg_loss = reg_loss_meter.avg
                    best_model_wts = model.state_dict()
                    best_acc = epoch_acc
                    best_f1 = f1_meter.avg
                    best_f1_pred = f1_pred_meter.avg
                else:
                    if args.metric == 'f1':
                        best_metric = 1 - best_f1
                        current_metric = 1 - f1_meter.avg
                    else:
                        best_metric = best_loss
                        current_metric = epoch_loss
                    if current_metric < best_metric:
                        best_loss = epoch_loss
                        best_reg_loss = reg_loss_meter.avg
                        best_model_wts = model.state_dict()
                        best_acc = epoch_acc
                        best_f1 = f1_meter.avg
                        best_f1_pred = f1_pred_meter.avg
            torch.cuda.empty_cache()
        scheduler.step()
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val reg_Loss: {:4f}'.format(best_reg_loss))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Best val f1: {:4f}'.format(best_f1))
    print('Best val f1_pred: {:4f}'.format(best_f1_pred))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def joint_loss(labels, class_preds, trans_targets, trans_preds, cell_number_targets, cell_number_preds, sample_sizes, balanced=False):
    pos = labels > 0
    num_pos = pos.sum().item()
    neg = labels == 0 # shape (batch_size, num_classes, input_size, input_size)
    trans_preds[torch.cat((neg,neg), dim=1)] = trans_targets[torch.cat((neg,neg), dim=1)]
    cell_number_preds[neg] = cell_number_targets[neg]

    #sample_sizes shape (num_classes, 2)

    ce_weights = torch.zeros(labels.data.size())
    for c in range(sample_sizes.shape[0]):
        cls_weights_c = sample_sizes[c].sum()/sample_sizes[c]
        neg_c = neg[:,c,:,:]
        pos_c = pos[:,c,:,:]
        ce_weights[:,c,:,:][neg_c] = cls_weights_c[0]
        ce_weights[:,c,:,:][pos_c] = cls_weights_c[1]

    ce_weights = ce_weights.to(device)

    #reg loss weighted by Num_Loss
    reg_loss = F.smooth_l1_loss(trans_preds, trans_targets, reduction='none')
    if balanced:
        reg_loss = reg_loss * cell_number_targets
    reg_loss = reg_loss.sum()
    cls_loss = F.binary_cross_entropy(class_preds, labels, weight=ce_weights)
    num_loss = F.smooth_l1_loss(cell_number_preds, cell_number_targets, reduction='sum')
    if num_pos > 0:
        return num_pos, cls_loss, reg_loss/num_pos, num_loss/num_pos
    else:
        return num_pos, cls_loss, reg_loss, num_loss

def calc_f1_batch(gt, pred, thresh_grid, r=6):
    #gt_num, pred_num, all are of shape len(images)* len(probs_thresh_grid)
    gt_num = gt.sum((-1,-2)).unsqueeze(-1).expand((gt.shape[0], gt.shape[1], len(thresh_grid))).numpy()
    tp = np.zeros((gt.shape[0], gt.shape[1],len(thresh_grid)))
    pred_num = np.zeros((gt.shape[0], gt.shape[1],len(thresh_grid)))
    # calculate precise, recall and f1 score
    gt_map = np.zeros(gt.shape)
    for c in range(args.num_classes):
        for n in range(gt.shape[0]):
            points = np.asarray((np.where(gt[n,c]>0)[0], np.where(gt[n,c]>0)[1])).transpose()
            for p in points:
                y = p[0]
                x = p[1]
                y_disk, x_disk = np.ogrid[-y: gt.shape[-2] - y, -x: gt.shape[-1] - x]
                disk_mask = y_disk ** 2 + x_disk ** 2 <= (args.r) ** 2
                gt_map[n, c, disk_mask] = 1
    pair_distances = []
    for i in range(len(thresh_grid)):
        pred_map = pred > thresh_grid[i]
        pred_num[:,:,i] = pred_map.sum((-2, -1))
        pred_map = np.minimum(pred_map,gt_map)
        pred_map, pair_distances_i = remove_non_assigned(pred_map, gt, args.num_classes, r)
        if i == 0:
            pair_distances = pair_distances_i
        result_map = gt_map * pred_map.numpy()
        tp[:,:,i] = result_map.sum((-2, -1))

    return tp, pred_num, gt_num, pair_distances


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = (self.sum + 1e-12) / (self.count + 1e-12)

class AccuracyMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.correct = np.zeros((args.num_classes, 2))
        self.count = np.zeros((args.num_classes, 2))
        self.acc = np.zeros(args.num_classes)
    def update(self, fg_corrects, fg_total, bg_corrects, bg_total):
        (self.correct)[:, 1] += fg_corrects
        (self.correct)[:, 0] += bg_corrects
        (self.count)[:, 1] += fg_total
        (self.count)[:, 0] += bg_total
        (self.acc) = (((self.correct)[:,0] + 1e-12) / ((self.count)[:,0] + 1e-12) +
                        ((self.correct)[:,1] + 1e-12) / ((self.count)[:,1] + 1e-12)) / 2

if __name__ == '__main__':
    main()
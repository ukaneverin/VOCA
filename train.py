from __future__ import print_function, division
import math
import time
from pdb import set_trace
from utils import *
import argparse


def train_model(model, optimizer, scheduler, dataloaders, sample_sizes, args, voca_data):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    since = time.time()

    best_model_wts = model.state_dict()
    best_loss = 0.0
    best_reg_loss = 0.0
    best_acc = 0.0

    for epoch in range(args.n_epochs):
        print('Epoch {}/{}'.format(epoch + 1, args.n_epochs), 'lr: {}'.format(optimizer.param_groups[0]['lr']))
        print('-' * 10)
        print('cls_loss\treg_loss\tnum_loss\tf1_best\t\tf1_pred\t\tthresh_loss\tacc\t')

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
            cls_acc_meter = AccuracyMeter(voca_data.num_classes)
            thresh_loss_meter = AverageMeter()

            torch.cuda.empty_cache()

            for sample in dataloaders[phase]:
                inputs = sample['image']
                task_map = sample['task_map']
                gt_map = sample['gt_map']
                gt_map = gt_map.permute(0, 3, 1, 2).type(torch.FloatTensor)

                label_map = task_map[:, :, :, :voca_data.num_classes].permute(0, 3, 1, 2).type(torch.FloatTensor)
                trans_map = task_map[:, :, :, voca_data.num_classes:3 * voca_data.num_classes].permute(0, 3, 1, 2).type(torch.FloatTensor)
                cell_number_map = task_map[:, :, :, 3 * voca_data.num_classes:].permute(0, 3, 1, 2).type(torch.FloatTensor)

                inputs = inputs.to(device)
                label_map = label_map.to(device)
                trans_map = trans_map.to(device)
                cell_number_map = cell_number_map.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                torch.cuda.empty_cache()
                if phase == 'train':
                    class_preds, trans_preds, num_preds, t_preds, hx = model(inputs)
                elif phase == 'val':
                    with torch.no_grad():
                        class_preds, trans_preds, num_preds, t_preds, hx = model(inputs)

                """
                If metrics is 'f1', voca needs to learn the best threshold and save the best model in terms of f1 score.
                Calculate best batch f1 based on various threholds and choose the best as the target to learn.
                """
                if args.metric == 'f1':
                    # get the accumulated confidence map
                    voca_map = np.zeros(label_map.shape)

                    weighted_confidence_map = class_preds * num_preds
                    x_coord_matrix = np.array([np.arange(label_map.shape[-1]), ] * label_map.shape[-2])
                    y_coord_matrix = np.array([np.arange(label_map.shape[-2]), ] * label_map.shape[-1]).transpose()
                    x_map = np.around(
                        trans_preds[:, :voca_data.num_classes, :, :].detach().cpu().numpy() + x_coord_matrix).astype(int)
                    y_map = np.around(trans_preds[:, voca_data.num_classes:2 * voca_data.num_classes, :,
                                      :].detach().cpu().numpy() + y_coord_matrix).astype(int)
                    for c in range(voca_data.num_classes):  # for each cell class
                        for n in range(label_map.shape[0]):  # batch size
                            x = x_map[n, c].reshape(-1)
                            y = y_map[n, c].reshape(-1)
                            valid_location_index = (x > 0) * (x < voca_map.shape[-1]) * (y > 0) * (y < voca_map.shape[-2])
                            x = x[valid_location_index]
                            y = y[valid_location_index]
                            x_coord = x_coord_matrix.reshape(-1)[valid_location_index]
                            y_coord = y_coord_matrix.reshape(-1)[valid_location_index]
                            np.add.at(voca_map[n, c], (y, x), weighted_confidence_map[n, c, y_coord, x_coord].detach().cpu().numpy())

                    # nms on the map
                    suppression_mask = torch.zeros(voca_map.shape)
                    for c in range(voca_data.num_classes):  # for each cell class
                        for n in range(voca_map.shape[0]):  # batch size
                            detected_coords = [x.tolist() + [voca_map[n, c, x[0], x[1]]]
                                               for x in np.asarray(np.where(voca_map[n, c, :, :] > 0)).transpose()]
                            detected_coords = np.asarray(detected_coords, dtype=float).reshape(-1, 3)
                            detected_coords = non_max_suppression(detected_coords, args.nms_r)  # non_max_suppression
                            for coords in detected_coords:
                                # real accumulated confidence map (-3 to +3 pixels)
                                suppression_mask[n, c, int(coords[0]), int(coords[1])] = voca_map[n, c,
                                             max(int(coords[0]) - 2, 0):min(int(coords[0]) + 3, voca_map.shape[-2]),
                                             max(int(coords[1]) - 2, 0):min(int(coords[1]) + 3, voca_map.shape[-1])].sum()
                    # calculate the f1 scores
                    probs_thresh_grid = np.linspace(0.1, 0.85, 16)[:-1]
                    minval = 0.0
                    maxval = math.pi * voca_data.r * voca_data.r
                    suppression_mask -= minval
                    suppression_mask = suppression_mask / (maxval - minval)

                    true_positive, pred_num, gt_num, pair_distances = calc_f1_batch(gt_map, suppression_mask,
                                                                                    probs_thresh_grid, voca_data.num_classes,
                                                                                    args.nms_r)
                    precision = np.divide(true_positive, pred_num, out=np.zeros(true_positive.shape), where=(pred_num) != 0)
                    recall = np.divide(true_positive, gt_num, out=np.zeros(true_positive.shape), where=(gt_num) != 0)
                    fscore = 2 * np.divide((precision * recall), (precision + recall),
                                           out=np.zeros(true_positive.shape), where=(precision + recall) != 0)
                    best_batch_fscore = fscore.max(axis=-1)
                    # select the threhold that gives the mean best f-score for each image crop in the batch.
                    t_best = torch.Tensor(probs_thresh_grid[np.argmax(fscore, -1)]).to(device)
                    f1_meter.update(best_batch_fscore.mean(), best_batch_fscore.shape[0] * best_batch_fscore.shape[1])
                    # loss for best threshold regression
                    thresh_loss = F.smooth_l1_loss(t_preds, t_best)
                    thresh_loss_meter.update(thresh_loss, t_preds.shape[0] * t_preds.shape[1])
                    # calculate the predicted f1 score based on t_preds
                    t_preds = t_preds.detach().cpu().unsqueeze(-1).expand(t_preds.shape[0],
                                                                          t_preds.shape[1],
                                                                          probs_thresh_grid.shape[0])
                    probs_thresh_grid = torch.Tensor(probs_thresh_grid).unsqueeze(0).unsqueeze(0).expand_as(t_preds)
                    pred_f1_index = np.argmax(t_preds < probs_thresh_grid, -1).numpy()
                    f1_preds = pred_f1_index.choose(np.rollaxis(fscore, 2, 0))
                    f1_pred_meter.update(f1_preds.mean(), f1_preds.shape[0] * f1_preds.shape[1])

                num_pos, cls_loss, reg_loss, num_loss = joint_loss(device, label_map, class_preds, trans_map,
                                                                   trans_preds, cell_number_map, num_preds,
                                                                   sample_sizes,
                                                                   balanced=False)

                # backward + optimize only if in training phase
                loss = cls_loss + reg_loss + num_loss

                if args.metric == 'f1':
                    loss += thresh_loss

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                label_pred = (class_preds.data >= 0.5).detach().cpu().numpy()
                label_map = label_map.detach().cpu().numpy()
                # balanced accuracy
                cell_corrects = (label_map * label_pred).sum((0, 2, 3))
                bg_corrects = ((1 - label_map) * (1 - label_pred)).sum((0, 2, 3))
                cell_total = label_map.sum((0, 2, 3))
                bg_total = (1 - label_map).sum((0, 2, 3))
                cls_acc_meter.update(cell_corrects, cell_total, bg_corrects, bg_total)

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

            epoch_acc = sum(cls_acc_meter.acc) / voca_data.num_classes
            epoch_loss = 1 - epoch_acc + reg_loss_meter.avg + num_loss_meter.avg
            epoch_thresh_loss = thresh_loss_meter.avg
            epoch_f1 = f1_meter.avg
            epoch_f1_pred = f1_pred_meter.avg
            print(
                '{} Cls_Loss: {:.4f} Reg_Loss: {:.4f} Num_Loss: {:.4f} Acc: {:.4f} Thresh_Loss: {:.4f} f1_best: {:.4f} f1_pred: {:.4f}'.format(
                    phase, cls_loss_meter.avg, reg_loss_meter.avg, num_loss_meter.avg, epoch_acc, epoch_thresh_loss,
                    epoch_f1, epoch_f1_pred))

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
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val reg_Loss: {:4f}'.format(best_reg_loss))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Best val f1: {:4f}'.format(best_f1))
    print('Best val f1_pred: {:4f}'.format(best_f1_pred))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

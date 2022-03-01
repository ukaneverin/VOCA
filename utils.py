import numpy as np
import torch
from scipy.optimize import linear_sum_assignment as lsa
import torch.nn.functional as F


def non_max_suppression(points, dis_thresh):
    # if there are no points, return an empty list
    if len(points) == 0:
        return np.asarray([]).reshape(-1, 3)

    if points.dtype.kind == 'i':
        points = points.astype('float')
    np.random.shuffle(points)

    pick = []

    x = points[:, 1]
    y = points[:, 0]
    score = points[:, 2]

    idxs = np.argsort(score)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        x_dis = x[idxs[:last]] - x[i]
        y_dis = y[idxs[:last]] - y[i]

        dis = (x_dis ** 2 + y_dis ** 2) ** 0.5
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(dis < dis_thresh)[0])))

    return points[pick]


def remove_non_assigned(pred_map, gt, num_classes, nms_r=6):
    pair_distances = []
    for c in range(num_classes):
        for n in range(gt.shape[0]):
            '''obtain the coords for pred and gt'''
            gt_coords_tuple = np.where(gt[n, c] > 0)
            gt_coords = np.asarray((gt_coords_tuple[0], gt_coords_tuple[1])).transpose()
            pred_coords_tuple = np.where(pred_map[n, c] > 0)
            pred_coords = np.asarray((pred_coords_tuple[0], pred_coords_tuple[1])).transpose()
            pred_coords = torch.Tensor(pred_coords.reshape(-1, 2))
            gt_coords = torch.Tensor(gt_coords.reshape(-1, 2))
            '''calculate the distance matrix'''
            m1 = pred_coords.shape[0]
            m2 = gt_coords.shape[0]
            d = pred_coords.shape[1]
            x = pred_coords.unsqueeze(1).expand(m1, m2, d)
            y = gt_coords.unsqueeze(0).expand(m1, m2, d)
            pred_x_gt = torch.pow(torch.pow(x - y, 2).sum(2), .5).numpy()
            '''assign pred to gt'''
            pred_ind, gt_ind = lsa(pred_x_gt)  # assigned preds
            '''delete the unassigned prediciton points from pred_map'''
            removed_preds_coords = np.delete(pred_coords, pred_ind, axis=0).numpy()
            pred_map[n, c, removed_preds_coords[:, 0], removed_preds_coords[:, 1]] = 0
            '''delete the assigned but too far points from pred_map'''
            too_far_points = pred_coords[pred_ind[pred_x_gt[pred_ind, gt_ind] > nms_r]].numpy()
            pred_map[n, c, too_far_points[:, 0], too_far_points[:, 1]] = 0
            pair_distances += pred_x_gt[pred_ind, gt_ind].reshape(-1).tolist()
    return pred_map, pair_distances


def calc_f1(gt, pred_map, num_classes, nms_r=6):
    # gt_num, pred_num, all are of shape len(images) * num_classes
    gt_num = gt.sum((-1, -2)).numpy()
    # calculate precise, recall and f1 score
    gt_map = np.zeros(gt.shape)
    for c in range(num_classes):
        for n in range(gt.shape[0]):
            points = np.asarray((np.where(gt[n, c] > 0)[0], np.where(gt[n, c] > 0)[1])).transpose()
            for p in points:
                y = p[0]
                x = p[1]
                y_disk, x_disk = np.ogrid[-y: gt.shape[-2] - y, -x: gt.shape[-1] - x]
                disk_mask = y_disk ** 2 + x_disk ** 2 <= nms_r ** 2
                gt_map[n, c, disk_mask] = 1

    pred_num = pred_map.sum((-2, -1))
    pred_map = np.minimum(pred_map, gt_map)
    pred_map, pair_distances = remove_non_assigned(pred_map, gt, num_classes, nms_r)

    result_map = gt_map * pred_map.numpy()
    tp = result_map.sum((-2, -1))
    return tp, pred_num, gt_num, pair_distances


def calc_f1_batch(gt, pred, thresh_grid, num_classes, nms_r=6):
    # gt_num, pred_num, all are of shape len(images)* len(probs_thresh_grid)
    gt_num = gt.sum((-1, -2)).unsqueeze(-1).expand((gt.shape[0], gt.shape[1], len(thresh_grid))).numpy()
    tp = np.zeros((gt.shape[0], gt.shape[1], len(thresh_grid)))
    pred_num = np.zeros((gt.shape[0], gt.shape[1], len(thresh_grid)))
    # calculate precise, recall and f1 score
    gt_map = np.zeros(gt.shape)
    for c in range(num_classes):
        for n in range(gt.shape[0]):
            points = np.asarray((np.where(gt[n, c] > 0)[0], np.where(gt[n, c] > 0)[1])).transpose()
            for p in points:
                y = p[0]
                x = p[1]
                y_disk, x_disk = np.ogrid[-y: gt.shape[-2] - y, -x: gt.shape[-1] - x]
                disk_mask = y_disk ** 2 + x_disk ** 2 <= nms_r ** 2
                gt_map[n, c, disk_mask] = 1
    pair_distances = []
    for i in range(len(thresh_grid)):
        pred_map = pred > thresh_grid[i]
        pred_num[:, :, i] = pred_map.sum((-2, -1))
        pred_map = np.minimum(pred_map, gt_map)
        pred_map, pair_distances_i = remove_non_assigned(pred_map, gt, num_classes, nms_r)
        if i == 0:
            pair_distances = pair_distances_i
        result_map = gt_map * pred_map.numpy()
        tp[:, :, i] = result_map.sum((-2, -1))

    return tp, pred_num, gt_num, pair_distances


def get_mean_and_std(batch_size, dataset, num_classes):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    sample_sizes = torch.zeros(num_classes, 2)
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
            mean[i] += inputs[:, i, :, :].mean(1).mean(1).sum()
            non_reduced_mean = inputs[:, i, :, :].mean(1, keepdim=True).mean(2, keepdim=True).expand_as(
                inputs[:, i, :, :])
            std[i] += (inputs[:, i, :, :] - non_reduced_mean).pow(2).sum()

        task_map = sample['task_map']
        for n in range(num_classes):
            sample_sizes[n, 0] += (task_map[:, :, :, n] == 0).sum().item()
            sample_sizes[n, 1] += (task_map[:, :, :, n] == 1).sum().item()
    std.div_(image_dimension_x * image_dimension_y * len(dataset) - 1).pow_(0.5)
    mean.div_(len(dataset))
    return mean.cpu(), std.cpu(), sample_sizes


def joint_loss(device, labels, class_preds, trans_targets, trans_preds, cell_number_targets, cell_number_preds,
               sample_sizes,
               balanced=False):
    pos = labels > 0
    num_pos = pos.sum().item()
    neg = labels == 0  # shape (batch_size, num_classes, input_size, input_size)
    trans_preds[torch.cat((neg, neg), dim=1)] = trans_targets[torch.cat((neg, neg), dim=1)]
    cell_number_preds[neg] = cell_number_targets[neg]

    # sample_sizes shape (num_classes, 2)

    ce_weights = torch.zeros(labels.data.size())
    for c in range(sample_sizes.shape[0]):
        cls_weights_c = sample_sizes[c].sum() / sample_sizes[c]
        neg_c = neg[:, c, :, :]
        pos_c = pos[:, c, :, :]
        ce_weights[:, c, :, :][neg_c] = cls_weights_c[0]
        ce_weights[:, c, :, :][pos_c] = cls_weights_c[1]

    ce_weights = ce_weights.to(device)

    # reg loss weighted by Num_Loss
    reg_loss = F.smooth_l1_loss(trans_preds, trans_targets, reduction='none')
    if balanced:
        reg_loss = reg_loss * cell_number_targets
    reg_loss = reg_loss.sum()
    cls_loss = F.binary_cross_entropy(class_preds, labels, weight=ce_weights)
    num_loss = F.smooth_l1_loss(cell_number_preds, cell_number_targets, reduction='sum')
    if num_pos > 0:
        return num_pos, cls_loss, reg_loss / num_pos, num_loss / num_pos
    else:
        return num_pos, cls_loss, reg_loss, num_loss


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

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.correct = np.zeros((self.num_classes, 2))
        self.count = np.zeros((self.num_classes, 2))
        self.acc = np.zeros(self.num_classes)

    def update(self, fg_corrects, fg_total, bg_corrects, bg_total):
        (self.correct)[:, 1] += fg_corrects
        (self.correct)[:, 0] += bg_corrects
        (self.count)[:, 1] += fg_total
        (self.count)[:, 0] += bg_total
        (self.acc) = (((self.correct)[:, 0] + 1e-12) / ((self.count)[:, 0] + 1e-12) +
                      ((self.correct)[:, 1] + 1e-12) / ((self.count)[:, 1] + 1e-12)) / 2

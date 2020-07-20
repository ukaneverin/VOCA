import numpy as np
import torch
from scipy.optimize import linear_sum_assignment as lsa

def remove_non_assigned(pred_map, gt, r=6):
	pair_distances = []
	'''obtain the coords for pred and gt'''
	gt_coords_tuple = np.where(gt > 0)
	gt_coords = np.asarray((gt_coords_tuple[0], gt_coords_tuple[1])).transpose()
	pred_coords_tuple = np.where(pred_map > 0)
	pred_coords = np.asarray((pred_coords_tuple[0], pred_coords_tuple[1])).transpose()
	pred_coords = torch.Tensor(pred_coords.reshape(-1,2))
	gt_coords = torch.Tensor(gt_coords.reshape(-1,2))
	'''calculate the distance matrix'''
	m1 = pred_coords.shape[0]
	m2 = gt_coords.shape[0]
	d = pred_coords.shape[1]
	x = pred_coords.unsqueeze(1).expand(m1, m2, d)
	y = gt_coords.unsqueeze(0).expand(m1, m2, d)
	pred_x_gt = torch.pow(torch.pow(x - y, 2).sum(2), .5).numpy()
	'''assign pred to gt'''
	pred_ind, gt_ind =  lsa(pred_x_gt) #assigned preds
	'''delete the unassigned prediciton points from pred_map'''
	removed_preds_coords = np.delete(pred_coords, pred_ind, axis=0).numpy()
	pred_map[removed_preds_coords[:,0], removed_preds_coords[:,1]] = 0
	'''delete the assigned but out of radius r points from pred_map'''
	too_far_points = pred_coords[pred_ind[pred_x_gt[pred_ind, gt_ind] > r]].numpy()
	pred_map[too_far_points[:,0],too_far_points[:,1]] = 0
	'''obtain the pair distances between assigned predictions and gt points'''
	pair_distances += pred_x_gt[pred_ind, gt_ind].reshape(-1).tolist()
	return pred_map, pair_distances


def calc_f1(gt, pred_map, r=6):
	#gt and pred_map are Tensors
	gt_num = gt.sum().item()
	gt_map = np.zeros(gt.shape) #gt_map masks the disk of radius r around gt positive points
	points = np.asarray((np.where(gt>0)[0], np.where(gt>0)[1])).transpose()
	for p in points:
		y = p[0]
		x = p[1]
		y_disk, x_disk = np.ogrid[-y: gt.shape[-2] - y, -x: gt.shape[-1] - x]
		disk_mask = y_disk ** 2 + x_disk ** 2 <= r ** 2
		gt_map[disk_mask] = 1

	pred_num = pred_map.sum().item()
	pred_map = np.minimum(pred_map,gt_map)
	pred_map, pair_distances = remove_non_assigned(pred_map, gt, r)

	result_map = gt_map * pred_map.numpy()
	tp = result_map.sum()

	#calculate f1 score
	precision = tp / pred_num
	recall = tp / gt_num
	f1 = 2*(precision * recall) / (precision + recall)
	return f1, precision, recall, pair_distances

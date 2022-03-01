import numpy as np
from imageio import imread, imsave
import os
from models.res import VOCA_Res
import torch
from torchvision import transforms
from utils import non_max_suppression, remove_non_assigned, calc_f1
from Dataset_classes import TestImageDataset
import math
from pdb import set_trace
import argparse


def visualization(image, pred_map, nms_r):
    colors = {
        0: [255, 165, 0],
        1: [0, 255, 0]
    }
    for c in range(pred_map.shape[0]):
        coords = np.where(pred_map[c] > 0)
        for i in range(len(coords[0])):
            x = coords[1][i]
            y = coords[0][i]
            y_disk, x_disk = np.ogrid[-y: image.shape[0] - y, -x: image.shape[1] - x]
            disk_mask_within = y_disk ** 2 + x_disk ** 2 <= (nms_r + 1) ** 2
            disk_mask_without = y_disk ** 2 + x_disk ** 2 >= nms_r ** 2
            disk_mask = disk_mask_within * disk_mask_without
            image[disk_mask, :] = colors[c]
    return image


def detection_validation(args, voca_data):
    val_save_path = os.path.join(voca_data.val_path, 'r%s_lr%s_%s' % (voca_data.r, args.lr, args.metric))
    if not os.path.exists(val_save_path):
        os.makedirs(val_save_path)
    if not os.path.exists(os.path.join(val_save_path, 'voca_map')):
        os.makedirs(os.path.join(val_save_path, 'voca_map'))
    if not os.path.exists(os.path.join(val_save_path, 'pred_map')):
        os.makedirs(os.path.join(val_save_path, 'pred_map'))
    plot_directory = os.path.join(val_save_path, 'prediction_images/')
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)

    # get the dimensions of the images
    h, w, _ = imread(os.path.join(voca_data.image_folder, voca_data.image_list[0])).shape
    suppression_mask_all = torch.zeros((len(voca_data.image_list), voca_data.num_classes, h, w))
    gt_map_all = torch.zeros((len(voca_data.image_list), voca_data.num_classes, h, w))
    processed_images = []
    for cv in range(voca_data.n_cv):
        # data transform with learned mean and std
        mean, std, sample_sizes = voca_data.get_mean_std(cv)
        data_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        # get device and load best model for eval
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = VOCA_Res(num_classes=voca_data.num_classes).to(device)
        model.load_state_dict(torch.load(os.path.join(voca_data.model_directory, 'r%s_lr%s_%s_cv%s.pth' %
                                                      (voca_data.r, args.lr, args.metric, cv))))
        model.eval()
        with open(os.path.join(voca_data.library_path, 'CV_%s_val.csv' % cv), 'r') as f:
            val_set_list = [line.strip() for line in f]
        for image_name in val_set_list:
            processed_images.append(image_name)
            image_file = os.path.join(voca_data.image_folder, image_name)
            image = imread(image_file)
            image = image[:, :, :3]

            gt_map = np.load(os.path.join(voca_data.gt_map_path, '%s.npy' % image_name))
            gt_map_all[voca_data.image_list.index(image_name)] = torch.FloatTensor(gt_map).permute(2, 0, 1)
            image_datasets = TestImageDataset(image.astype(float), voca_data.input_size, data_transforms)

            dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=36, shuffle=True, num_workers=6)

            t_map = torch.zeros((voca_data.num_classes, image.shape[0], image.shape[1]))
            suppression_mask = torch.zeros((voca_data.num_classes, image.shape[0], image.shape[1]))
            for sample in dataloaders:
                inputs = sample['image_crop']
                image_coords = sample['coords'].numpy()
                inputs = inputs.to(device)
                # forward
                with torch.no_grad():
                    class_probs, trans_preds, num_preds, t_preds, hx = model(inputs)
                # get the accumulated confidence map
                voca_map_batch = np.zeros(class_probs.shape)

                weighted_confidence_map = class_probs * num_preds
                x_coord_matrix = np.array([np.arange(class_probs.shape[-1]), ] * class_probs.shape[-2])
                y_coord_matrix = np.array([np.arange(class_probs.shape[-2]), ] * class_probs.shape[-1]).transpose()
                x_map = np.around(
                    trans_preds[:, :voca_data.num_classes, :, :].detach().cpu().numpy() + x_coord_matrix).astype(int)
                y_map = np.around(trans_preds[:, voca_data.num_classes:2 * voca_data.num_classes, :,
                                  :].detach().cpu().numpy() + y_coord_matrix).astype(int)
                for c in range(voca_data.num_classes):  # for each cell class
                    for n in range(class_probs.shape[0]):  # batch size
                        x = x_map[n, c].reshape(-1)
                        y = y_map[n, c].reshape(-1)
                        valid_location_index = (x > 0) * (x < voca_map_batch.shape[-1]) * (y > 0) * (
                                y < voca_map_batch.shape[-2])
                        x = x[valid_location_index]
                        y = y[valid_location_index]
                        x_coord = x_coord_matrix.reshape(-1)[valid_location_index]
                        y_coord = y_coord_matrix.reshape(-1)[valid_location_index]
                        np.add.at(voca_map_batch[n, c], (y, x),
                                  weighted_confidence_map[n, c, y_coord, x_coord].detach().cpu().numpy())

                # nms on the map
                suppression_mask_batch = torch.zeros(voca_map_batch.shape)
                for c in range(voca_data.num_classes):  # for each cell class
                    for n in range(voca_map_batch.shape[0]):  # batch size
                        detected_coords = [x.tolist() + [voca_map_batch[n, c, x[0], x[1]]]
                                           for x in np.asarray(np.where(voca_map_batch[n, c, :, :] > 0)).transpose()]
                        detected_coords = np.asarray(detected_coords, dtype=float).reshape(-1, 3)
                        detected_coords = non_max_suppression(detected_coords, args.nms_r)  # non_max_suppression
                        for coords in detected_coords:
                            # real accumulated confidence map (-3 to +3 pixels)
                            suppression_mask_batch[n, c, int(coords[0]), int(coords[1])] = voca_map_batch[n, c,
                                       max(int(coords[0]) - 2, 0):min(int(coords[0]) + 3, voca_map_batch.shape[-2]),
                                       max(int(coords[1]) - 2, 0):min(int(coords[1]) + 3, voca_map_batch.shape[-1])].sum()
                minval = 0.0
                maxval = math.pi * voca_data.r * voca_data.r
                suppression_mask_batch -= minval
                suppression_mask_batch = suppression_mask_batch / (maxval - minval)

                for n in range(class_probs.shape[0]):  # batch size
                    x = int(image_coords[n, 0])
                    y = int(image_coords[n, 1])
                    t_map[:, y:y + voca_data.input_size, x:x + voca_data.input_size] = \
                        t_preds[n].unsqueeze(-1).unsqueeze(-1).expand_as(weighted_confidence_map[n]).detach().cpu()
                    suppression_mask[:, y:y + voca_data.input_size, x:x + voca_data.input_size] = \
                        suppression_mask_batch[n].detach().cpu()

            # threshold the peaks by t_map
            suppression_mask[suppression_mask < t_map] = 0
            suppression_mask[suppression_mask < 0.05] = 0
            np.save(os.path.join(val_save_path, 'pred_map/%s_final_map.npy' % image_name), suppression_mask)
            image = visualization(image, suppression_mask, args.nms_r)
            imsave(os.path.join(plot_directory, '%s_predictions.png' % image_name), image)
            suppression_mask_all[voca_data.image_list.index(image_name)] = suppression_mask

    true_positive, pred_num, gt_num, pair_distances = calc_f1(gt_map_all, suppression_mask_all, voca_data.num_classes, args.nms_r)
    # get the image and channel wise fscore
    precision = np.divide(true_positive, pred_num, out=np.zeros(true_positive.shape), where=(pred_num) != 0)
    recall = np.divide(true_positive, gt_num, out=np.zeros(true_positive.shape), where=(gt_num) != 0)
    fscore = 2 * np.divide((precision * recall), (precision + recall),
                           out=np.zeros(true_positive.shape), where=(precision + recall) != 0)
    fscore = np.delete(fscore, voca_data.image_list.index(list(set(voca_data.image_list) - set(processed_images))[0]), axis=0)
    print('Fscore (n,c) ---- ')
    print(fscore)
    # get the total fscore
    precision_all = true_positive.sum(0) / pred_num.sum(0).numpy()
    recall_all = true_positive.sum(0) / gt_num.sum(0)
    fscore_all = 2 * (precision_all * recall_all) / (precision_all + recall_all)
    print('Precision: ', precision_all)
    print('Recall: ', recall_all)
    print('Fscore: ', fscore_all)

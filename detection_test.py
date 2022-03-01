import numpy as np
from imageio import imread, imsave
import os
from models.res import VOCA_Res
import torch
from torchvision import transforms
from utils import non_max_suppression, remove_non_assigned, calc_f1
from Dataset_classes import TestImageDataset
import math
import openslide
from SlideTileExtractor import extract_tissue
import time
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


def write_coords(suppression_mask, hx_map, origin_coords, channel, image_name, cell_file):
    suppression_mask = suppression_mask.numpy()
    plot_coords = np.transpose(np.nonzero(suppression_mask[channel]))
    for dc in plot_coords:
        cell_file.write(
            '%s,%s,%s,%s,%s\n' % (
                image_name, dc[1] + origin_coords[0],
                dc[0] + origin_coords[1],
                suppression_mask[channel, dc[0], dc[1]],
                ','.join(hx_map[:, dc[0], dc[1]].numpy().astype(str))))


def voca_prediction(image, args, voca_data, model, device, data_transforms):
    image_datasets = TestImageDataset(image.astype(float), voca_data.input_size, data_transforms)
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=36, shuffle=True, num_workers=6)

    t_map = torch.zeros((voca_data.num_classes, image.shape[0], image.shape[1]))
    suppression_mask = torch.zeros((voca_data.num_classes, image.shape[0], image.shape[1]))
    hx_map = torch.zeros((64, image.shape[0], image.shape[1]))
    for sample in dataloaders:
        inputs = sample['image_crop']
        image_coords = sample['coords'].numpy()
        inputs = inputs.to(device)
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
                                                                                   max(int(coords[0]) - 2, 0):min(
                                                                                       int(coords[0]) + 3,
                                                                                       voca_map_batch.shape[-2]),
                                                                                   max(int(coords[1]) - 2, 0):min(
                                                                                       int(coords[1]) + 3,
                                                                                       voca_map_batch.shape[-1])].sum()
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
            hx_map[:, y:y + voca_data.input_size, x:x + voca_data.input_size] = hx[n].detach()

    # threshold the peaks by t_map
    suppression_mask[suppression_mask < t_map] = 0
    suppression_mask[suppression_mask < 0.05] = 0

    return suppression_mask, hx_map


def set_up(image_file, args, voca_data, prediction_directory):
    image_name = image_file.split('/')[-1]
    if not os.path.exists(prediction_directory):
        os.makedirs(prediction_directory)
    cell_files = []
    for c in range(voca_data.num_classes):
        cell_files.append(open(os.path.join(prediction_directory, '%s_%s.csv' % (image_name, voca_data.cell_type_dict[c])), 'w+'))
    # data transform with learned mean and std
    mean, std, sample_sizes = voca_data.get_mean_std(args.cv)
    data_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    # get device and load best model for eval
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = VOCA_Res(num_classes=voca_data.num_classes).to(device)
    model.load_state_dict(torch.load(os.path.join(voca_data.model_directory, 'r%s_lr%s_%s_cv%s.pth' %
                                                  (voca_data.r, args.lr, args.metric, args.cv))))
    model.eval()

    return cell_files, model, device, data_transforms, image_name


def detection_roi(image_file, args, voca_data, prediction_directory):
    cell_files, model, device, data_transforms, image_name = set_up(image_file, args, voca_data, prediction_directory)

    image = imread(image_file)
    image = image[:, :, :3]
    suppression_mask, hx_map = voca_prediction(image, args, voca_data, model, device, data_transforms)

    image = visualization(image, suppression_mask, args.nms_r)
    imsave(os.path.join(prediction_directory, '%s_predictions.png' % image_name), image)

    origin_coords = [0, 0]
    for c in range(voca_data.num_classes):
        write_coords(suppression_mask, hx_map, origin_coords, c, image_name, cell_files[c])
    for c in range(voca_data.num_classes):
        cell_files[c].close()


def detection_ws(args, voca_data, prediction_directory):
    cell_files, model, device, data_transforms, image_name = set_up(args.slide_file, args, voca_data, prediction_directory)

    slide = openslide.OpenSlide(args.slide_file)
    patch_size = 762
    truegrid = extract_tissue.make_sample_grid(slide, patch_size=patch_size,
                                          mpp=0.5, power=None, min_cc_size=1,
                                          max_ratio_size=10, dilate=True,
                                          erode=False, prune=False,
                                          overlap=1, bmp=args.mask_file,
                                          oversample=False)

    level, mult = extract_tissue.find_level(slide, 0.5, patchsize=patch_size)

    for ws_coords in truegrid:
        if mult != 1.0:
            # mostly happens with GT450 slides
            image = slide.read_region(ws_coords, level, (np.int(np.round(patch_size * mult)), np.int(np.round(patch_size * mult))))
            image = image.resize((patch_size, patch_size))
        else:
            image = slide.read_region(ws_coords, level, (patch_size, patch_size))
        image = np.array(image)
        image = image[:, :, :3]
        suppression_mask, hx_map = voca_prediction(image, args, voca_data, model, device, data_transforms)
        for c in range(voca_data.num_classes):
            write_coords(suppression_mask, hx_map, ws_coords, c, image_name, cell_files[c])
    for c in range(voca_data.num_classes):
        cell_files[c].close()

import numpy as np
import os
from models.res import VOCA_Res
import torch
from torchvision import transforms
import time
from Dataset_classes import LymphocytesTestImage
import openslide
from SlideTileExtractor import extract_tissue
import math
import argparse
from utils import non_max_suppression

parser = argparse.ArgumentParser(description='VOCA testing')
parser.add_argument('--dataset', default='lung_tt', type=str, help='lung_tt(which means til+tumor), lung_til, breast_til')
parser.add_argument('--annotation', default='AR', type=str, help='this specifies where the annotations are')
parser.add_argument('--output_path', default='/lila/data/fuchs/xiec/results/VOCA', type=str, help='working directory')
parser.add_argument('--num_classes', default=2, type=int, help='how many cell types')
parser.add_argument('--input_size', default=127, type=int, help='input image size to model')
parser.add_argument('--grid_size', default=127, type=int, help='grid size')
parser.add_argument('--patch_size', default=762, type=int, help='patch size')
parser.add_argument('--r', default=12, type=int, help='the only hyperparameter r')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--model', default='res', type=str, help='which backbone model to use: res dense vgg etc.')
parser.add_argument('--loss', default='weighted', type=str, help='weighted, balanced, focal')
parser.add_argument('--num_workers', default=8, type=int, help='how many workers for dataloader')
parser.add_argument('--batch_size', default=8, type=int, help='bactch size')
parser.add_argument('--metric', default='f1', type=str,
                    help='acc, f1; this tells whether to compute the \
                    f1 score for every batch and save best model based on which metric')
parser.add_argument('--nms_r', default=6, type=int, help='the distance threshold for non-max suppresion')
parser.add_argument('--vis', action='store_true', help='if parsed, maps will be saved for visualization')
parser.add_argument('--cv', default=0, type=int, help='choose the model trained with which cross validation')
parser.add_argument('--id', default=557685, type=int, help='the slide id you want to run VOCA on')
parser.add_argument('--use_bmp', action='store_true', help='if parsed, will only run voca on the annotated region')

def main():
    global args
    global device
    args = parser.parse_args()

    #data transform with learned mean and std
    cv = args.cv
    mean = torch.FloatTensor(np.load(os.path.join(args.output_path, args.dataset, 'split/CV_%s_mean.npy' % cv)))
    std = torch.FloatTensor(np.load(os.path.join(args.output_path, args.dataset, 'split/CV_%s_std.npy' % cv)))

    data_transforms = transforms.Compose([
                      transforms.ToTensor(),
                      transforms.Normalize(mean, std)
                      ])

    # get device and load best model for eval
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = VOCA_Res().to(device)

    model_directory = os.path.join(args.output_path, args.dataset, 'trained_models/')
    model.load_state_dict(torch.load(model_directory+'r%s_lr%s_%s_cv%s.pth' %
                                     (args.r, args.lr, args.metric, cv)))
    model.eval()

    """
    make grid for slides
    This dataset is 20x .5 mpp. Only detecting the lymphocytes; Be VERY cafeful about the resolution, coords, etc...
    """
    if (args.dataset).split('_')[0] == 'lung':
        slide_path_1 = '/lila/data/fuchs/projects/lung/impacted/%s.svs' % args.id  # the name of the file
        slide_path_2 = '/lila/data/fuchs/projects/lung/de-identified_slides_from_luke_2_15_19/%s.svs' % args.id
        try:
            slide = openslide.OpenSlide(slide_path_1)
        except:
            slide = openslide.OpenSlide(slide_path_2)
    elif (args.dataset).split('_')[0] == 'breast':
        slide_path = '/lila/data/fuchs/projects/breast-infiltration/svs/%s.svs' % args.id
        slide = openslide.OpenSlide(slide_path)

    if (args.dataset).split('_')[0] == 'lung':
        til_file = open('/lila/data/fuchs/projects/lung/cell_nuclei_coords/%s_lymphocytes.csv' % args.id, 'w+')
        tumor_file = open('/lila/data/fuchs/projects/lung/cell_nuclei_coords/%s_tumorcells.csv' % args.id, 'w+')
    elif (args.dataset).split('_')[0] == 'breast':
        til_file = open(
            '/lila/data/fuchs/projects/breast-infiltration/cell_nuclei_coords/%s_lymphocytes.csv' % args.id, 'w+')
        tumor_file = open(
            '/lila/data/fuchs/projects/breast-infiltration/cell_nuclei_coords/%s_tumorcells.csv' % args.id, 'w+')

    if args.use_bmp:
        if args.annotation == 'AR':
            bmp_file = '/lila/data/fuchs/projects/lung/impacted/labels_AR/%s_label.bmp' % args.id
        elif args.annotation == 'IO':
            bmp_file = '/lila/data/fuchs/projects/lung/impacted/labels_IO/%s_label.bmp' % args.id
    else:
        bmp_file = None

    if args.annotation == 'AR':
        til_level_file = open('/lila/data/fuchs/projects/lung/cell_nuclei_coords/til_level_AR.csv', 'a+')
    if args.annotation == 'IO':
        til_level_file = open('/lila/data/fuchs/projects/lung/cell_nuclei_coords/til_level_IO.csv', 'a+')
    truegrid = extract_tissue.make_sample_grid(slide, patch_size=args.patch_size,
                                          mpp=0.5, power=None, min_cc_size=1,
                                          max_ratio_size=10, dilate=True,
                                          erode=False, prune=False,
                                          overlap=1, bmp=bmp_file,
                                          oversample=False)

    level, _ = extract_tissue.find_level(slide, 0.5, patchsize=args.patch_size)

    tile_number = 0
    til_count = 0
    for ws_coords in truegrid:
        since = time.time()
        image = slide.read_region(ws_coords, level, (args.patch_size, args.patch_size))
        image = np.array(image)
        image = image[:, :, :3]
        image = image.astype(float)

        image_datasets = LymphocytesTestImage(image, args.grid_size, args.input_size, data_transforms)

        dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=36, shuffle=True, num_workers=10)


        t_map = torch.zeros((args.num_classes, image.shape[0], image.shape[1]))
        suppression_mask = torch.zeros((args.num_classes, image.shape[0], image.shape[1]))
        hx_map = torch.zeros((64, image.shape[0], image.shape[1]))
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
                trans_preds[:, :args.num_classes, :, :].detach().cpu().numpy() + x_coord_matrix).astype(int)
            y_map = np.around(trans_preds[:, args.num_classes:2 * args.num_classes, :,
                              :].detach().cpu().numpy() + y_coord_matrix).astype(int)
            for c in range(args.num_classes):  # for each cell class
                for n in range(class_probs.shape[0]):  # batch size
                    x = x_map[n, c].reshape(-1)
                    y = y_map[n, c].reshape(-1)
                    valid_location_index = (x > 0) * (x < voca_map_batch.shape[-1]) * (y > 0) * (y < voca_map_batch.shape[-2])
                    x = x[valid_location_index]
                    y = y[valid_location_index]
                    # voca_map[n, c, y, x] += weighted_confidence_map[n, c, y, x].detach().cpu().numpy()
                    np.add.at(voca_map_batch[n, c], (y, x), weighted_confidence_map[n, c, y, x].detach().cpu().numpy())

            # nms on the map
            suppression_mask_batch = torch.zeros(voca_map_batch.shape)
            for c in range(args.num_classes):  # for each cell class
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
                                                                                     voca_map_batch.shape[-2]), \
                                                                                 max(int(coords[1]) - 2, 0):min(
                                                                                     int(coords[1]) + 3,
                                                                                     voca_map_batch.shape[-1])].sum()
            # normalize suppression_mask
            # minval = suppression_mask_batch.min(-2, keepdim=True)[0].min(-1, keepdim=True)[0]
            # maxval = suppression_mask_batch.max(-2, keepdim=True)[0].max(-1, keepdim=True)[0]
            minval = 0.0
            maxval = math.pi * args.r * args.r
            suppression_mask_batch -= minval
            suppression_mask_batch = suppression_mask_batch / (maxval - minval)

            for n in range(class_probs.shape[0]):  # batch size
                x = int(image_coords[n, 0])
                y = int(image_coords[n, 1])
                t_map[:, y:y + args.input_size, x:x + args.input_size] = \
                    t_preds[n].unsqueeze(-1).unsqueeze(-1).expand_as(weighted_confidence_map[n]).detach().cpu()
                suppression_mask[:, y:y + args.input_size, x:x + args.input_size] = \
                    suppression_mask_batch[n].detach().cpu()
                hx_map[:, y:y + args.input_size, x:x + args.input_size] = \
                    hx[n].detach()

        #threshold the peaks by t_map
        suppression_mask[suppression_mask < t_map] = 0
        #threshold baseline
        suppression_mask[suppression_mask < 0.05] = 0

        """
        write the coords to file
        """
        suppression_mask = suppression_mask.numpy()
        #write TILs
        plot_coords = np.transpose(np.nonzero(suppression_mask[1]))
        til_count += len(plot_coords)
        for dc in plot_coords:
            til_file.write(
                '%s,%s,%s,%s,%s\n' % (
                args.id, dc[1] + ws_coords[0],
                dc[0] + ws_coords[1],
                suppression_mask[1, dc[0], dc[1]], # channel 1 is til
                ','.join(hx_map[:, dc[0], dc[1]].numpy().astype(str))))
        # write tumor cells
        plot_coords = np.transpose(np.nonzero(suppression_mask[0]))
        for dc in plot_coords:
            tumor_file.write(
                '%s,%s,%s,%s,%s\n' % (
                args.id, dc[1] + ws_coords[0],
                dc[0] + ws_coords[1],
                suppression_mask[0, dc[0], dc[1]],
                ','.join(hx_map[:, dc[0], dc[1]].numpy().astype(str))))
        time_elapsed = time.time() - since
        print('tile %s/%s of %s completed in %s seconds!' % (tile_number, len(truegrid), args.id, time_elapsed))
        tile_number += 1

    til_level = til_count / tile_number
    til_level_file.write('%s,%s\n' % (args.id, til_level))
    til_level_file.close()
    til_file.close()
    tumor_file.close()

if __name__ == '__main__':
    main()



















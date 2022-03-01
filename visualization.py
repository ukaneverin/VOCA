import numpy as np
import glob
from imageio import imread, imsave
import os
import argparse
from pdb import set_trace

parser = argparse.ArgumentParser(description='VOCA testing')
parser.add_argument('--dataset', default='thyroid', type=str, help='lung_tt(which means til+tumor), lung_til, breast_til')
parser.add_argument('--output_path', default='/lila/data/fuchs/xiec/results/VOCA', type=str, help='working directory')
parser.add_argument('--num_classes', default=1, type=int, help='how many cell types')
parser.add_argument('--h', default=700, type=int, help='the size of the images in the dataset')
parser.add_argument('--w', default=700, type=int, help='the size of the images in the dataset')
parser.add_argument('--input_size', default=127, type=int, help='input image size to model')
parser.add_argument('--grid_size', default=127, type=int, help='grid size')
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
args = parser.parse_args()

val_save_path = os.path.join(args.output_path, args.dataset, 'val_results/r%s_lr%s_%s/' %
                             (args.r, args.lr, args.metric))

# get the list of all images
image_name_all = []
annotation_path = os.path.join(args.output_path, args.dataset, 'annotations/')
for file in glob.glob(annotation_path + '*.json'):
    if os.stat(file).st_size > 10:
        image_name_all.append(file.split('/')[-1].replace('.json', ''))

plot_directory = os.path.join(val_save_path, 'prediction_images/')
if not os.path.exists(plot_directory):
    os.makedirs(plot_directory)

"""
Color scheme: {0: Tumor cells, orange; 1: Lymphocytes, green}
"""
colors = {
    0: [255, 165, 0],
    1: [0, 255, 0]
}
for image_name in image_name_all:
    image = imread(os.path.join(args.output_path, args.dataset, 'images/%s' % image_name))
    image = image[:, :, :3]

    pred_map = np.load(os.path.join(val_save_path, 'pred_map/%s_final_map.npy' % image_name))

    for c in range(args.num_classes):
        coords = np.where(pred_map[c] > 0)
        for i in range(len(coords[0])):
            x = coords[1][i]
            y = coords[0][i]
            y_disk, x_disk = np.ogrid[-y: image.shape[0] - y, -x: image.shape[1] - x]
            disk_mask_within = y_disk ** 2 + x_disk ** 2 <= (args.nms_r + 1) ** 2
            disk_mask_without = y_disk ** 2 + x_disk ** 2 >= (args.nms_r) ** 2
            disk_mask = disk_mask_within * disk_mask_without
            image[disk_mask, :] = colors[c]

    imsave(plot_directory + '/%s_predictions.png' % image_name, image)

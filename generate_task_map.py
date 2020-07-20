import numpy as np
import glob
from imageio import imread,imwrite
import argparse
import os
from pdb import set_trace
import json
parser = argparse.ArgumentParser(description='VOCA task map generation from the annotation files')
parser.add_argument('--dataset', default='lung_tt', type=str, help='lung_tt(which means til+tumor), lung_til, breast_til')
parser.add_argument('--output_path', default='/lila/data/fuchs/xiec/results/VOCA', type=str, help='working directory')
parser.add_argument('--num_classes', default=2, type=int, help='how many cell types')
parser.add_argument('--input_size', default=127, type=int, help='input image size to model')
parser.add_argument('--r', default=12, type=int, help='the only hyperparameter r')
parser.add_argument('--cv', default=5, type=int, help='how many folds of cross validation')
args = parser.parse_args()

task_map_save_path = os.path.join(args.output_path, args.dataset, 'task_map/')
if not os.path.exists(task_map_save_path):
    os.mkdir(task_map_save_path)

gt_map_save_path = os.path.join(args.output_path, args.dataset, 'gt_map/')
if not os.path.exists(gt_map_save_path):
    os.mkdir(gt_map_save_path)

image_list = []
annotation_path = os.path.join(args.output_path, args.dataset, 'annotations/')
for file in glob.glob(annotation_path+'*.json'):
    if os.stat(file).st_size > 10: # no annotation; note that there might be tiles with no cells. but not in our case
        image_list.append(file.split('/')[-1].replace('.json', ''))

split_save_path = os.path.join(args.output_path, args.dataset, 'split/')
if not os.path.exists(split_save_path):
    os.mkdir(split_save_path)

#get the cross valication split for image names and patch library
for i in range(args.cv):

    val_list = image_list[i * int(len(image_list)/args.cv) : (i+1) * int(len(image_list)/args.cv)]
    train_list = list(set(image_list) - set(val_list))

    with open(os.path.join(split_save_path, 'CV_%s_train.csv' % i), 'w+') as train_file:
        for img in train_list:
            train_file.write(img + '\n')
    with open(os.path.join(split_save_path, 'CV_%s_val.csv' % i), 'w+') as val_file:
        for img in val_list:
            val_file.write(img + '\n')

    patch_library_train = open(os.path.join(split_save_path, 'library_%s_train.csv' % i), 'w+')
    for image_name in train_list:
        image = imread(os.path.join(args.output_path, args.dataset, 'images/%s' % image_name))
        for crop_x in range(0, image.shape[1]-args.input_size, 17):
            for crop_y in range(0, image.shape[0]-args.input_size, 17):
                patch_library_train.write('%s,%s,%s\n' % (image_name,crop_x,crop_y))
    patch_library_train.close()

    patch_library_val = open(os.path.join(split_save_path, 'library_%s_val.csv' % i), 'w+')
    for image_name in val_list:
        image = imread(os.path.join(args.output_path, args.dataset, 'images/%s' % image_name))
        for crop_x in range(0, image.shape[1] - args.input_size, 17):
            for crop_y in range(0, image.shape[0] - args.input_size, 17):
                patch_library_val.write('%s,%s,%s\n' % (image_name, crop_x, crop_y))
    patch_library_val.close()

#generate the task maps
for image_name in image_list:
    image_path = os.path.join(args.output_path, args.dataset, 'images/%s' % image_name)
    image =  imread(image_path)
    image = image[:,:,:3]
    image_label_file =  os.path.join(args.output_path, args.dataset, 'annotations/%s.json' % image_name)
    with open(image_label_file, 'r') as f:
        coords_data = json.load(f)

    #create a mask: 0: background; 1: cell; c channels, each channel represent one cell type
    gt_map = np.zeros((image.shape[0], image.shape[1], args.num_classes))
    task_map = np.zeros((image.shape[0], image.shape[1], args.num_classes * 4))
    dist_map = np.ones((image.shape[0], image.shape[1], args.num_classes))*\
               ((image.shape[0])**2 + (image.shape[1])**2)
    print("Generating task_map for image: %s" % image_name)
    for p in coords_data:
        x=max(min(int(p['x']), image.shape[1]-1), 0) #some times annotation is off image lmao
        y=max(min(int(p['y']), image.shape[0]-1), 0)
        c = int(p['class'])
        #create the point map
        gt_map[y,x,c] = 1 # y is axis 0, x is axis 1; y = vertical, x = horizontal; (0,0) is top left

        #create the task maps------------------------
        # cls_map
        y_disk, x_disk = np.ogrid[-y : image.shape[0] - y, -x : image.shape[1] - x]
        disk_mask = y_disk**2 + x_disk**2 <= (args.r)**2
        task_map[disk_mask, c] = 1

        #x, y map; recording a dist_map to know whether should update to current ground truth
        x_coord_matrix = np.array([np.arange(image.shape[1]),]*image.shape[0])
        y_coord_matrix = np.array([np.arange(image.shape[0]),]*image.shape[1]).transpose()
        x_map = x - x_coord_matrix
        y_map = y - y_coord_matrix
        #update mask is where in disk mask the distance to the current cell is smaller than previous smallest dist
        update_mask = ((x_map**2 + y_map**2) < dist_map[:,:,c]) * disk_mask
        #now we update the dist_map to the smallest dist
        dist_map[disk_mask,c] = np.minimum((x_map**2 + y_map**2)[disk_mask], dist_map[disk_mask,c])
        task_map[update_mask,args.num_classes+c] = x_map[update_mask]
        task_map[update_mask, 2*args.num_classes + c] = y_map[update_mask]

        #wt map
        task_map[disk_mask, 3*args.num_classes + c] += 1

    task_map_save_file = os.path.join(task_map_save_path, '%s_r%s.npy' % (image_name, args.r))
    np.save(task_map_save_file, task_map)
    gt_map_save_file = os.path.join(gt_map_save_path, '%s.npy' % image_name)
    np.save(gt_map_save_file, gt_map)
    # imwrite('test_task_map.png', task_map[:, :, c])
    # imwrite('test_dist_map.png', dist_map[:, :, c])
    # imwrite('test_x_map.png', task_map[:, :, args.num_classes + c])
    # imwrite('test_y_map.png', task_map[:, :, 2 * args.num_classes + c])
    # imwrite('test_wt_map.png', task_map[:, :, 3 * args.num_classes +c])



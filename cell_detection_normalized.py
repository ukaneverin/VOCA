import numpy as np
from imageio import imread,imsave
import glob
import os
from models import * #models defined
import torch
from torch.autograd import Variable
from torchvision import transforms
import time
from scipy.optimize import linear_sum_assignment as lsa
from scipy.spatial.distance import cdist
from Dataset_classes import LymphocytesTestImage
from skimage.util import invert
from skimage import img_as_ubyte
import math
import argparse
from pdb import set_trace
parser = argparse.ArgumentParser(description='VOCA testing')
parser.add_argument('--dataset', default='lung_tt', type=str, help='lung_tt(which means til+tumor), lung_til, breast_til')
parser.add_argument('--output_path', default='/lila/data/fuchs/xiec/results/VOCA', type=str, help='working directory')
parser.add_argument('--num_classes', default=2, type=int, help='how many cell types')
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

def main():
    global args
    global device
    args = parser.parse_args()

    val_save_path = os.path.join(args.output_path, args.dataset, 'val_results_%s_%s_r%s_lr%s_%s/' %
                                 (args.model, args.loss, args.r, args.lr, args.metric))
    if not os.path.exists(val_save_path):
        os.makedirs(val_save_path)
    if not os.path.exists(os.path.join(val_save_path, 'voca_map')):
        os.makedirs(os.path.join(val_save_path, 'voca_map'))
    if not os.path.exists(os.path.join(val_save_path, 'pred_map')):
        os.makedirs(os.path.join(val_save_path, 'pred_map'))

    #get the list of all images
    image_name_all = []
    annotation_path = os.path.join(args.output_path, args.dataset, 'annotations/')
    for file in glob.glob(annotation_path + '*.json'):
        if os.stat(file).st_size > 10:
            image_name_all.append(file.split('/')[-1].replace('.json', ''))
    suppression_mask_all = torch.zeros((len(image_name_all), args.num_classes, args.h, args.w))
    gt_map_all = torch.zeros((len(image_name_all), args.num_classes, args.h, args.w))
    processed_images = []
    for cv in range(5):
        #data transform with learned mean and std
        mean = torch.FloatTensor(np.load(os.path.join(args.output_path, args.dataset, 'split/CV_%s_mean.npy' % cv)))
        std = torch.FloatTensor(np.load(os.path.join(args.output_path, args.dataset, 'split/CV_%s_std.npy' % cv)))

        data_transforms = transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize(mean, std)
                          ])

        # get device and load best model for eval
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if args.model == 'vgg':
            model = cell_vgg_cls_reg('config2').to(device)
        elif args.model == 'vggsplit':
            model = cell_vgg_cls_reg_split('config2').to(device)
        elif args.model == 'vggsub':
            model = cell_vgg_cls_reg_sub('config2').to(device)
        elif args.model == 'vggskip':
            model = cell_vgg_cls_skip('config2').to(device)
        elif args.model == 'res':
            model = ResNet18().to(device)
        elif args.model == 'dense':
            model = DenseNet121().to(device)

        model_directory = os.path.join(args.output_path, args.dataset, 'trained_models/')
        model.load_state_dict(torch.load(model_directory+'%s_%s_r%s_lr%s_%s_cv%s.pth' %
                                         (args.model, args.loss, args.r, args.lr, args.metric, cv)))
        model.eval()

        with open(os.path.join(args.output_path, args.dataset, 'split/CV_%s_val.csv' % cv), 'r') as f:
            val_set_list = [line.strip() for line in f]


        for image_name in val_set_list:
            processed_images.append(image_name)
            image_file = os.path.join(args.output_path, args.dataset, 'images/', image_name)
            image =  imread(image_file)
            image = image[:,:,:3]
            image = image.astype(float)

            gt_map = np.load(os.path.join(args.output_path, args.dataset, 'gt_map/', '%s.npy' % image_name))
            gt_map_all[image_name_all.index(image_name)] = torch.FloatTensor(gt_map).permute(2,0,1)
            image_datasets = LymphocytesTestImage(image, args.grid_size, args.input_size, data_transforms)

            dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=36, shuffle=True, num_workers=6)

            trans_map_x = torch.zeros((args.num_classes, image.shape[0], image.shape[1]))
            trans_map_y = torch.zeros((args.num_classes, image.shape[0], image.shape[1]))
            wt_mask = torch.zeros((args.num_classes, image.shape[0], image.shape[1]))
            t_map = torch.zeros((args.num_classes, image.shape[0], image.shape[1]))
            suppression_mask = torch.zeros((args.num_classes, image.shape[0], image.shape[1]))
            voca_map = torch.zeros((args.num_classes, image.shape[0], image.shape[1]))
            for sample in dataloaders:
                inputs = sample['image_crop']
                image_coords = sample['coords'].numpy()
                inputs = inputs.to(device)
                # forward
                with torch.no_grad():
                    class_probs, trans_preds, num_preds, t_preds = model(inputs)
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
                    if args.vis:
                        wt_mask[:, y:y+args.input_size, x:x+args.input_size] = \
                            weighted_confidence_map[n].detach().cpu()
                        trans_map_x[:, y:y+args.input_size, x:x+args.input_size] = \
                            trans_preds[n,:args.num_classes].detach().cpu()
                        trans_map_y[:, y:y + args.input_size, x:x + args.input_size] = \
                            trans_preds[n, args.num_classes:2 * args.num_classes].detach().cpu()
                        voca_map[:, y:y + args.input_size, x:x + args.input_size] = \
                            torch.FloatTensor(voca_map_batch[n])
                    t_map[:, y:y + args.input_size, x:x + args.input_size] = \
                        t_preds[n].unsqueeze(-1).unsqueeze(-1).expand_as(weighted_confidence_map[n]).detach().cpu()
                    suppression_mask[:, y:y + args.input_size, x:x + args.input_size] = \
                        suppression_mask_batch[n].detach().cpu()


            if args.vis:
                for c in range(args.num_classes):
                    imsave(val_save_path+'voca_map/%s_voca_map_%s.png' % (image_name, c), voca_map[c])
                    imsave(val_save_path+'voca_map/%s_trans_x_%s.png' % (image_name, c), trans_map_x[c])
                    imsave(val_save_path+'voca_map/%s_trans_y_%s.png' % (image_name, c), trans_map_y[c])
                    imsave(val_save_path+'voca_map/%s_wt_map_%s.png' % (image_name, c), wt_mask[c])
                    imsave(val_save_path + 'voca_map/%s_t_map_%s.png' % (image_name, c), t_map[c])
                    imsave(val_save_path + 'voca_map/%s_peak_map_%s.png' % (image_name, c), suppression_mask[c])

            #threshold the peaks by t_map
            suppression_mask[suppression_mask < t_map] = 0
            suppression_mask[suppression_mask < 0.05] = 0
            if args.vis:
                for c in range(args.num_classes):
                    imsave(val_save_path + 'voca_map/%s_final_map_%s.png' % (image_name, c), suppression_mask[c])
            np.save(val_save_path + 'pred_map/%s_final_map.npy' % image_name, suppression_mask)
            suppression_mask_all[image_name_all.index(image_name)] = suppression_mask

    true_positive, pred_num, gt_num, pair_distances = calc_f1(gt_map_all,
                                                              suppression_mask_all,
                                                              r=6)
    # get the image and channel wise fscore
    precision = np.divide(true_positive, pred_num, out=np.zeros(true_positive.shape),
                          where=(pred_num) != 0)
    recall = np.divide(true_positive, gt_num, out=np.zeros(true_positive.shape), where=(gt_num) != 0)
    fscore = 2 * np.divide((precision * recall), (precision + recall),
                           out=np.zeros(true_positive.shape), where=(precision + recall) != 0)
    fscore = np.delete(fscore, image_name_all.index(list(set(image_name_all) - set(processed_images))[0]), axis=0)
    print('Fscore (n,c) ---- ')
    print(fscore)
    # get the total fscore
    precision_all = true_positive.sum(0) / pred_num.sum(0).numpy()
    recall_all = true_positive.sum(0) / gt_num.sum(0)
    fscore_all = 2 * (precision_all * recall_all) / (precision_all + recall_all)
    print('Precision: ', precision_all)
    print('Recall: ', recall_all)
    print('Fscore: ', fscore_all)

def non_max_suppression(points, disThresh):
    # if there are no points, return an empty list
    if len(points) == 0:
        return np.asarray([]).reshape(-1,3)

    if points.dtype.kind == 'i':
        points = points.astype('float')
    np.random.shuffle(points)

    pick = []

    x = points[:,1]
    y = points[:,0]
    score = points[:,2]

    idxs = np.argsort(score)

    while len(idxs) > 0:

        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        x_dis = x[idxs[:last]] -  x[i]
        y_dis = y[idxs[:last]] -  y[i]

        dis = (x_dis**2+y_dis**2)**0.5
        idxs = np.delete(idxs, np.concatenate(([last],
        	np.where(dis < disThresh)[0])))

    return points[pick]

def remove_non_assigned(pred_map, gt, r=6):
    pair_distances = []
    for c in range(args.num_classes):
        for n in range(gt.shape[0]):
            '''obtain the coords for pred and gt'''
            gt_coords_tuple = np.where(gt[n,c] > 0)
            gt_coords = np.asarray((gt_coords_tuple[0], gt_coords_tuple[1])).transpose()
            pred_coords_tuple = np.where(pred_map[n,c] > 0)
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
            pred_map[n, c, removed_preds_coords[:,0], removed_preds_coords[:,1]] = 0
            '''delete the assigned but too far points from pred_map'''
            too_far_points = pred_coords[pred_ind[pred_x_gt[pred_ind, gt_ind] > r]].numpy()
            pred_map[n,c,too_far_points[:,0],too_far_points[:,1]] = 0
            pair_distances += pred_x_gt[pred_ind, gt_ind].reshape(-1).tolist()
    return pred_map, pair_distances


def calc_f1(gt, pred_map, r=6, assign = True):
    #gt_num, pred_num, all are of shape len(images) * num_classes
    gt_num = gt.sum((-1,-2)).numpy()
    tp = np.zeros((gt.shape[0], gt.shape[1]))
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

    pred_num = pred_map.sum((-2,-1))
    if assign == True:
        pred_map = np.minimum(pred_map,gt_map)
        pred_map, pair_distances = remove_non_assigned(pred_map, gt, r)

    result_map = gt_map * pred_map.numpy()
    tp = result_map.sum((-2,-1))
    if assign == True:
        return tp, pred_num, gt_num, pair_distances
    return tp, pred_num, gt_num

if __name__ == '__main__':
    main()
from __future__ import print_function, division
import torch.optim as optim
from torch.optim import lr_scheduler
import subprocess
from tqdm import tqdm
from torchvision import transforms
import os
from Dataset_classes import VocaData, TrainImagesDataset, TestImageDataset
from models.res import VOCA_Res
from voca_transforms import *
from pdb import set_trace
from utils import *
import argparse
from train import train_model
from detection_validation import detection_validation
from detection_test import detection_roi, detection_ws

parser = argparse.ArgumentParser(description='VOCA training')
parser.add_argument('--dataset', default='thyroid', type=str, help='lung_tt')
parser.add_argument('--image_folder', default='/lila/data/fuchs/xiec/results/VOCA/thyroid_images', type=str,
                    help='lung_tt: /lila/data/fuchs/xiec/results/VOCA/lung_tt/images')
parser.add_argument('--n_cv', default=5, type=int, help='K-fold cross validation')
parser.add_argument('--cv', default=0, type=int, help='which cross validation fold')
parser.add_argument('--output_path', default='/lila/data/fuchs/xiec/results/VOCA', type=str, help='working directory')

parser.add_argument('--num_classes', default=1, type=int, help='how many cell types; lung_tt:2')
parser.add_argument('--input_size', default=127, type=int, help='input image size to model')

parser.add_argument('--r', default=12, type=int, help='disk size')
parser.add_argument('--nms_r', default=6, type=int, help='the distance threshold for non-max suppresion')
parser.add_argument('--num_workers', default=8, type=int, help='how many workers for dataloader')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--batch_size', default=8, type=int, help='bactch size')
parser.add_argument('--n_epochs', default=10, type=int)
parser.add_argument('--subsample', default=0.1, type=float, help='percentage of sampled tiles per epoch')
parser.add_argument('--metric', default='f1', type=str,
                    help='acc, f1; this tells whether to compute the \
                    f1 score for every batch, and save best model based on which metric')
parser.add_argument('--slide_file', type=str, help='path to the .svs slide to run VOCA on')
parser.add_argument('--mask_file', type=str, help='path to the mask of the slide; specify as None if no mask')
parser.add_argument('--stage', default='train', type=str,
                    help='stage in the pipeline: [process, train, validation, roi_inference, ws_inference]')


def main():
    args = parser.parse_args()
    # Define a customized function to get the annotations

    class MyVocaData(VocaData):
        def get_annotation(self):
            # url = 'https://slides-res.mskcc.org/slides/vanderbc@mskcc.org/40;crops_tumor_til'
            url = 'https://slides-res.mskcc.org/slides/kezlarib@mskcc.org/97;crops'
            for i in tqdm(range(0, len(self.image_list))):
                file_path = self.annotation_path + self.image_list[i] + '.json'
                print("Download cell annotation for: ", self.image_list[i])
                subprocess.call(
                    ['wget -l 0 "%s;%s/getSVGLabels/nucleus" -O %s' % (url, self.image_list[i].lower(), file_path)],
                    shell=True)

    root = os.path.join(args.output_path, args.dataset)
    # specify the dictionary of prediction channels to cell types; e.g. {0: 'tumor', 1: 'lymphocytes'}
    cell_type_dict = {
        0: 'all'
    }
    voca_data = MyVocaData(root, args, cell_type_dict)

    if args.stage == 'process':
        voca_data.process()

    elif args.stage == 'train':
        # load or calculate the mean and standard deviation of rgb channels of training image crops,
        # and ratio of positive/negative pixels for each cell type
        mean, std, sample_sizes = voca_data.get_mean_std(args.cv)
        data_transforms = {
            'train': transforms.Compose([
                # can add more augmentations here for training data but need to modify the voca_transforms script,
                # since augmentation sometimes need to apply on the task maps
                RandomVerticalFlip(),
                RandomHorizontalFlip(),
                RandomRotate(),
                CorlorJitter(),
                ToTensor(),
                Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                ToTensor(),
                Normalize(mean, std)
            ])
        }
        image_datasets = {phase: TrainImagesDataset(phase, data_transforms[phase], args.cv, args.subsample, **voca_data.__dict__)
                          for phase in ['train', 'val']}
        dataloaders = {phase: torch.utils.data.DataLoader(image_datasets[phase], batch_size=args.batch_size,
                                                          shuffle=True, num_workers=args.num_workers)
                       for phase in ['train', 'val']}
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = VOCA_Res(num_classes=args.num_classes).to(device)
        optimizer_ft = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9)
        # Decay LR by a factor of 0.1 every 3 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=int((args.n_epochs - 1) / 3), gamma=0.1)
        model = train_model(model, optimizer_ft, exp_lr_scheduler, dataloaders, sample_sizes, args, voca_data)
        torch.save(model.state_dict(), os.path.join(
            voca_data.model_directory, 'r%s_lr%s_%s_cv%s.pth' % (args.r, args.lr, args.metric, args.cv)))

    elif args.stage == 'validation':
        detection_validation(args, voca_data)

    elif args.stage == 'roi_inference':
        import glob
        # roi_list = glob.glob('/lila/data/fuchs/xiec/results/VOCA/thyroid_test_roi/*.png')
        # prediction_directory = '/lila/data/fuchs/xiec/results/VOCA/thyroid_test_roi/predictions'
        roi_list = glob.glob('/lila/data/fuchs/xiec/results/VOCA/thyroid_test_roi_smear/*.png')
        prediction_directory = '/lila/data/fuchs/xiec/results/VOCA/thyroid_test_roi_smear/predictions'
        for image_file in roi_list:
            detection_roi(image_file, args, voca_data, prediction_directory)

    elif args.stage == 'ws_inference':
        prediction_directory = '/lila/data/fuchs/xiec/results/VOCA/thyroid_svs_predictions'
        detection_ws(args, voca_data, prediction_directory)

if __name__ == '__main__':
    main()

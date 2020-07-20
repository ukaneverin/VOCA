from os import system
import os.path
import time

if not os.path.exists('crop_aug_patch.log'):
    system('bsub -n 2 -W 2:00 -q gpuqueue -gpu "num=1" -o crop_aug_patch.log python3 crop_augmentation_patch.py')
    print('bsub -n 2 -W 2:00 -q gpuqueue -gpu "num=1" -o crop_aug_patch.log python3 crop_augmentation_patch.py')

while not os.path.exists('crop_aug_patch.log'):
    time.sleep(10)
if not os.path.exists('sharp_relabel_0.5.log'):
    system('bsub -n 2 -W 1:00 -q gpuqueue -gpu "num=1" -o sharp_relabel_0.5.log python3 sharp_relabel.py 0.5')
    print('bsub -n 2 -W 1:00 -q gpuqueue -gpu "num=1" -o sharp_relabel_0.5.log python3 sharp_relabel.py 0.5')

while not os.path.exists('sharp_relabel_0.5.log'):
    time.sleep(10)
if not os.path.exists('sharp_relabel_0.8.log'):
    system('bsub -n 2 -W 1:00 -q gpuqueue -gpu "num=1" -o sharp_relabel_0.8.log python3 sharp_relabel.py 0.8')
    print('bsub -n 2 -W 1:00 -q gpuqueue -gpu "num=1" -o sharp_relabel_0.8.log python3 sharp_relabel.py 0.8')

while not os.path.exists('sharp_relabel_0.8.log'):
    time.sleep(10)
if not os.path.exists('sharp_relabel_0.3.log'):
    system('bsub -n 2 -W 1:00 -q gpuqueue -gpu "num=1" -o sharp_relabel_0.3.log python3 sharp_relabel.py 0.3')
    print('bsub -n 2 -W 1:00 -q gpuqueue -gpu "num=1" -o sharp_relabel_0.3.log python3 sharp_relabel.py 0.3')


while not os.path.exists('sharp_relabel_0.5.log'):
    time.sleep(10)
if not os.path.exists('calc_ms.log'):
    system('bsub -n 2 -W 1:00 -q gpuqueue -gpu "num=1" -o calc_ms.log python3 patch_train_cls.py calc_ms 0.5')
    print('bsub -n 2 -W 1:00 -q gpuqueue -gpu "num=1" -o calc_ms.log python3 patch_train_cls.py calc_ms 0.5')


while not os.path.exists('calc_ms.log'):
    time.sleep(10)
if not os.path.exists('cls_0.5.log'):
    system('bsub -n 4 -W 6:00 -q gpuqueue -gpu "num=1" -o cls_0.5.log python3 patch_train_cls.py train 0.5')
    print('bsub -n 4 -W 6:00 -q gpuqueue -gpu "num=1" -o cls_0.5.log python3 patch_train_cls.py train 0.5')
if not os.path.exists('joint_0.5.log'):
    system('bsub -n 4 -W 6:00 -q gpuqueue -gpu "num=1" -o joint_0.5.log python3 joint_train.py train 0.5')
    print('bsub -n 4 -W 6:00 -q gpuqueue -gpu "num=1" -o joint_0.5.log python3 joint_train.py train 0.5')
if not os.path.exists('cls_0.8.log'):
    system('bsub -n 4 -W 6:00 -q gpuqueue -gpu "num=1" -o cls_0.8.log python3 patch_train_cls.py train 0.8')
    print('bsub -n 4 -W 6:00 -q gpuqueue -gpu "num=1" -o cls_0.8.log python3 patch_train_cls.py train 0.8')
if not os.path.exists('joint_0.8.log'):
    system('bsub -n 4 -W 6:00 -q gpuqueue -gpu "num=1" -o joint_0.8.log python3 joint_train.py train 0.8')
    print('bsub -n 4 -W 6:00 -q gpuqueue -gpu "num=1" -o joint_0.8.log python3 joint_train.py train 0.8')
if not os.path.exists('cls_0.3.log'):
    system('bsub -n 4 -W 6:00 -q gpuqueue -gpu "num=1" -o cls_0.3.log python3 patch_train_cls.py train 0.3')
    print('bsub -n 4 -W 6:00 -q gpuqueue -gpu "num=1" -o cls_0.3.log python3 patch_train_cls.py train 0.3')
if not os.path.exists('joint_0.3.log'):
    system('bsub -n 4 -W 6:00 -q gpuqueue -gpu "num=1" -o joint_0.3.log python3 joint_train.py train 0.3')
    print('bsub -n 4 -W 6:00 -q gpuqueue -gpu "num=1" -o joint_0.3.log python3 joint_train.py train 0.3')


while not os.path.exists('cls_0.5.log'):
    time.sleep(10)
if not os.path.exists('tune_0.5.log'):
    system('bsub -n 4 -W 6:00 -q gpuqueue -gpu "num=1" -o tune_0.5.log python3 fine_tune_regression.py train 0.5')
    print('bsub -n 4 -W 6:00 -q gpuqueue -gpu "num=1" -o tune_0.5.log python3 fine_tune_regression.py train 0.5')

while not os.path.exists('cls_0.8.log'):
    time.sleep(10)
if not os.path.exists('tune_0.8.log'):
    system('bsub -n 4 -W 6:00 -q gpuqueue -gpu "num=1" -o tune_0.8.log python3 fine_tune_regression.py train 0.8')
    print('bsub -n 4 -W 6:00 -q gpuqueue -gpu "num=1" -o tune_0.8.log python3 fine_tune_regression.py train 0.8')

while not os.path.exists('cls_0.3.log'):
    time.sleep(10)
if not os.path.exists('tune_0.3.log'):
    system('bsub -n 4 -W 6:00 -q gpuqueue -gpu "num=1" -o tune_0.3.log python3 fine_tune_regression.py train 0.3')
    print('bsub -n 4 -W 6:00 -q gpuqueue -gpu "num=1" -o tune_0.3.log python3 fine_tune_regression.py train 0.3')

while not os.path.exists('tune_0.5.log'):
    time.sleep(10)
if not os.path.exists('learn_hyper_cls_0.5.log'):
    system('bsub -n 2 -W 2:00 -R rusage[mem=8] -q gpuqueue -gpu "num=1" -o learn_hyper_cls_0.5.log python3 learn_hyper.py cls 0.5')
    print('bsub -n 2 -W 2:00 -R rusage[mem=8] -q gpuqueue -gpu "num=1" -o learn_hyper_cls_0.5.log python3 learn_hyper.py cls 0.5')
if not os.path.exists('learn_hyper_tune_0.5.log'):
    system('bsub -n 2 -W 2:00 -R rusage[mem=8] -q gpuqueue -gpu "num=1" -o learn_hyper_tune_0.5.log python3 learn_hyper.py tune 0.5')
    print('bsub -n 2 -W 2:00 -R rusage[mem=8] -q gpuqueue -gpu "num=1" -o learn_hyper_tune_0.5.log python3 learn_hyper.py tune 0.5')
if not os.path.exists('learn_hyper_joint_0.5.log'):
    system('bsub -n 2 -W 2:00 -R rusage[mem=8] -q gpuqueue -gpu "num=1" -o learn_hyper_joint_0.5.log python3 learn_hyper.py joint 0.5')
    print('bsub -n 2 -W 2:00 -R rusage[mem=8] -q gpuqueue -gpu "num=1" -o learn_hyper_joint_0.5.log python3 learn_hyper.py joint 0.5')

while not os.path.exists('tune_0.8.log'):
    time.sleep(10)
if not os.path.exists('learn_hyper_cls_0.8.log'):
    system('bsub -n 2 -W 2:00 -R rusage[mem=8] -q gpuqueue -gpu "num=1" -o learn_hyper_cls_0.8.log python3 learn_hyper.py cls 0.8')
    print('bsub -n 2 -W 2:00 -R rusage[mem=8] -q gpuqueue -gpu "num=1" -o learn_hyper_cls_0.8.log python3 learn_hyper.py cls 0.8')
if not os.path.exists('learn_hyper_tune_0.8.log'):
    system('bsub -n 2 -W 2:00 -R rusage[mem=8] -q gpuqueue -gpu "num=1" -o learn_hyper_tune_0.8.log python3 learn_hyper.py tune 0.8')
    print('bsub -n 2 -W 2:00 -R rusage[mem=8] -q gpuqueue -gpu "num=1" -o learn_hyper_tune_0.8.log python3 learn_hyper.py tune 0.8')
if not os.path.exists('learn_hyper_joint_0.8.log'):
    system('bsub -n 2 -W 2:00 -R rusage[mem=8] -q gpuqueue -gpu "num=1" -o learn_hyper_joint_0.8.log python3 learn_hyper.py joint 0.8')
    print('bsub -n 2 -W 2:00 -R rusage[mem=8] -q gpuqueue -gpu "num=1" -o learn_hyper_joint_0.8.log python3 learn_hyper.py joint 0.8')

while not os.path.exists('tune_0.3.log'):
    time.sleep(10)
if not os.path.exists('learn_hyper_cls_0.3.log'):
    system('bsub -n 2 -W 2:00 -R rusage[mem=8] -q gpuqueue -gpu "num=1" -o learn_hyper_cls_0.3.log python3 learn_hyper.py cls 0.3')
    print('bsub -n 2 -W 2:00 -R rusage[mem=8] -q gpuqueue -gpu "num=1" -o learn_hyper_cls_0.3.log python3 learn_hyper.py cls 0.3')
if not os.path.exists('learn_hyper_tune_0.3.log'):
    system('bsub -n 2 -W 2:00 -R rusage[mem=8] -q gpuqueue -gpu "num=1" -o learn_hyper_tune_0.3.log python3 learn_hyper.py tune 0.3')
    print('bsub -n 2 -W 2:00 -R rusage[mem=8] -q gpuqueue -gpu "num=1" -o learn_hyper_tune_0.3.log python3 learn_hyper.py tune 0.3')
if not os.path.exists('learn_hyper_joint_0.3.log'):
    system('bsub -n 2 -W 2:00 -R rusage[mem=8] -q gpuqueue -gpu "num=1" -o learn_hyper_joint_0.3.log python3 learn_hyper.py joint 0.3')
    print('bsub -n 2 -W 2:00 -R rusage[mem=8] -q gpuqueue -gpu "num=1" -o learn_hyper_joint_0.3.log python3 learn_hyper.py joint 0.3')

while not os.path.exists('learn_hyper_cls_0.5.log'):
    time.sleep(10)
if not os.path.exists('detection_cls_0.5.log'):
    system('bsub -n 2 -W 2:00 -R rusage[mem=8] -q gpuqueue -gpu "num=1" -o detection_cls_0.5.log python3 cell_detection_suppression.py cls 0.5')
    print('bsub -n 2 -W 2:00 -R rusage[mem=8] -q gpuqueue -gpu "num=1" -o detection_cls_0.5.log python3 cell_detection_suppression.py cls 0.5')

while not os.path.exists('learn_hyper_cls_0.8.log'):
    time.sleep(10)
if not os.path.exists('detection_cls_0.8.log'):
    system('bsub -n 2 -W 2:00 -R rusage[mem=8] -q gpuqueue -gpu "num=1" -o detection_cls_0.8.log python3 cell_detection_suppression.py cls 0.8')
    print('bsub -n 2 -W 2:00 -R rusage[mem=8] -q gpuqueue -gpu "num=1" -o detection_cls_0.8.log python3 cell_detection_suppression.py cls 0.8')

while not os.path.exists('learn_hyper_cls_0.3.log'):
    time.sleep(10)
if not os.path.exists('detection_cls_0.3.log'):
    system('bsub -n 2 -W 2:00 -R rusage[mem=8] -q gpuqueue -gpu "num=1" -o detection_cls_0.3.log python3 cell_detection_suppression.py cls 0.3')
    print('bsub -n 2 -W 2:00 -R rusage[mem=8] -q gpuqueue -gpu "num=1" -o detection_cls_0.3.log python3 cell_detection_suppression.py cls 0.3')

while not os.path.exists('learn_hyper_joint_0.5.log'):
    time.sleep(10)
if not os.path.exists('detection_joint_0.5.log'):
    system('bsub -n 2 -W 2:00 -R rusage[mem=8] -q gpuqueue -gpu "num=1" -o detection_joint_0.5.log python3 cell_detection_suppression.py joint 0.5')
    print('bsub -n 2 -W 2:00 -R rusage[mem=8] -q gpuqueue -gpu "num=1" -o detection_joint_0.5.log python3 cell_detection_suppression.py joint 0.5')

while not os.path.exists('learn_hyper_joint_0.8.log'):
    time.sleep(10)
if not os.path.exists('detection_joint_0.8.log'):
    system('bsub -n 2 -W 2:00 -R rusage[mem=8] -q gpuqueue -gpu "num=1" -o detection_joint_0.8.log python3 cell_detection_suppression.py joint 0.8')
    print('bsub -n 2 -W 2:00 -R rusage[mem=8] -q gpuqueue -gpu "num=1" -o detection_joint_0.8.log python3 cell_detection_suppression.py joint 0.8')

while not os.path.exists('learn_hyper_joint_0.3.log'):
    time.sleep(10)
if not os.path.exists('detection_joint_0.3.log'):
    system('bsub -n 2 -W 2:00 -R rusage[mem=8] -q gpuqueue -gpu "num=1" -o detection_joint_0.3.log python3 cell_detection_suppression.py joint 0.3')
    print('bsub -n 2 -W 2:00 -R rusage[mem=8] -q gpuqueue -gpu "num=1" -o detection_joint_0.3.log python3 cell_detection_suppression.py joint 0.3')

while not os.path.exists('learn_hyper_tune_0.5.log'):
    time.sleep(10)
if not os.path.exists('detection_tune_0.5.log'):
    system('bsub -n 2 -W 2:00 -R rusage[mem=8] -q gpuqueue -gpu "num=1" -o detection_tune_0.5.log python3 cell_detection_suppression.py tune 0.5')
    print('bsub -n 2 -W 2:00 -R rusage[mem=8] -q gpuqueue -gpu "num=1" -o detection_tune_0.5.log python3 cell_detection_suppression.py tune 0.5')

while not os.path.exists('learn_hyper_tune_0.8.log'):
    time.sleep(10)
if not os.path.exists('detection_tune_0.8.log'):
    system('bsub -n 2 -W 2:00 -R rusage[mem=8] -q gpuqueue -gpu "num=1" -o detection_tune_0.8.log python3 cell_detection_suppression.py tune 0.8')
    print('bsub -n 2 -W 2:00 -R rusage[mem=8] -q gpuqueue -gpu "num=1" -o detection_tune_0.8.log python3 cell_detection_suppression.py tune 0.8')

while not os.path.exists('learn_hyper_tune_0.3.log'):
    time.sleep(10)
if not os.path.exists('detection_tune_0.3.log'):
    system('bsub -n 2 -W 2:00 -R rusage[mem=8] -q gpuqueue -gpu "num=1" -o detection_tune_0.3.log python3 cell_detection_suppression.py tune 0.3')
    print('bsub -n 2 -W 2:00 -R rusage[mem=8] -q gpuqueue -gpu "num=1" -o detection_tune_0.3.log python3 cell_detection_suppression.py tune 0.3')

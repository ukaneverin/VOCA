import sys
import glob
import numpy as np
import os

data_directory = '/lila/data/fuchs/xiec/TIL_detection/'

dis_thresh = float(sys.argv[1])

def relabel(phase):
    for f in glob.glob(data_directory + 'Crops_Anne_%s/*.npy' % phase):
        image_name = f.split('/')[-1]
        task_map = np.load(f)
        mask_name = image_name.split('.')[0]+'.png'
        mask = np.load(data_directory + 'Crops_Anne_%s/mask/%s.npy' % (phase, mask_name))
        for i in range(task_map.shape[0]):
            for j in range(task_map.shape[1]):
                if task_map[i,j,0] != 0:
                    x_trans = task_map[i,j,1]
                    y_trans = task_map[i,j,2]
                    if (x_trans**2+y_trans**2)**0.5 > dis_thresh:
                        task_map[i,j,:]=0
                    else:
                        cell_number = 0
                        for x in range(i-int(dis_thresh)-1, i+int(dis_thresh)+2):
                            for y in range(j-int(dis_thresh)-1, j+int(dis_thresh)+2):
                                if x>=0 and x<700 and y>=0 and y<700:
                                    dist = ((x-i)**2+(y-j)**2)**0.5
                                    if dist <= dis_thresh and mask[x, y] == 1:
                                        cell_number += 1
                        task_map[i,j,-1]=cell_number
        save_path = data_directory + 'Crops_Anne_%s/dis_thresh_%s/' % (phase, dis_thresh)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(save_path + image_name , task_map)

for phase in ['train', 'test', 'val']:
    relabel(phase)

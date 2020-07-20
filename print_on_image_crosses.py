import numpy as np
import glob
from scipy.misc import imread, imsave
import os
import sys
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

slide_id = sys.argv[1]
thresh = float(sys.argv[2])

data_directory = '/lila/data/fuchs/projects/lung/nuclei_vs_mask_test_images/images_to_print_on/'

plot_directory = data_directory+'printed/'

with open('/lila/data/fuchs/projects/lung/nuclei_vs_mask_test_images/coords/%s_lymphocytes_0.2.csv' % slide_id) as coord_file:
    coords = np.asarray([lines.strip().split(',') for lines in coord_file])[:,1:]
print(coords)

if not os.path.exists(plot_directory):
    os.makedirs(plot_directory)


original_image = imread(data_directory+'%s.png' % slide_id)
original_image = original_image[:,:,:3]

for c in coords:
    x = int(c[1])
    y = int(c[0])
    score = float(c[2])

    print_coords = []
    for xx in range(x-6,x+7):
        for yy in range(y-6,y+7):
            if abs(xx-x)<=1 and xx <= original_image.shape[0]-1 and yy <= original_image.shape[1]-1 and score >= thresh:
                original_image[xx, yy] = [0,255,0]
            elif abs(yy-y)<=1 and xx <= original_image.shape[0]-1 and yy <= original_image.shape[1]-1 and score >= thresh:
                original_image[xx, yy] = [0,255,0]


imsave(plot_directory+'/%s_printed_%s.png' % (slide_id, thresh), original_image)

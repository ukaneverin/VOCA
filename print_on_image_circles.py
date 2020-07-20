import numpy as np
import glob
from scipy.misc import imread, imsave
import os
import sys
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

data_directory = '/lila/data/fuchs/xiec/TIL_detection/'

probs_thresh_grid = np.linspace(0,255,11)[:-1]


for overlap_percentage in [9.0]:
    plot_directory = data_directory+'Crops_Anne_test_plot_%s' % overlap_percentage
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)
    #print(glob.glob(data_directory+'Crops_Anne_test_all_peak_suppression_%s/*.npy' % overlap_percentage))
    for image_file in glob.glob(data_directory+'Crops_Anne_test_all_peak_suppression_%s/*.npy' % overlap_percentage):
        mask =  np.load(image_file)
        image_name_full =  image_file.split('/')[-1]

        option = image_name_full.split('_')[-4].split('.')[0]
        config = image_name_full.split('_')[-3].split('.')[0]
        train_set = image_name_full.split('_')[-2].split('.')[0]
        test_set = image_name_full.split('_')[-1].split('.')[0]
        learned_hyper_file = open(data_directory+'hyper_learned/hyper_file_%s_%s_%s_%s_%s.txt' % (option, config, overlap_percentage, train_set, test_set), 'r')
        learned_hyper = [line.strip().split(',') for line in learned_hyper_file][-1]
        plot_thresh = float(learned_hyper[2])

        image_name = image_name_full.split('.png')[0]+'.png'
        original_image = imread(data_directory+'labeled_images/%s_labeled.png' % image_name)
        original_image = original_image[:,:,:3]

        thresh_image = np.copy(original_image)
        height, width, depth = thresh_image.shape
        #dpi = 80
        figsize = width/100 , height/100
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])
        # Hide spines, ticks, etc.
        ax.axis('off')
        ax.imshow(thresh_image, interpolation = 'nearest')
        for x in range(original_image.shape[0]):
            for y in range(original_image.shape[1]):
                intensity = mask[x,y]
                if intensity > plot_thresh:
                    try:
                        circ = Circle((y,x), 6, color = 'y', fill=False, linestyle = 'dashed')
                        ax.add_patch(circ)
                    except:
                        pass
        #ax.set(xlim=[0, width], ylim=[0, height], aspect=1)
        plt.savefig(plot_directory+'/%s_thresh_%s_plot.png' % (image_name_full, int(plot_thresh)))
        plt.close(fig)
        #imsave(plot_directory+'/%s_thresh_%s_plot.png' % (image_name_full, int(plot_thresh)),  thresh_image)

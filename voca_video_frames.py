from scipy.misc import imread,imsave,toimage
import numpy as np
from skimage.util import invert
import sys
save_path = '/lila/home/xiec/TIL_detection/frames/'
ori_I = imread('/lila/home/xiec/TIL_detection/393408_patch_1_crop3.png_labeled.png')
imsave(save_path + 'ori.png', ori_I[:127, :127, :])
sys.exit()
image_path = '/lila/data/fuchs/xiec/TIL_detection/Crops_Anne_test_mask_suppression_18.0/'
cls_map = np.load(image_path+'393408_patch_1_crop3.png_mask_cls_jointweighted_res.npy')
wt_map = np.load(image_path+'393408_patch_1_crop3.png_mask_wt_jointweighted_res.npy')
trans_x_map = np.load(image_path+'393408_patch_1_crop3.png_mask_transx_jointweighted_res.npy')
trans_y_map = np.load(image_path+'393408_patch_1_crop3.png_mask_transy_jointweighted_res.npy')
peak_map = np.load(image_path+'393408_patch_1_crop3.png_mask_jointweighted_res.npy')



'''create weighted image'''
weighted_cls_map = cls_map * wt_map

'''create frames of moving'''
frame_maps = weighted_cls_map
for i in range(18):
    frame_map = np.zeros(weighted_cls_map.shape)
    for x in range(weighted_cls_map.shape[0]):
        for y in range(cls_map.shape[1]):
            new_x = int(round(x + (i+1) * trans_x_map[x,y]/18))
            new_y = int(round(y + (i+1) * trans_y_map[x,y]/18))
            if new_x>=0 and new_x<700 and new_y>=0 and new_y<700:
                frame_map[new_x, new_y] += weighted_cls_map[x,y]
    frame_maps = np.dstack((frame_maps, frame_map))

#normalize frame_maps to uint8
# max_val = (frame_maps[:,:,-1].max())/10
# min_val = frame_maps[:,:,0].min()
# frame_maps -= min_val
# frame_maps *= (255.0/(max_val-min_val))

#save frames as images without rescaling
frame_maps = frame_maps[:127, :127, :]
imsave(save_path+"x.png", invert(trans_x_map[:127, :127]))
imsave(save_path+"y.png", invert(trans_y_map[:127, :127]))
imsave(save_path+"w.png", invert(wt_map[:127, :127]))
imsave(save_path+"peak.png", invert(peak_map[:127, :127]))
for i in range(frame_maps.shape[-1]):
    if i == 0 :
        frame = invert(frame_maps[:,:,i])
        max_val = frame.max()
        min_val = frame.min()
        frame -= min_val
        v = 150.0
        frame *= ( v/(max_val-min_val))
        frame += 255.0-v
        toimage(frame, cmin=0, cmax=255).save(save_path+"frame_%s.png" % i)
    else:
        frame = invert(frame_maps[:,:,i])
        imsave(save_path+"frame_%s.png" % i, frame)

cls_frame = invert(cls_map[:127, :127])
max_val = cls_frame.max()
min_val = cls_frame.min()
cls_frame -= min_val
v = 150.0
cls_frame *= ( v/(max_val-min_val))
cls_frame += 255.0-v
toimage(cls_frame, cmin=0, cmax=255).save(save_path+"cls_frame.png")
    #toimage(frame, cmin=0, cmax=255).save(save_path+"frame_%s.png" % i)

import numpy as np
import os
from models.res import VOCA_Res
import torch
from torchvision import transforms
import time
from Dataset_classes import TestImageDataset
import openslide
from SlideTileExtractor import extract_tissue
import math
import argparse
from utils import non_max_suppression
import glob
from pdb import set_trace

IO_test_ids = [x.split('/')[-1].split('.')[0] for x in glob.glob('/lila/data/fuchs/projects/lung/IO_test/*.svs')]
#IO_test_ids = [x.split('/')[-1].split('.')[0] for x in glob.glob('/lila/data/fuchs/xiec/results/IO_patch_homo/IO_geo/map2_rs896_k4997_nmst0/*')]
for id in IO_test_ids:
    slide_path = '/lila/data/fuchs/projects/lung/IO_test/%s.svs' % id
    #slide_path = '/lila/data/fuchs/projects/lung/impacted/%s.svs' % id
    slide = openslide.OpenSlide(slide_path)
    level, mult = extract_tissue.find_level(slide, 0.5, patchsize=762)
    if mult != 1.0:
        set_trace()
        image = slide.read_region((0, 0), level, (np.int(np.round(762 * mult)), np.int(np.round(762 * mult))))
        image = image.resize((762, 762))
        image = np.array(image)
        image = image[:, :, :3]
        image = image.astype(float)
    print(level, mult, id)
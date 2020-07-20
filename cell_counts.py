from os import system
import glob
from pdb import set_trace
slide_ids = [line.strip().split('/')[-1].split('_')[0] for line in glob.glob('/lila/data/fuchs/projects/lung/impacted/labels_AR/*.bmp')]

tumor_level_file = open('/lila/data/fuchs/projects/lung/cell_nuclei_coords/tumor_level_AR', 'w+')
for slide_id in slide_ids:
	train_log_file = '/lila/home/xiec/projects/VOCA/%s.log' % slide_id
	with open(train_log_file, 'r') as f:
		train_log = [lines.strip() for lines in f]
	num_tiles = train_log[-1].split('/')[-1].split(' ')[0]

	tumor_detection_file = '/lila/data/fuchs/projects/lung/cell_nuclei_coords/%s_tumorcells.csv' % slide_id
	with open(tumor_detection_file, 'r') as f:
		tumor_coords = [lines.strip() for lines in f]
	tumor_count = len(tumor_coords)
	tl = tumor_count / int(num_tiles)
	tumor_level_file.write('%s,%s\n' % (slide_id, tl))


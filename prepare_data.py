import os
import argparse
import glob
from tqdm import tqdm
import subprocess

parser = argparse.ArgumentParser(description='VOCA data preparation')
parser.add_argument('--dataset', default='lung_tt', type=str, help='lung_tt(which means til+tumor), lung_til, breast_til')
parser.add_argument('--output_path', default='/lila/data/fuchs/xiec/results/VOCA', type=str, help='working directory')

args = parser.parse_args()
image_folder = os.path.join(args.output_path, args.dataset, 'images/')
if not os.path.exists(image_folder):
	os.mkdir(image_folder)
image_name_list = []
for image_path in glob.glob(image_folder + '*.png'):
	image_name_list.append(image_path.strip().split('/')[-1])

for i in tqdm(range(0, len(image_name_list))):
	save_path = os.path.join(args.output_path, args.dataset, 'annotations/')
	if not os.path.exists(save_path):
		os.mkdir(save_path)
	file_path = save_path + image_name_list[i] + '.json'
	print(image_name_list[i])
	# subprocess.call(['wget -l 0 "https://slides-res.mskcc.org/slides/vanderbc@mskcc.org/40;crops_tumor_til;screenshot_398112.svs_x20717_y16975_z1.0082000000000002_deg0_image.png/getSVGLabels/nucleus" -O %s' % file_path],
	# 				shell=True)
	subprocess.call(['wget -l 0 "https://slides-res.mskcc.org/slides/vanderbc@mskcc.org/40;crops_tumor_til;%s/getSVGLabels/nucleus" -O %s' % (image_name_list[i].lower(), file_path)],
					shell=True)

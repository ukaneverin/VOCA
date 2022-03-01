from os import system
import glob
from pdb import set_trace
test_list = glob.glob('/lila/data/fuchs/projects/thyroid_thinprep/*.svs')
for slide in test_list:
	slide_id = slide.split('/')[-1].split('.')[0]
	system('bsub -W 30:00 -q gpuqueue -n 10 -gpu "num=1" -R "span[ptile=10] rusage[mem=8]" -o %s.log python thyroid.py '
		   '--subsample 0.5 --n_epochs 16 --cv 0 --stage ws_inference --slide_file %s' % (slide_id, slide))

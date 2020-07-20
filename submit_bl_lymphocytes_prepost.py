from os import system
import glob
from pdb import set_trace
slide_ids = [line.strip().split('/')[-1].split('_')[0] for line in glob.glob('/lila/data/fuchs/projects/lung/impacted/labels_AR/*.bmp')]

for slide_id in slide_ids:
    system('bsub -W 48:00 -q gpuqueue -n 10 -gpu "num=1" -R "span[ptile=10] rusage[mem=4]" -R V100 -sla ldSC -m "ld04 ld05 ld06 ld07" -o %s.log '
           'python cell_detection_normalized_ws.py --use_bmp --id %s' % (slide_id, slide_id))

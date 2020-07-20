from os import system
import glob
from pdb import set_trace

with open('/lila/home/xiec/projects/IO_patch_homo/full_variables_cases_collected.csv', 'r') as f:
    slide_ids = set(lines.strip().split(',')[0] for lines in f)

for slide_id in slide_ids:
    system('bsub -W 24:00 -q gpuqueue -n 10 -gpu "num=1" -R "span[ptile=10] rusage[mem=4]" -R V100 -sla ldSC -m "ld05 ld06 ld07" -o %s.log '
           'python cell_detection_normalized_ws.py --id %s' % (slide_id, slide_id))


from os import system
import os.path
import time
import sys
import glob


slide_ids = ['393379', '393380', '393382', '393396', '393408', '393436']


for slide_id in slide_ids:
    system("bsub -W 64:00 -n 1 -R rusage[mem=4] -gpu 'num=1' -R V100 -R span[ptile=1] -sla ldSC -q gpuqueue -m 'ld02 ld03 ld04 ld05 ld06 ld07' -o %s.log python3 cell_detection_normalized_ws.py jointweighted 12.0 res bl 9 %s breast" % (slide_id, slide_id))

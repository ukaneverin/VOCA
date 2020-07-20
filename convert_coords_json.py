import json
import sys
import os
import numpy as np

slide_id = sys.argv[1]
project_id = sys.argv[2]
streaming = False

with open('/lila/home/xiec/projects/IO_patch_homo/slide_viewer_list.csv', 'r') as f:
    #this allows matching between the slide path on viewer from slide id in project 90
    slide_path_dict = {lines.strip().split(';')[-1].split('.')[0]:lines.strip() for lines in f}

try:
    with open('/lila/data/fuchs/projects/lung/cell_nuclei_coords/%s_lymphocytes_0.2.csv' % slide_id, 'r') as coords_file:
        coord_info = [lines.strip().split(',')[:-1] for lines in coords_file]
except:
    with open('/lila/data/fuchs/projects/breast-infiltration/cell_nuclei_coords/%s_lymphocytes_0.2.csv' % slide_id, 'r') as coords_file:
        coord_info = [lines.strip().split(',')[:-1] for lines in coords_file]

json_object_list = []
if project_id == '90':
    for line in coord_info:
        coord_object = {"project_id":"%s" % project_id,"image_id":"%s" % slide_path_dict[slide_id],"label_type":"nucleus","x":"%s" % line[1],"y":"%s" % line[2],"class":"0","classname":"Tissue 1"}
        json_object_list.append(coord_object)
elif project_id == '44':
    for line in coord_info:
        coord_object = {"project_id":"%s" % project_id,"image_id":"%s.svs" % slide_id,"label_type":"nucleus","x":"%s" % line[1],"y":"%s" % line[2],"class":"0","classname":"Tissue 1"}
        json_object_list.append(coord_object)

if not os.path.exists('/lila/home/xiec/coords_upload/%s/' % slide_id):
    os.makedirs('/lila/home/xiec/coords_upload/%s/' % slide_id)
if streaming:
    """
    Streaming the json upload, since large files didn't work.
    """

    stream_number = int(np.ceil(len(json_object_list)/100))
    for i in range(stream_number):
        with open('/lila/home/xiec/coords_upload/%s/%s.json' % (slide_id, i), 'w+') as json_file:
            json_object_list_stream = json_object_list[i*100:(i+1)*100]
            json.dump(json_object_list_stream, json_file)
else:
    with open('/lila/home/xiec/coords_upload/%s/%s.json' % (slide_id, slide_id), 'w+') as json_file:
        json.dump(json_object_list, json_file)
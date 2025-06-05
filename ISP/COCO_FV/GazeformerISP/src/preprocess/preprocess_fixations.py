import scipy.io as sio
import numpy as np
import json
import os

# We use the same test set as paper
# "Visual Scanpath Prediction using IOR-ROI Recurrent Mixture Density Network" provided

with open('../../../PHAT/data/coco_freeview_fixations.json') as f:
    data = json.load(f)

new_data = []
for d in data:
    if d['split'] == 'val':
        d['split'] = 'validation'
    if d['task'] =='potted plant':
        d['task'] = 'potted_plant'
    if d['task'] =='stop sign':
        d['task'] = 'stop_sign'
    new_data.append(d)
    
save_json_file = 'src/data/FV/fixations.json'
with open(save_json_file, 'w') as f:
    json.dump(new_data, f, indent=2)
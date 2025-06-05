import scipy.io as sio
import numpy as np
import json
import os

# sort the list so it ranked by name and subject
def sort_list(scanpaths):
    scanpaths = sorted(data, key=lambda x: (x['task'], x['name'], x['subject'], x['split']))
    return scanpaths

with open('../../../PHAT/data/coco_search_fixations_512x320_on_target_allvalid.json') as f:
    data = json.load(f)
print('total number of scanpaths including TA and TP:', len(data))

for sp in data:
    sp['subject'] -= 1

ta, tp = [], []
tp = list(filter(lambda x: x['condition'] == 'present', data))
ta = list(filter(lambda x: x['condition'] == 'absent', data))

tp = sorted(tp, key=lambda x: (x['task'], x['name'], x['subject'], x['split']))
ta = sorted(ta, key=lambda x: (x['task'], x['name'], x['subject'], x['split']))

with open('src/data/TPTA/TA_fixations.json', 'w') as f:
    json.dump(ta, f, indent=4)
with open('src/data/TPTA/TP_fixations.json', 'w') as f:
    json.dump(tp, f, indent=4)



import numpy as np
import json
import os

# take 280 instances from training set as train, and 280 as pseudo-valid
def get_original_training_file():
    with open('src/data/OSIE/fixations.json') as f:
        data = json.load(f)
    train_count = 0
    new_sp = []
    for sp in data:
        if sp['split'] == 'train':
            if train_count == 280*15:
                sp['split'] = 'pseudo_val'
                new_sp.append(sp)
            else:
                new_sp.append(sp)
                train_count += 1
        elif sp['split'] == 'validation':
            sp['split'] = 'val'
            new_sp.append(sp)
        else:
            new_sp.append(sp)
    with open('src/data/OSIE/fixations_pseudo_sp_original.json', 'w') as f:
        json.dump(new_sp, f, indent=4) 



if __name__ == '__main__':
    get_original_training_file()

    log_dir = 'osie-useremb-ex-subject-01234-490-train'
    # replace_gt_labels(log_dir)


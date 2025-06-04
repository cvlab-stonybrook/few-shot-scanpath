import json

def get_prediction_list(args, predict_results, img_name, pred, subject_idx):
    X = [int(item[0]) for item in pred]
    Y = [int(item[1]) for item in pred]
    T = [int(round(item[2] * 1000, 3)) for item in pred]
    if args.ex_subject[0] != -1:
        predict_results.append(
            {'name':img_name, 'subject': recover_subject_ids(args.subject_num, args.ex_subject, subject_idx),'X': X, 'Y': Y, 'T': T})
    elif args.fewshot_subject[0] != -1:
        predict_results.append({'name':img_name, 'subject': args.fewshot_subject[subject_idx],'X': X, 'Y': Y, 'T': T}) 
    else:
        predict_results.append({'name':img_name, 'subject': subject_idx,'X': X, 'Y': Y, 'T': T})    
    return predict_results 


# map training subject_ids to original_subject_ids because ex_subjects list will
# change original_subject_ids to range (0, num_subjects)
def recover_subject_ids(subject_num, ex_subject, subject_idx):
    all_subject_ids = list(range(subject_num+len(ex_subject)))
    left_subject_ids = [id for id in all_subject_ids if id not in ex_subject]
    mapping = {new_id: original_id for new_id, original_id in enumerate(left_subject_ids)}
    return mapping[subject_idx]

# replace gt scanpath to pseudo scanpath
def replace_gt_labels(log_info_folder, predict_results):
    with open('src/data/OSIE/fixations_pseudo_sp_original.json') as f:
        data = json.load(f)
    new_sp = []
    for gt_sp in data:
        for p_sp in predict_results:
            if gt_sp['name'] == p_sp['name'] and gt_sp['subject'] == p_sp['subject']:
                gt_sp['X'] = p_sp['X']
                gt_sp['Y'] = p_sp['Y']
                gt_sp['T'] = p_sp['T']
                gt_sp['split'] = 'train'
                gt_sp['length'] = len(p_sp['X'])
                break
        new_sp.append(gt_sp)
    with open(f'{log_info_folder}/fixations_pseudo_sp.json', 'w') as f:
        json.dump(new_sp, f, indent=4)
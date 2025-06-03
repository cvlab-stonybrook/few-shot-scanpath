
from torchvision import transforms
import numpy as np
from .utils import compute_search_cdf, preprocess_fixations, filter_scanpath, select_fewshot_subject
from .utils import cutFixOnTarget
from .data import  Siamese_Triplet_Gaze


def process_data(target_trajs,
                 dataset_root,
                 target_annos,
                 hparams,
                 device):

    print("using", hparams.Train.repr, 'dataset:', hparams.Data.name, 'TAP:',
          hparams.Data.TAP)

    # Rescale fixations and images if necessary
    if hparams.Data.name == 'OSIE':
        ori_h, ori_w = 600, 800
        rescale_flag = hparams.Data.im_h != ori_h
    elif hparams.Data.name == 'COCO-Search18' or hparams.Data.name == 'COCO-Freeview':
        ori_h, ori_w = 320, 512
        rescale_flag = hparams.Data.im_h != ori_h
    elif hparams.Data.name == 'MIT1003':
        rescale_flag = False # Use rescaled scanpaths
    elif hparams.Data.name == 'CAT2000':
        ori_h, ori_w = 1080, 1920
        rescale_flag = hparams.Data.im_h != ori_h
    else:
        print(f"dataset {hparams.Data.name} not supported")
        raise NotImplementedError
    if rescale_flag:
        print(
            f"Rescaling image and fixation to {hparams.Data.im_h}x{hparams.Data.im_w}"
        )
        size = (hparams.Data.im_h, hparams.Data.im_w)
        ratio_h = hparams.Data.im_h / ori_h
        ratio_w = hparams.Data.im_w / ori_w
        for traj in target_trajs:
            traj['X'] = np.array(traj['X']) * ratio_w
            traj['Y'] = np.array(traj['Y']) * ratio_h
            traj['rescaled'] = True


    size = (hparams.Data.im_h, hparams.Data.im_w)
    transform_train = transforms.Compose([
        transforms.Resize(size=size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        transforms.Resize(size=size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    ])

    valid_target_trajs = list(
        filter(lambda x: x['split'] == 'test', target_trajs))
    

    is_coco_dataset = hparams.Data.name == 'COCO-Search18' or hparams.Data.name == 'COCO-Freeview'


    target_init_fixs = {}
    for traj in target_trajs:
        key = traj['task'] + '*' + traj['name'] + '*' + traj['condition']
        if is_coco_dataset:
            # Force center initialization for COCO-Search18
            target_init_fixs[key] = (0.5, 0.5)  
        else:
            target_init_fixs[key] = (traj['X'][0] / hparams.Data.im_w,
                                     traj['Y'][0] / hparams.Data.im_h)
    cat_names = list(np.unique([x['task'] for x in target_trajs]))
    catIds = dict(zip(cat_names, list(range(len(cat_names)))))

    human_mean_cdf = None
    # training fixation data
    train_target_trajs = list(
        filter(lambda x: x['split'] == 'train', target_trajs))

    # fewshot_subject indicating subject ids for unseen subjects
    if hparams.Data.fewshot_subject[0] != -1:
        train_target_trajs = select_fewshot_subject(hparams.Train.log_dir,
            train_target_trajs, hparams.Data.fewshot_subject, 
            hparams.Data.num_fewshot, hparams.Data.num_subjects, hparams.Data.random_support, 'train')
        
    # print statistics
    traj_lens = list(map(lambda x: x['length'], train_target_trajs))
    avg_traj_len, std_traj_len = np.mean(traj_lens), np.std(traj_lens)
    print('average train scanpath length : {:.3f} (+/-{:.3f})'.format(
        avg_traj_len, std_traj_len))
    print('num of train trajs = {}'.format(len(train_target_trajs)))

    train_task_img_pair = np.unique([
        traj['task'] + '*' + traj['name'] + '*' + traj['condition']
        for traj in train_target_trajs
    ])
    train_fix_labels = preprocess_fixations(
        train_target_trajs,
        hparams.Data.patch_size,
        hparams.Data.patch_num,
        hparams.Data.im_h,
        hparams.Data.im_w,
        truncate_num=hparams.Data.max_traj_length,
        has_stop=hparams.Data.has_stop,
        sample_scanpath=False,
        min_traj_length_percentage=0,
        discretize_fix=hparams.Data.discretize_fix,
        remove_return_fixations=hparams.Data.remove_return_fixations,
        is_coco_dataset=is_coco_dataset,
    )

    # validation fixation data
    valid_target_trajs = list(
        filter(lambda x: x['split'] == 'test', target_trajs))
    
    
    # print statistics
    traj_lens = list(map(lambda x: x['length'], valid_target_trajs))
    avg_traj_len, std_traj_len = np.mean(traj_lens), np.std(traj_lens)
    print('average valid scanpath length : {:.3f} (+/-{:.3f})'.format(
        avg_traj_len, std_traj_len))
    print('num of valid trajs = {}'.format(len(valid_target_trajs)))

    
    if hparams.Data.TAP in ['TP', 'TAP']:
        tp_trajs = list(
        filter(
            lambda x: x['condition'] == 'present' and x['split'] == 'test',
            target_trajs))
        human_mean_cdf, _ = compute_search_cdf(
            tp_trajs, target_annos, hparams.Data.max_traj_length)
        print('target fixation prob (valid).:', human_mean_cdf)

    valid_fix_labels = preprocess_fixations(
        valid_target_trajs,
        hparams.Data.patch_size,
        hparams.Data.patch_num,
        hparams.Data.im_h,
        hparams.Data.im_w,
        truncate_num=hparams.Data.max_traj_length,
        has_stop=hparams.Data.has_stop,
        sample_scanpath=False,
        min_traj_length_percentage=0,
        discretize_fix=hparams.Data.discretize_fix,
        remove_return_fixations=hparams.Data.remove_return_fixations,
        is_coco_dataset=is_coco_dataset,
    )


    # original HAT code is generate training examples for each fixation, 
    # in SE-Net, we only select the whole scanpath
    train_fix_labels = filter_scanpath(train_fix_labels)
    valid_fix_labels = filter_scanpath(valid_fix_labels)
    
    
    train_HG_dataset = Siamese_Triplet_Gaze(dataset_root,
                                            train_fix_labels,
                                            target_annos,
                                            hparams.Data,
                                            transform_train,
                                            catIds,
                                            device,
                                            blur_action=True)
    valid_HG_dataset = Siamese_Triplet_Gaze(dataset_root,
                                            valid_fix_labels,
                                            target_annos,
                                            hparams.Data,
                                            transform_test,
                                            catIds,
                                            device,
                                            blur_action=True)

    if hparams.Data.TAP == ['TP', 'TAP']:
        cutFixOnTarget(target_trajs, target_annos)
    print("num of training and eval fixations = {}, {}".format(
        len(train_HG_dataset), len(valid_HG_dataset)))

    return {
        'catIds': catIds,
        'gaze_train': train_HG_dataset,
        'gaze_valid': valid_HG_dataset,
        'bbox_annos': target_annos,
        'valid_scanpaths': valid_target_trajs,
        'human_cdf': human_mean_cdf,
    }

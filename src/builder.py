import sys

sys.path.append('../common')

from common.dataset import process_data
from .models import UserEmbeddingNet
from common.utils import adjust_subjects
import json
from os.path import join

import numpy as np
import torch
from torch.utils.data import DataLoader


def build(hparams, dataset_root, device, is_eval=False, split=1):
    dataset_name = hparams.Data.name

    bbox_annos = np.load(
        join(dataset_root, 'bbox_annos.npy'),
        allow_pickle=True).item() if dataset_name == 'COCO-Search18' else {}

    with open(join(dataset_root, hparams.Data.fix_path), 'r') as json_file:
        human_scanpaths = json.load(json_file)
    if dataset_name == 'COCO-Search18':
        n_tasks = 18
    else:
        n_tasks = 1

    # hparams.Data.subject indicating which subjects are unseen subjects
    if hparams.Data.subject[0] != -1:
        print(f"skip subject {hparams.Data.subject} data!")
        human_scanpaths = adjust_subjects(human_scanpaths, hparams.Data.subject)

    # Filtering training data
    if hparams.Data.TAP == 'TP':
        human_scanpaths = list(
            filter(lambda x: x['condition'] == 'present', human_scanpaths))
        human_scanpaths = list(
            filter(lambda x: x['fixOnTarget'], human_scanpaths))
    elif hparams.Data.TAP == 'FV':
        human_scanpaths = list(
            filter(lambda x: x['condition'] == 'freeview', human_scanpaths))
        n_tasks = 1

    # process fixation data
    dataset = process_data(
        human_scanpaths,
        dataset_root,
        bbox_annos,
        hparams,
        device)

    batch_size = hparams.Train.batch_size
    n_workers = hparams.Train.n_workers

    tag = False if is_eval else True
    bs = batch_size // 2 if is_eval else batch_size
    train_HG_loader = DataLoader(dataset['gaze_train'],
                                 batch_size=bs,
                                 shuffle=tag,
                                 num_workers=n_workers,
                                 drop_last=True,
                                 pin_memory=True)
    print('num of training batches =', len(train_HG_loader))

    
    valid_HG_loader = DataLoader(dataset['gaze_valid'],
                                    batch_size=batch_size//2,
                                    shuffle=False,
                                    num_workers=n_workers,
                                    drop_last=False,
                                    pin_memory=True)


    # Create model
    emb_size = hparams.Model.embedding_dim
    n_heads = hparams.Model.n_heads
    hidden_size = hparams.Model.hidden_dim


    model = UserEmbeddingNet(
            hparams.Data,
            num_decoder_layers=hparams.Model.n_dec_layers,
            hidden_dim=emb_size,
            nhead=n_heads,
            ntask=n_tasks,
            num_output_layers=hparams.Model.num_output_layers,
            train_encoder=hparams.Train.train_backbone,
            train_pixel_decoder=hparams.Train.train_pixel_decoder,
            dropout=hparams.Train.dropout,
            dim_feedforward=hidden_size,
            num_encoder_layers=hparams.Model.n_enc_layers)
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=hparams.Train.adam_lr,
                                  betas=hparams.Train.adam_betas)

    # Load weights from checkpoint when available
    if len(hparams.Model.checkpoint) > 0:
        print(f"loading weights from {hparams.Model.checkpoint} in {hparams.Train.transfer_learn} setting.")
        ckp = torch.load(join(hparams.Train.log_dir, hparams.Model.checkpoint), map_location=device)

        model.load_state_dict(ckp['model'], strict=False)
        # optimizer.load_state_dict(ckp['optimizer'])
        global_step = ckp['step']
    else:
        global_step = 0

    if hparams.Train.parallel:
        model = torch.nn.DataParallel(model)

    bbox_annos = dataset['bbox_annos']
    human_cdf = dataset['human_cdf']


    is_lasts = [x[5] for x in dataset['gaze_train'].fix_labels]
    term_pos_weight = len(is_lasts) / np.sum(is_lasts) - 1
    print("termination pos weight: {:.3f}".format(term_pos_weight))

    return (model, optimizer, train_HG_loader, valid_HG_loader, term_pos_weight, global_step)

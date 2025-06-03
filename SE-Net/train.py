"""
Two-pathway (Ventral and Dorsal) Transformer Training Script.
This script is a simplified version of the training scripts in 
https://github.com/cvlab-stonybrook/Scanpath_Prediction
"""
import argparse
import os
import random

import numpy as np

import datetime
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from src.builder import build
from common.config import JsonConfig
from common.utils import (
    transform_fixations, )
from src.eval_user import evaluate_user_siamese

SEED = 0
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)


def parse_args():
    """Parse args."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--hparams',
                        type=str,
                        help='hyper parameters config file path')
    parser.add_argument('--dataset-root', type=str, help='dataset root path')
    parser.add_argument('--eval-only',
                        action='store_true',
                        help='perform evaluation only')
    parser.add_argument(
        '--split',
        type=int,
        default=1,
        help='dataset split for MIT1003/CAT2000 only (default=1)')
    return parser.parse_args()


def log_dict(writer, scalars, step, prefix):
    for k, v in scalars.items():
        writer.add_scalar(prefix + "/" + k, v, step)


def compute_output(instance, model, pa):
    img = instance['true_state'].to(device)

    duration = instance['duration'].to(device)
    task_emb= instance['task_emb'].to(device)
    inp_seq, inp_seq_high = transform_fixations(instance['normalized_fixations'],
                                                    instance['is_padding'],
                                                    hparams.Data,
                                                    False,
                                                    return_highres=True)
    inp_seq = inp_seq.to(device)
    inp_padding_mask = (inp_seq == pa.pad_idx)
    logits = model(img, inp_seq, inp_padding_mask, inp_seq_high.to(device), duration, task_emb)
    return logits

def compute_loss(model, batch, losses, loss_funcs, pa):
    all_user_emb = {}
    all_pred = {}
    all_bbox = {}
    for key, instance in batch.items():
        logit = compute_output(instance, model, pa)
        all_user_emb[key] = logit['user_emb']
        all_pred[key] = logit['pred_subject_id']
        if "bbox_pred" in losses:
            all_bbox[key] = logit['bbox_pred']

    loss_dict = {}


    if "subject_id_pred" in losses:
        total_subject_id_loss = []
        for key, _ in batch.items():
            gt_subject_id = batch[key]['subject_id'].to(device).long()
            pred_subject_id = all_pred[key].squeeze()
            total_subject_id_loss.append(loss_funcs['subject_id_pred'](pred_subject_id, gt_subject_id))
        loss_dict['subject_id_pred'] = sum(total_subject_id_loss)
    
    if "triplet_loss" in losses:
        loss_dict['triplet_loss'] = loss_funcs['triplet_loss'](
            all_user_emb['anchor'], all_user_emb['positive'], all_user_emb['negative'])

    if "bbox_pred" in losses:
        total_bbox_loss = []
        for key, _ in batch.items():
            gt_bbox = batch[key]['bbox'].to(device)
            pred_bbox = all_bbox[key]
            total_bbox_loss.append(loss_funcs['subject_id_pred'](pred_subject_id, gt_subject_id))
        loss_dict['bbox_pred'] = sum(total_bbox_loss)


    return loss_dict


def train_iter(model, optimizer, batch, losses, loss_weights, loss_funcs, pa):
    assert len(losses) > 0, "no loss func assigned!"
    model.train()
    optimizer.zero_grad()

    loss_dict = compute_loss(model, batch, losses, loss_funcs, pa)
    loss = 0
    for k, v in loss_dict.items():
        loss += v * loss_weights[k]
    loss.backward()
    optimizer.step()

    for k in loss_dict:
        loss_dict[k] = loss_dict[k].item()

    return loss_dict


def run_evaluation():
    # gazeloader = train_gaze_loader
    gazeloader = val_gaze_loader
    if args.eval_only:   # to generate unseen subject embeddings
        gazeloader = train_gaze_loader
    # use the following line if you want to evaluate the subject classification accuracy
    gazeloader = val_gaze_loader
    evaluate_user_siamese(
            global_step,
            model,
            device,
            gazeloader,
            hparams,
            log_dir=log_dir)


if __name__ == '__main__':
    args = parse_args()
    hparams = JsonConfig(args.hparams)
    dir = os.path.dirname(args.hparams)

    dataset_root = args.dataset_root
    if dataset_root[-1] == '/':
        dataset_root = dataset_root[:-1]
    device = torch.device('cuda')
    # if hparams.Data.name in ['MIT1003', 'CAT2000']:
    #     hparams.Train.log_dir += f'_split{args.split}'

    model, optimizer, train_gaze_loader, val_gaze_loader, term_pos_weight, global_step = build(
            hparams, dataset_root, device, args.eval_only, args.split)

    log_dir = hparams.Train.log_dir
    if args.eval_only:
        run_evaluation()
    else:
        writer = SummaryWriter(log_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        print("Log dir:", log_dir)
        log_folder_runs = "./runs/{}".format(log_dir.split('/')[-1])
        if not os.path.exists(log_folder_runs):
            os.system(f"mkdir -p {log_folder_runs}")

        # Write configuration file to the log dir
        hparams.dump(log_dir, 'config.json')

        print_every = hparams.Train.print_every
        max_iters = hparams.Train.max_iters
        save_every = hparams.Train.checkpoint_every
        eval_every = hparams.Train.evaluate_every
        pad_idx = hparams.Data.pad_idx
        loss_funcs = {
            "subject_id_pred":
            torch.nn.CrossEntropyLoss(),
            "triplet_loss":
            torch.nn.TripletMarginLoss(margin=5, p=2),
            "pairwise_loss":
            torch.nn.TripletMarginLoss(margin=5, p=2),
            "bbox_pred":
            torch.nn.L1Loss()
        }

        loss_weights = {
            "subject_id_pred": hparams.Train.subject_id_pred_weight,
            "triplet_loss": hparams.Train.triplet_loss,
            "bbox_pred": 1.0,
        }
        losses = hparams.Train.losses
        loss_dict_avg = dict(zip(losses, [0] * len(losses)))
        print("loss weights:", loss_weights)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=hparams.Train.lr_steps, gamma=0.1)

        s_epoch = int(global_step / len(train_gaze_loader))

        last_time = datetime.datetime.now()
        for i_epoch in range(s_epoch, int(1e5)):
            scheduler.step()
            for i_batch, batch in enumerate(train_gaze_loader):
                loss_dict = train_iter(model, optimizer, batch, losses,
                                       loss_weights, loss_funcs, hparams.Data)
                for k in loss_dict:
                    loss_dict_avg[k] += loss_dict[k]

                if global_step % print_every == print_every - 1:
                    for k in loss_dict_avg:
                        loss_dict_avg[k] /= print_every

                    time = datetime.datetime.now()
                    eta = str((time - last_time) / print_every *
                              (max_iters - global_step))
                    last_time = time
                    time = str(time)
                    log_msg = "[{}], eta: {}, iter: {}, progress: {:.2f}%, epoch: {}, total loss: {:.3f}".format(
                        time[time.rfind(' ') + 1:time.rfind('.')],
                        eta[:eta.rfind('.')],
                        global_step,
                        (global_step / max_iters) * 100,
                        i_epoch,
                        np.sum(list(loss_dict_avg.values())),
                    )

                    for k, v in loss_dict_avg.items():
                        log_msg += " {}_loss: {:.3f}".format(k, v)

                    print(log_msg)
                    log_dict(writer, loss_dict_avg, global_step, 'train')
                    writer.add_scalar('train/lr',
                                      optimizer.param_groups[0]["lr"],
                                      global_step)
                    for k in loss_dict_avg:
                        loss_dict_avg[k] = 0
                    
                # save checkpoint
                if global_step % save_every == save_every - 1 and global_step < 30000:
                    save_path = os.path.join(log_dir, f"ckp_{global_step}.pt")
                    if isinstance(model, torch.nn.DataParallel):
                        model_weights = model.module.state_dict()
                    else:
                        model_weights = model.state_dict()
                    torch.save(
                        {
                            'model': model_weights,
                            'optimizer': optimizer.state_dict(),
                            'step': global_step + 1,
                        },
                        save_path,
                    )
                    print(f"Saved checkpoint to {save_path}.")
                # Evaluate
                if global_step % eval_every == eval_every - 1:
                    run_evaluation()

                    writer.add_scalar('train/epoch',
                                      global_step / len(train_gaze_loader),
                                      global_step)
                    os.system(f"cp {log_dir}/events* {log_folder_runs}")

                
                global_step += 1
                if global_step >= max_iters:
                    print("Exit program!")
                    break
            else:
                continue
            break  # Break outer loop

        # Copy to log file to ./runs
        os.system(f"cp {log_dir}/events* {log_folder_runs}")
import sys
sys.path.append('../common')
import time
import torch, json
import pickle
from tqdm import tqdm
from os.path import join
from common.utils import (
    transform_fixations, )


def evaluate_user_siamese(
             global_step,
             model,
             device,
             gazeloader,
             hparams,
             log_dir=None):

    
    start_time = time.time()
    model.eval()
    num_subjects = hparams.Data.num_subjects
    
    correct_predictions_per_class_top1 ={class_idx: 0 for class_idx in range(num_subjects)}
    correct_predictions_per_class_top3 ={class_idx: 0 for class_idx in range(num_subjects)}
    correct_predictions_per_class_top5 ={class_idx: 0 for class_idx in range(num_subjects)}
    total_samples_per_class = {class_idx: 0 for class_idx in range(num_subjects)}
    class_embeddings = torch.zeros(num_subjects, hparams.Model.embedding_dim)
    class_counts = torch.zeros(num_subjects)
    user_emb_dict = {i: [] for i in range(num_subjects)}
    total_pred = []
    total_gt = []
    all_attn_weights = []

    for i_batch, batch in enumerate(tqdm(gazeloader)):
        batch = batch['anchor']
        img = batch['true_state'].to(device)
        duration = batch['duration'].to(device)
        task_emb = batch['task_emb'].to(device)
        inp_seq, inp_seq_high = transform_fixations(batch['normalized_fixations'],
                                                    batch['is_padding'],
                                                    hparams.Data,
                                                    False,
                                                    return_highres=True)
        inp_seq = inp_seq.to(device)
        inp_padding_mask = (inp_seq == hparams.Data.pad_idx)
        logits = model(img, inp_seq, inp_padding_mask, inp_seq_high.to(device), duration, task_emb)
        bs = img.size(0)
        gt_subject_id = batch['subject_id']
        pred_subject_id = logits['pred_subject_id'].cpu()

        # # compute classification accuracy
        _, pred_labels_top1 = torch.max(pred_subject_id, 1)
        _, pred_labels_top3 = torch.topk(pred_subject_id, k=3, dim=1)
        _, pred_labels_top5 = torch.topk(pred_subject_id, k=5, dim=1)

        for class_idx in range(hparams.Data.num_subjects):
            correct_predictions_per_class_top1[class_idx] += ((pred_labels_top1 == class_idx) & (gt_subject_id == class_idx)).sum().item()
            correct_predictions_per_class_top3[class_idx] += ((pred_labels_top3 == class_idx).any(dim=1) & (gt_subject_id == class_idx)).sum().item()
            correct_predictions_per_class_top5[class_idx] += ((pred_labels_top5 == class_idx).any(dim=1) & (gt_subject_id == class_idx)).sum().item()
            total_samples_per_class[class_idx] += (gt_subject_id == class_idx).sum().item()

        # get user_embedding for each user
        save_user_emb = True
        if save_user_emb:
            user_emb = logits['user_emb'].detach().cpu()   
            class_embeddings.index_add_(0, gt_subject_id.cpu(), user_emb)
            class_counts.index_add_(0, gt_subject_id.cpu(), torch.ones_like(gt_subject_id, dtype=torch.float))
        
        # set true if you want to visualize attention weights of transformer layers
        save_user_emb_dict = False
        if save_user_emb_dict:
            user_emb = logits['user_emb'].detach().cpu().numpy()
            gt_subject_id = gt_subject_id.detach().cpu().numpy()
            # print(gt_subject_id)
            for user_emb_idx, sub_id in enumerate(gt_subject_id):
                user_emb_dict[sub_id].append(user_emb[user_emb_idx])

        
        save_attn_weights = False
        if save_attn_weights:
            for attn_idx in range(bs):
                win = (gt_subject_id[attn_idx] == pred_labels_top1[attn_idx])
                all_attn_weights.append({'name': batch['img_name'][attn_idx], 
                                        'subject': batch['subject_id'][attn_idx].cpu(),
                                        'task': batch['task_name'][attn_idx],
                                        'fixs': batch['original_fixs'][attn_idx].cpu(),
                                        'duration': batch['duration'][attn_idx].cpu(),
                                        # 'img_attn': logits['task_image_weights'][attn_idx].detach().cpu(),
                                        'fix_attn': logits['fix_attn_weights'][attn_idx].detach().cpu(),
                                        'win': win})

        
        total_pred.extend(pred_labels_top1.tolist())
        total_gt.extend(gt_subject_id.tolist())
    
    if hparams.Data.fewshot_subject[0] != -1 and hparams.Data.num_fewshot == 1:
        torch.save(class_embeddings, f'{log_dir}/fewshot_user_embedding_{hparams.Data.num_fewshot}.pt')
        return 0
    
    avg_top1_accuracy = compute_average_accuracy(correct_predictions_per_class_top1, total_samples_per_class, num_subjects)
    avg_top3_accuracy = compute_average_accuracy(correct_predictions_per_class_top3, total_samples_per_class, num_subjects)
    avg_top5_accuracy = compute_average_accuracy(correct_predictions_per_class_top5, total_samples_per_class, num_subjects)

    # get user_emb and save
    nonzero_mask = class_counts > 0
    class_embeddings[nonzero_mask] /= class_counts[nonzero_mask].unsqueeze(1)
    if hparams.Data.fewshot_subject[0] != -1:
        print(f'save fewshot embedding to {log_dir}/fewshot_user_embedding_{hparams.Data.num_fewshot}.pt')
        torch.save(class_embeddings, f'{log_dir}/fewshot_user_embedding_{hparams.Data.num_fewshot}.pt')
    else:
        torch.save(class_embeddings, f'{log_dir}/train_user_embedding_no_vsencoder.pt')


    if hparams.Data.fewshot_subject[0] != -1:
        with open(f'{log_dir}/fewshot_user_emb_dict.pkl', 'wb') as pickle_file:
            pickle.dump(user_emb_dict, pickle_file)
    else:
        with open(f'{log_dir}/train_user_emb_dict.pkl', 'wb') as pickle_file:
            pickle.dump(user_emb_dict, pickle_file)

    accuracy_dict = {'top1_accuracy': avg_top1_accuracy,
                     'top3_accuracy': avg_top3_accuracy,
                     'top5_accuracy': avg_top5_accuracy}

    print(accuracy_dict)

    # use knn to compute seen accuracy
    with open(join(log_dir, f'metrics_user_{global_step}.json'), 'w') as f:
        json.dump(accuracy_dict, f, indent=4)

    if save_attn_weights:
        with open(f'{log_dir}/train_attn_weights.pkl', 'wb') as pickle_file:
            pickle.dump(all_attn_weights, pickle_file)

    end_time = time.time()
    print('time for evaluation: ', end_time - start_time)

    
def compute_average_accuracy(correct_predictions, total_samples, num_classes):
    per_class_accuracy = [
        (correct_predictions[class_idx] / total_samples[class_idx] * 100) if total_samples[class_idx] > 0 else 0
        for class_idx in range(num_classes)
    ]
    # Return the average accuracy across all classes
    return sum(per_class_accuracy) / num_classes

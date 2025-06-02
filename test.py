import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import numpy as np
import scipy.stats

import time
import os
import argparse
from os.path import join
from tqdm import tqdm
import datetime
import json
import sys

from dataset.dataset import OSIE_evaluation
from utils.evaluation import human_evaluation, evaluation, evaluation_by_subject, human_evaluation_by_subject, \
    comprehensive_evaluation_by_subject
from utils.logger import Logger
from models.sampling import Sampling

# from models.gazeformer_baseline import gazeformer
# from models.models import Transformer

# encoder_decoder
from models.subject_encoder_decoder_models import Transformer
from models.gazeformer_subject_encoder_decoder import gazeformer

# encoder
# from models.subject_encoder_models import Transformer
# from models.gazeformer_subject_encoder import gazeformer

# decoder
# from models.subject_decoder_models import Transformer
# from models.gazeformer_subject_decoder import gazeformer

parser = argparse.ArgumentParser(description="Scanpath prediction for images")
parser.add_argument("--mode", type=str, default="test", help="Selecting running mode (default: test)")
parser.add_argument("--img_dir", type=str, default="/media/xianyu/CVPR_2024/OSIE/images", help="Directory to the image data (stimuli)")
parser.add_argument("--feat_dir", type=str, default="/media/xianyu/CVPR_2024/OSIE/image_features",
                    help="Directory to the image feature data (stimuli)")
parser.add_argument("--fix_dir", type=str, default="/media/xianyu/CVPR_2024/OSIE/processed", help="Directory to the raw fixation file")
parser.add_argument("--width", type=int, default=512, help="Width of input data")
parser.add_argument("--height", type=int, default=384, help="Height of input data")
parser.add_argument("--origin_width", type=int, default=800, help="original Width of input data")
parser.add_argument("--origin_height", type=int, default=600, help="original Height of input data")
parser.add_argument('--im_h', default=24, type=int, help="Height of feature map input to encoder")
parser.add_argument('--im_w', default=32, type=int, help="Width of feature map input to encoder")
parser.add_argument("--batch", type=int, default=1, help="Batch size")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--gpu_ids", type=list, default=[0,1,2], help="Used gpu ids")
parser.add_argument("--evaluation_dir", type=str, default="../runs/OSIE_baseline_40epoch_rl_decay0.1_25epoch_seed1_ft_subj_embedding_all_params_560img_lr1e-4",
                    help="Resume from a specific directory")
parser.add_argument("--eval_repeat_num", type=int, default=1, help="Repeat number for evaluation")
parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the generated scanpath")
parser.add_argument("--max_length", type=int, default=16, help="Maximum length of the generated scanpath")

parser.add_argument('--patch_size', default=16, type=int,
                        help="Patch size of feature map input with respect to fixation image dimensions (320X512)")
parser.add_argument('--num_encoder', default=6, type=int, help="Number of transformer encoder layers")
parser.add_argument('--num_decoder', default=6, type=int, help="Number of transformer decoder layers")
parser.add_argument('--hidden_dim', default=512, type=int, help="Hidden dimensionality of transformer layers")
parser.add_argument('--nhead', default=8, type=int, help="Number of heads for transformer attention layers")
parser.add_argument('--img_hidden_dim', default=2048, type=int, help="Channel size of initial ResNet feature map")
parser.add_argument('--lm_hidden_dim', default=768, type=int,
                    help="Dimensionality of target embeddings from language model")
parser.add_argument('--encoder_dropout', default=0.1, type=float, help="Encoder dropout rate")
parser.add_argument('--decoder_dropout', default=0.2, type=float, help="Decoder and fusion step dropout rate")
parser.add_argument('--cls_dropout', default=0.4, type=float, help="Final scanpath prediction dropout rate")

parser.add_argument('--cuda', default=0, type=int, help="CUDA core to load models and data")
parser.add_argument("--subject_num", type=int, default=15, help="The number of the subject in OSIE")
parser.add_argument("--subject_feature_dim", type=int, default=128, help="The dim of the subject feature")
parser.add_argument("--action_map_num", type=int, default=4, help="The dim of action map")
args = parser.parse_args()

# For reproducibility - refer https://pytorch.org/docs/stable/notes/randomness.html
# These five lines control all the major sources of randomness.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

transform = transforms.Compose([
                                transforms.Resize((args.height * 2, args.width * 2)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])

def main():

    # load logger
    log_dir = args.evaluation_dir
    checkpoints_dir = os.path.join(log_dir, "checkpoints")
    log_info_folder = os.path.join(log_dir, "log")
    log_file = os.path.join(log_info_folder, "log_test.txt")
    predicts_file = os.path.join(log_dir, "test_predicts.json")
    logger = Logger(log_file)

    logger.info("The args corresponding to testing process are: ")
    for (key, value) in vars(args).items():
        logger.info("{key:20}: {value:}".format(key=key, value=value))

    test_dataset = OSIE_evaluation(args.img_dir, args.feat_dir, args.fix_dir, action_map=(args.im_h, args.im_w),
                         resize=(args.height, args.width), type="test", transform=transform)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=4,
        collate_fn=test_dataset.collate_func
    )

    device = torch.device('cuda:{}'.format(args.cuda))

    # encoder + decoder
    transformer = Transformer(num_encoder_layers=args.num_encoder, nhead=args.nhead,
                              subject_feature_dim=args.subject_feature_dim, d_model=args.hidden_dim,
                              num_decoder_layers=args.num_decoder, encoder_dropout=args.encoder_dropout,
                              decoder_dropout=args.decoder_dropout, dim_feedforward=args.hidden_dim,
                              img_hidden_dim=args.img_hidden_dim, lm_dmodel=args.lm_hidden_dim, device=device, args=args).cuda()

    model = gazeformer(transformer, spatial_dim=(args.im_h, args.im_w), args=args,
                       subject_num=args.subject_num, subject_feature_dim=args.subject_feature_dim,
                       action_map_num=args.action_map_num,
                       dropout=args.cls_dropout, max_len=args.max_length).cuda()

    # encoder
    # transformer = Transformer(num_encoder_layers=args.num_encoder, nhead=args.nhead,
    #                           subject_feature_dim=args.subject_feature_dim, d_model=args.hidden_dim,
    #                           num_decoder_layers=args.num_decoder, encoder_dropout=args.encoder_dropout,
    #                           decoder_dropout=args.decoder_dropout, dim_feedforward=args.hidden_dim,
    #                           img_hidden_dim=args.img_hidden_dim, lm_dmodel=args.lm_hidden_dim, device=device).cuda()
    #
    # model = gazeformer(transformer, spatial_dim=(args.im_h, args.im_w), args=args,
    #                    subject_num=args.subject_num, subject_feature_dim=args.subject_feature_dim,
    #                    dropout=args.cls_dropout, max_len=args.max_length).cuda()

    # decoder
    # transformer = Transformer(num_encoder_layers=args.num_encoder, nhead=args.nhead,
    #                           d_model=args.hidden_dim,
    #                           num_decoder_layers=args.num_decoder, encoder_dropout=args.encoder_dropout,
    #                           decoder_dropout=args.decoder_dropout, dim_feedforward=args.hidden_dim,
    #                           img_hidden_dim=args.img_hidden_dim, lm_dmodel=args.lm_hidden_dim, device=device).cuda()
    #
    # model = gazeformer(transformer, spatial_dim=(args.im_h, args.im_w), args=args,
    #                    subject_num=args.subject_num, subject_feature_dim=args.subject_feature_dim,
    #                    action_map_num=args.action_map_num,
    #                    dropout=args.cls_dropout, max_len=args.max_length).cuda()

    sampling = Sampling(convLSTM_length=args.max_length, min_length=args.min_length,
                        map_width=args.im_w, map_height=args.im_h,
                        width=args.width, height=args.height)

    # Load checkpoint to start evaluation.
    # Infer iteration number through file name (it's hacky but very simple), so don't rename
    test_checkpoint = torch.load(os.path.join(checkpoints_dir, "checkpoint.pth"))
    for key in test_checkpoint:
        if key == "optimizer":
            continue
        else:
            model.load_state_dict(test_checkpoint[key])

    if len(args.gpu_ids) > 1:
        model = nn.DataParallel(model, args.gpu_ids)

    # get the human baseline score
    # human_metrics, human_metrics_std, gt_scores_of_each_images = human_evaluation_by_subject(test_loader)
    # logger.info("The metrics for human performance are: ")
    # for metrics_key in human_metrics.keys():
    #     for (key, value) in human_metrics[metrics_key].items():
    #         logger.info("{metrics_key:10}-{key:15}: {value:.4f} +- {std:.4f}".format
    #                     (metrics_key=metrics_key, key=key, value=value, std=human_metrics_std[metrics_key][key]))

    model.eval()
    repeat_num = args.eval_repeat_num
    all_gt_fix_vectors = []
    all_predict_fix_vectors = []
    predict_results = []
    with tqdm(total=len(test_loader) * repeat_num) as pbar_test:
        for i_batch, batch in enumerate(test_loader):
            tmp = [batch["images"], batch["fix_vectors"], batch["task_embeddings"], batch["subjects"]]
            tmp = [_ if not torch.is_tensor(_) else _.cuda() for _ in tmp]
            # merge the first two dim
            tmp = [_.view(-1, *_.shape[2:]) if torch.is_tensor(_) else _ for _ in tmp]
            images, gt_fix_vectors, task_embeddings, subjects = tmp
            # task = images.new_zeros((images.shape[0], args.lm_hidden_dim))

            N, _, C = images.shape

            with torch.no_grad():
                predict = model(src=images, subjects=subjects, task=task_embeddings)

            log_normal_mu = predict["log_normal_mu"]
            log_normal_sigma2 = predict["log_normal_sigma2"]
            all_actions_prob = predict["all_actions_prob"]

            image_prediction_dict = {_: [] for _ in range(len(batch["img_names"]))}
            all_gt_fix_vectors.extend(gt_fix_vectors)
            for trial in range(repeat_num):
                samples = sampling.random_sample(all_actions_prob, log_normal_mu, log_normal_sigma2)
                prob_sample_actions = samples["selected_actions_probs"]
                durations = samples["durations"]
                sample_actions = samples["selected_actions"]
                sampling_random_predict_fix_vectors, _, _ = sampling.generate_scanpath(
                    images, prob_sample_actions, durations, sample_actions)

                for idx in range(len(batch["img_names"])):
                    image_prediction_dict[idx].extend(
                        sampling_random_predict_fix_vectors[idx * args.subject_num:(idx + 1) * args.subject_num])

                # save the result to json
                for index in range(N):
                    image_idx = index // args.subject_num
                    subject_idx = index % args.subject_num
                    predict_result = dict()
                    one_sampling_random_predict_fix_vectors = sampling_random_predict_fix_vectors[index]
                    fix_vector_array = np.array(one_sampling_random_predict_fix_vectors.tolist())
                    predict_result["img_names"] = batch["img_names"][image_idx]
                    predict_result["task"] = batch["tasks"][image_idx][subject_idx]
                    predict_result["subject"] = subject_idx
                    predict_result["X"] = list(fix_vector_array[:, 0])
                    predict_result["Y"] = list(fix_vector_array[:, 1])
                    predict_result["T"] = list(fix_vector_array[:, 2] * 1000)
                    predict_result["length"] = len(predict_result["X"])
                    predict_results.append(predict_result)

                pbar_test.update(1)

            all_predict_fix_vectors.extend(list(image_prediction_dict.values()))

    cur_metrics, cur_metrics_std, score_details = comprehensive_evaluation_by_subject(all_gt_fix_vectors,
                                                                                      all_predict_fix_vectors,
                                                                                      args)

    score_details_list = []
    for value in score_details:
        score_details_list.extend(value)

    for idx in range(len(predict_results)):
        predict_results[idx]["score"] = score_details_list[idx]

    with open(predicts_file, 'w') as f:
        json.dump(predict_results, f, indent=2)

    # Print and log all evaluation metrics to tensorboard.
    logger.info("The metrics for best model performance are: ")
    for metrics_key in cur_metrics.keys():
        for (metric_name, metric_value) in cur_metrics[metrics_key].items():
            logger.info("{metrics_key:10}-{metric_name:15}: {metric_value:.4f}".format
                        (metrics_key=metrics_key, metric_name=metric_name, metric_value=metric_value))

if __name__ == "__main__":
    main()

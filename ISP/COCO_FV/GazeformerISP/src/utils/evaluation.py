import pandas as pd
import torch
import numpy as np
import scipy.stats
import copy

from tqdm import tqdm
import multimatch_gaze as multimatch
from utils.evaltools.scanmatch import ScanMatch
from utils.evaltools.visual_attention_metrics import string_edit_distance, scaled_time_delay_embedding_similarity

def comprehensive_evaluation_by_subject(gt_fix_vectors, predict_fix_vectors, args, is_eliminating_nan=True):
    # args.subject_num = 3
    # row: user prediction
    # col: gt prediction
    collect_multimatch_rlts = np.zeros((len(predict_fix_vectors), args.subject_num, args.subject_num, 5)) - 1
    collect_scanmatch_with_duration_rlts = np.zeros((len(predict_fix_vectors), args.subject_num, args.subject_num)) - 1
    collect_scanmatch_without_duration_rlts = np.zeros((len(predict_fix_vectors), args.subject_num, args.subject_num)) - 1
    collect_SED_rlts = np.zeros((len(predict_fix_vectors), args.subject_num, args.subject_num)) - 1
    collect_STDE_rlts = np.zeros((len(predict_fix_vectors), args.subject_num, args.subject_num)) - 1

    collect_multimatch_diag_rlts = np.zeros((len(predict_fix_vectors), args.subject_num, 5)) - 1
    collect_scanmatch_with_duration_diag_rlts = np.zeros((len(predict_fix_vectors), args.subject_num)) - 1
    collect_scanmatch_without_duration_diag_rlts = np.zeros((len(predict_fix_vectors), args.subject_num)) - 1
    collect_SED_diag_rlts = np.zeros((len(predict_fix_vectors), args.subject_num)) - 1
    collect_STDE_diag_rlts = np.zeros((len(predict_fix_vectors), args.subject_num)) - 1

    scores_of_each_images = np.zeros((len(predict_fix_vectors), args.subject_num, args.subject_num, 9)) - 1

    # create a ScanMatch object
    ScanMatchwithDuration = ScanMatch(Xres=args.width, Yres=args.height, Xbin=16, Ybin=12, Offset=(0, 0), TempBin=50, Threshold=3.5)
    ScanMatchwithoutDuration = ScanMatch(Xres=args.width, Yres=args.height, Xbin=16, Ybin=12, Offset=(0, 0), Threshold=3.5)

    stimulus = np.zeros((args.height, args.width, 3), dtype=np.float32)
    with tqdm(total=len(gt_fix_vectors)) as pbar:
        for index in range(len(gt_fix_vectors)):
            gt_fix_vector = gt_fix_vectors[index]
            predict_fix_vector = predict_fix_vectors[index]
            for row_idx in range(len(predict_fix_vector)):
                for col_idx in range(len(gt_fix_vector)):
                    scores_of_given_image = []
                    inner_gt_fix_vector = gt_fix_vector[col_idx]
                    inner_predict_fix_vector = predict_fix_vector[row_idx]
                    # calculate multimatch
                    mm_inner_gt_fix_vector = inner_gt_fix_vector.copy()
                    mm_inner_predict_fix_vector = inner_predict_fix_vector.copy()

                    if len(mm_inner_gt_fix_vector) < 3:
                        padding_vector = []
                        for _ in range(3 - len(mm_inner_gt_fix_vector)):
                            padding_vector.append((1., 1., 1e-3))
                        padding_vector = np.array(padding_vector, dtype={'names': ('start_x', 'start_y', 'duration'),
                                                                         'formats': ('f8', 'f8', 'f8')})
                        mm_inner_gt_fix_vector = np.concatenate([mm_inner_gt_fix_vector, padding_vector])
                    if len(mm_inner_predict_fix_vector) < 3:
                        padding_vector = []
                        for _ in range(3 - len(mm_inner_predict_fix_vector)):
                            padding_vector.append((1., 1., 1e-3))
                        padding_vector = np.array(padding_vector, dtype={'names': ('start_x', 'start_y', 'duration'),
                                                                         'formats': ('f8', 'f8', 'f8')})
                        mm_inner_predict_fix_vector = np.concatenate([mm_inner_predict_fix_vector, padding_vector])
                    rlt = multimatch.docomparison(mm_inner_gt_fix_vector, mm_inner_predict_fix_vector,
                                                  screensize=[args.width, args.height])
                    collect_multimatch_rlts[index, row_idx, col_idx] = np.array(rlt)
                    if row_idx == col_idx:
                        collect_multimatch_diag_rlts[index, row_idx] = np.array(rlt)
                    scores_of_given_image_with_gt = list(copy.deepcopy(rlt))

                    # perform scanmatch
                    # we need to transform the scale of time from s to ms
                    # with duration
                    np_fix_vector_1 = np.array([list(_) for _ in list(inner_gt_fix_vector)])
                    np_fix_vector_2 = np.array([list(_) for _ in list(inner_predict_fix_vector)])
                    np_fix_vector_1[:, -1] *= 1000
                    np_fix_vector_2[:, -1] *= 1000
                    sequence1_wd = ScanMatchwithDuration.fixationToSequence(np_fix_vector_1).astype(np.int32)
                    sequence2_wd = ScanMatchwithDuration.fixationToSequence(np_fix_vector_2).astype(np.int32)
                    (score, align, f) = ScanMatchwithDuration.match(sequence1_wd, sequence2_wd)
                    collect_scanmatch_with_duration_rlts[index, row_idx, col_idx] = score
                    if row_idx == col_idx:
                        collect_scanmatch_with_duration_diag_rlts[index, row_idx] = score
                    scores_of_given_image_with_gt.append(score)
                    # without duration
                    sequence1_wod = ScanMatchwithoutDuration.fixationToSequence(np_fix_vector_1).astype(np.int32)
                    sequence2_wod = ScanMatchwithoutDuration.fixationToSequence(np_fix_vector_2).astype(np.int32)
                    (score, align, f) = ScanMatchwithoutDuration.match(sequence1_wod, sequence2_wod)
                    collect_scanmatch_without_duration_rlts[index, row_idx, col_idx] = score
                    if row_idx == col_idx:
                        collect_scanmatch_without_duration_diag_rlts[index, row_idx] = score
                    scores_of_given_image_with_gt.append(score)

                    # perfrom SED
                    sed = string_edit_distance(stimulus, np_fix_vector_1, np_fix_vector_2)
                    collect_SED_rlts[index, row_idx, col_idx] = sed
                    if row_idx == col_idx:
                        collect_SED_diag_rlts[index, row_idx] = sed
                    scores_of_given_image_with_gt.append(sed)

                    # perfrom STDE
                    stde = scaled_time_delay_embedding_similarity(np_fix_vector_1, np_fix_vector_2, stimulus)
                    collect_STDE_rlts[index, row_idx, col_idx] = stde
                    if row_idx == col_idx:
                        collect_STDE_diag_rlts[index, row_idx] = stde
                    scores_of_given_image_with_gt.append(stde)

                    scores_of_each_images[index, row_idx, col_idx] = scores_of_given_image_with_gt

            pbar.update(1)

    collect_multimatch_diag_rlts = np.array(collect_multimatch_diag_rlts)
    collect_multimatch_diag_rlts = collect_multimatch_diag_rlts.reshape(-1, 5)
    collect_multimatch_diag_rlts = collect_multimatch_diag_rlts[(collect_multimatch_diag_rlts==-1).sum(-1)==0]
    if is_eliminating_nan:
        collect_multimatch_diag_rlts = collect_multimatch_diag_rlts[np.isnan(collect_multimatch_diag_rlts.sum(axis=1)) == False]
    multimatch_metric_mean = np.mean(collect_multimatch_diag_rlts, axis=0)
    multimatch_metric_std = np.std(collect_multimatch_diag_rlts, axis=0)

    collect_scanmatch_with_duration_diag_rlts = collect_scanmatch_with_duration_diag_rlts[
        collect_scanmatch_with_duration_diag_rlts!=-1]
    collect_scanmatch_without_duration_diag_rlts = collect_scanmatch_without_duration_diag_rlts[
        collect_scanmatch_without_duration_diag_rlts != -1]
    scanmatch_with_duration_metric_mean = np.mean(collect_scanmatch_with_duration_diag_rlts)
    scanmatch_with_duration_metric_std = np.std(collect_scanmatch_with_duration_diag_rlts)
    scanmatch_without_duration_metric_mean = np.mean(collect_scanmatch_without_duration_diag_rlts)
    scanmatch_without_duration_metric_std = np.std(collect_scanmatch_without_duration_diag_rlts)


    SED_metrics_rlts = np.array(collect_SED_diag_rlts)
    STDE_metrics_rlts = np.array(collect_STDE_diag_rlts)
    SED_metrics_rlts = SED_metrics_rlts.reshape(-1, len(gt_fix_vector))
    STDE_metrics_rlts = STDE_metrics_rlts.reshape(-1, len(gt_fix_vector))

    SED_metrics_rlts = SED_metrics_rlts[SED_metrics_rlts != -1]
    STDE_metrics_rlts = STDE_metrics_rlts[STDE_metrics_rlts != -1]
    SED_metrics_mean = SED_metrics_rlts.mean()
    SED_metrics_std = SED_metrics_rlts.std()
    STDE_metrics_mean = STDE_metrics_rlts.mean()
    STDE_metrics_std = STDE_metrics_rlts.std()

    SED_best_metrics= SED_metrics_rlts
    STDE_best_metrics = STDE_metrics_rlts
    SED_best_metrics_mean = SED_best_metrics.mean()
    SED_best_metrics_std = SED_best_metrics.std()
    STDE_best_metrics_mean = STDE_best_metrics.mean()
    STDE_best_metrics_std = STDE_best_metrics.std()

    cur_metrics = dict()
    cur_metrics_std = dict()

    multimatch_cur_metrics = dict()
    multimatch_cur_metrics["vector"] = multimatch_metric_mean[0]
    multimatch_cur_metrics["direction"] = multimatch_metric_mean[1]
    multimatch_cur_metrics["length"] = multimatch_metric_mean[2]
    multimatch_cur_metrics["position"] = multimatch_metric_mean[3]
    multimatch_cur_metrics["duration"] = multimatch_metric_mean[4]
    cur_metrics["MultiMatch"] = multimatch_cur_metrics

    scanmatch_cur_metrics = dict()
    scanmatch_cur_metrics["w/o duration"] = scanmatch_without_duration_metric_mean
    scanmatch_cur_metrics["with duration"] = scanmatch_with_duration_metric_mean
    cur_metrics["ScanMatch"] = scanmatch_cur_metrics

    multimatch_cur_metrics_std = dict()
    multimatch_cur_metrics_std["vector"] = multimatch_metric_std[0]
    multimatch_cur_metrics_std["direction"] = multimatch_metric_std[1]
    multimatch_cur_metrics_std["length"] = multimatch_metric_std[2]
    multimatch_cur_metrics_std["position"] = multimatch_metric_std[3]
    multimatch_cur_metrics_std["duration"] = multimatch_metric_std[4]
    cur_metrics_std["MultiMatch"] = multimatch_cur_metrics_std

    scanmatch_cur_metrics_std = dict()
    scanmatch_cur_metrics_std["w/o duration"] = scanmatch_without_duration_metric_std
    scanmatch_cur_metrics_std["with duration"] = scanmatch_with_duration_metric_std
    cur_metrics_std["ScanMatch"] = scanmatch_cur_metrics_std

    VAME_cur_metrics = dict()
    VAME_cur_metrics["SED"] = SED_metrics_mean
    VAME_cur_metrics["STDE"] = STDE_metrics_mean
    VAME_cur_metrics["SED_best"] = SED_best_metrics_mean
    VAME_cur_metrics["STDE_best"] = STDE_best_metrics_mean
    cur_metrics["VAME"] = VAME_cur_metrics

    VAME_cur_metrics_std = dict()
    VAME_cur_metrics_std["SED"] = SED_metrics_std
    VAME_cur_metrics_std["STDE"] = STDE_metrics_std
    VAME_cur_metrics_std["SED_best"] = SED_best_metrics_std
    VAME_cur_metrics_std["STDE_best"] = STDE_best_metrics_std
    cur_metrics_std["VAME"] = VAME_cur_metrics_std

    scanmatch_with_duration_p2g = p2g(len(gt_fix_vectors), args.subject_num, collect_scanmatch_with_duration_rlts, mode="max", return_ranks=False)
    # scanmatch_with_duration_g2p = g2p(len(gt_fix_vectors), 15, collect_scanmatch_with_duration_rlts, mode="max", return_ranks=False)
    #
    # scanmatch_without_duration_p2g = p2g(len(gt_fix_vectors), 15, collect_scanmatch_without_duration_rlts, mode="max",
    #                                   return_ranks=False)
    # scanmatch_without_duration_g2p = g2p(len(gt_fix_vectors), 15, collect_scanmatch_without_duration_rlts, mode="max",
    #                                   return_ranks=False)

    cur_metrics["retrieval scanmatch w/ duration"] = {
        "pmrr": scanmatch_with_duration_p2g[4],
        "pr1": scanmatch_with_duration_p2g[0],
        "pr3": scanmatch_with_duration_p2g[1],
        "pr5": scanmatch_with_duration_p2g[2],
        "rsum": scanmatch_with_duration_p2g[0] + scanmatch_with_duration_p2g[1] + scanmatch_with_duration_p2g[2]
    }


    scores_of_each_images = scores_of_each_images.tolist()

    return cur_metrics, cur_metrics_std, scores_of_each_images

def p2g(npts, subject_num, scores, mode="max", return_ranks=False):
    """
    Prediction->Ground-Truth
    scores: (K, N, N) matrix of similarity
    mode: max -> larger the better
        : min -> smaller the better
    """
    # npts = images.shape[0]

    if mode == "max":
        pass
    else:
        scores = -scores

    ranks = np.zeros((npts, subject_num))
    top1 = np.zeros((npts, subject_num))


    for index in range(npts):
        for subj in range(subject_num):
            if scores[index, subj].sum() == - scores[index, subj].shape[0]:
                ranks[index, subj] = -1
                top1[index, subj] = -1
            else:
                inds = np.argsort(scores[index, subj])[::-1]
                ranks[index, subj] = np.where(inds == subj)[0][0]
                top1[index, subj] = inds[0]

    # Compute metrics
    tmp_rank = ranks.reshape(-1)
    tmp_top1 = top1.reshape(-1)
    tmp_rank = tmp_rank[tmp_rank != -1]
    tmp_top1 = tmp_top1[tmp_top1 != -1]
    r1 = 100.0 * len(np.where(tmp_rank < 1)[0]) / len(tmp_rank)
    r3 = 100.0 * len(np.where(tmp_rank < 2)[0]) / len(tmp_rank)
    r5 = 100.0 * len(np.where(tmp_rank < 5)[0]) / len(tmp_rank)
    r10 = 100.0 * len(np.where(tmp_rank < 10)[0]) / len(tmp_rank)
    medr = np.floor(np.median(tmp_rank)) + 1
    meanr = tmp_rank.mean() + 1
    mrr = (1 / (tmp_rank + 1)).mean()
    if return_ranks:
        return (r1, r3, r5, r10, mrr, medr, meanr), (ranks, top1)
    else:
        return (r1, r3, r5, r10, mrr, medr, meanr)

def g2p(npts, subject_num, scores, mode="max", return_ranks=False):
    """
    Prediction->Ground-Truth
    scores: (K, N, N) matrix of similarity
    mode: max -> larger the better
        : min -> smaller the better
    """
    # npts = images.shape[0]

    if mode == "max":
        pass
    else:
        scores = -scores

    ranks = np.zeros((npts, subject_num))
    top1 = np.zeros((npts, subject_num))


    for index in range(npts):
        for subj in range(subject_num):
            inds = np.argsort(scores[index, :, subj])[::-1]
            ranks[index, subj] = np.where(inds == subj)[0][0]
            top1[index, subj] = inds[0]

    # Compute metrics
    tmp_rank = ranks.reshape(-1)
    tmp_top1 = top1.reshape(-1)
    r1 = 100.0 * len(np.where(tmp_rank < 1)[0]) / len(tmp_rank)
    r3 = 100.0 * len(np.where(tmp_rank < 3)[0]) / len(tmp_rank)
    r5 = 100.0 * len(np.where(tmp_rank < 5)[0]) / len(tmp_rank)
    r10 = 100.0 * len(np.where(tmp_rank < 10)[0]) / len(tmp_rank)
    medr = np.floor(np.median(tmp_rank)) + 1
    meanr = tmp_rank.mean() + 1
    mrr = (1 / (tmp_rank + 1)).mean()
    if return_ranks:
        return (r1, r3, r5, r10, mrr, medr, meanr), (ranks, top1)
    else:
        return (r1, r3, r5, r10, mrr, medr, meanr)


def pairs_eval(gt_fix_vectors, predict_fix_vectors, ScanMatchwithDuration, ScanMatchwithoutDuration,
               is_eliminating_nan=True):
    pairs_summary_metric = []
    stimulus = np.zeros((320, 512, 3), dtype=np.float32)
    for index in range(len(gt_fix_vectors)):
        gt_fix_vector = gt_fix_vectors[index]
        predict_fix_vector = predict_fix_vectors[index]
        collect_rlts = []

        mm_gt_fix_vector = gt_fix_vector.copy()
        mm_predict_fix_vector = predict_fix_vector.copy()

        if len(mm_gt_fix_vector) < 3:
            padding_vector = []
            for _ in range(3 - len(mm_gt_fix_vector)):
                padding_vector.append((1., 1., 1e-3))
            padding_vector = np.array(padding_vector,
                                      dtype={'names': ('start_x', 'start_y', 'duration'),
                                             'formats': ('f8', 'f8', 'f8')})
            mm_gt_fix_vector = np.concatenate([mm_gt_fix_vector, padding_vector])
        if len(mm_predict_fix_vector) < 3:
            padding_vector = []
            for _ in range(3 - len(mm_predict_fix_vector)):
                padding_vector.append((1., 1., 1e-3))
            padding_vector = np.array(padding_vector,
                                      dtype={'names': ('start_x', 'start_y', 'duration'),
                                             'formats': ('f8', 'f8', 'f8')})
            mm_predict_fix_vector = np.concatenate([mm_predict_fix_vector, padding_vector])
        rlt = multimatch.docomparison(mm_gt_fix_vector, mm_predict_fix_vector, screensize=[320, 240])

        if np.any(np.isnan(rlt)):
            rlt = list(rlt)
            rlt.extend([np.nan, np.nan, np.nan, np.nan])
            collect_rlts.append(rlt)
        else:
            # perform scanmatch
            # we need to transform the scale of time from s to ms
            # with duration
            np_fix_vector_1 = np.array([list(_) for _ in list(gt_fix_vector)])
            np_fix_vector_2 = np.array([list(_) for _ in list(predict_fix_vector)])
            np_fix_vector_1[:, -1] *= 1000
            np_fix_vector_2[:, -1] *= 1000

            sequence1_wd = ScanMatchwithDuration.fixationToSequence(np_fix_vector_1).astype(np.int32)
            sequence2_wd = ScanMatchwithDuration.fixationToSequence(np_fix_vector_2).astype(np.int32)
            (score_wd, align_wd, f_wd) = ScanMatchwithDuration.match(sequence1_wd, sequence2_wd)
            # without duration
            sequence1_wod = ScanMatchwithoutDuration.fixationToSequence(np_fix_vector_1).astype(np.int32)
            sequence2_wod = ScanMatchwithoutDuration.fixationToSequence(np_fix_vector_2).astype(np.int32)
            (score_wod, align_wod, f_wod) = ScanMatchwithoutDuration.match(sequence1_wod, sequence2_wod)

            # perfrom SED
            sed = string_edit_distance(stimulus, np_fix_vector_1, np_fix_vector_2)
            # perfrom STDE
            stde = scaled_time_delay_embedding_similarity(np_fix_vector_1, np_fix_vector_2, stimulus)

            rlt = list(rlt)
            rlt.extend([score_wod, score_wd, sed, stde])
            collect_rlts.append(rlt)
        collect_rlts = np.array(collect_rlts)
        if is_eliminating_nan:
            collect_rlts = collect_rlts[np.isnan(collect_rlts.sum(axis=1)) == False]
        if collect_rlts.shape[0] != 0:
            metric_mean = np.sum(collect_rlts, axis=0)
            metric_value = np.zeros((11,), dtype=np.float32)
            metric_value[:7] = metric_mean[:7]
            metric_value[7] = metric_mean[7]
            metric_value[8] = metric_mean[8]
            metric_value[9] = collect_rlts[:, 7].min()
            metric_value[10] = collect_rlts[:, 8].max()
        else:
            metric_value = np.array([np.nan] * 11)
        pairs_summary_metric.append(metric_value)

    return np.array(pairs_summary_metric)

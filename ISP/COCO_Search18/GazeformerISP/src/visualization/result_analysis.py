import os
import sys
import json
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
import scipy.io
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

def per_task_performance():
    log_dir = 'TP-original-ex-789-fewshot-10-random1'
    shot = 10
    sample_support = log_dir[-1]
    with open(f'result/{log_dir}/log/prediction.json') as f1:
        pred = json.load(f1)
    with open(f'result/{log_dir}/sampling/{shot}_shot/sample_{sample_support}.json') as f2:
        train = json.load(f2)
    
    train_task_list = set([t['task'] for t in train])
    pred_task_list = set([t['task'] for t in pred])
    task_avg_performance = {key: np.array([0.,0.,0.]) for key in pred_task_list}
    task_avg_p_num = {key: 0 for key in pred_task_list}

    print(train_task_list)

    for sp in pred:
        score = np.array(sp['score']).mean(axis=0)
        mm = score[0:5].mean()
        sm = score[5:7].mean()
        sed = score[7]
        task_avg_performance[sp['task']] += np.array([mm, sm, sed])
        task_avg_p_num[sp['task']] += 1
    
    for key, value in task_avg_performance.items():
        value /= task_avg_p_num[key]
    
    seen_task, unseen_task = np.array([0., 0., 0.]), np.array([0., 0., 0.])
    for key, value in task_avg_performance.items():
        if key not in train_task_list:
        else:
            seen_task += value
    unseen_task /= (18 - len(train_task_list))
    seen_task /= len(train_task_list)

    print(unseen_task, seen_task)
    
def per_task_performance_useremb():
    log_dir = 'TP-useremb-ex-789-task-img-encoder'
    shot = 10
    sample_support = 7
    with open(f'result/{log_dir}/log/prediction.json') as f1:
        pred = json.load(f1)
    with open(f'../../../PHAT/result/{log_dir}/sampling/{shot}_shot/sample_{sample_support}.json') as f2:
        train = json.load(f2)
    
    train_task_list = set([t['task'] for t in train])
    pred_task_list = set([t['task'] for t in pred])
    task_avg_performance = {key: np.array([0.,0.,0.]) for key in pred_task_list}
    task_avg_p_num = {key: 0 for key in pred_task_list}

    print(train_task_list)

    for sp in pred:
        score = np.array(sp['score']).mean(axis=0)
        mm = score[0:5].mean()
        sm = score[5:7].mean()
        sed = score[7]
        task_avg_performance[sp['task']] += np.array([mm, sm, sed])
        task_avg_p_num[sp['task']] += 1
    
    for key, value in task_avg_performance.items():
        value /= task_avg_p_num[key]
    
    seen_task, unseen_task = np.array([0., 0., 0.]), np.array([0., 0., 0.])
    for key, value in task_avg_performance.items():
        if key not in train_task_list:
            unseen_task += value
        else:
            seen_task += value
    unseen_task /= (18 - len(train_task_list))
    seen_task /= len(train_task_list)

    print(unseen_task, seen_task)

    
        



per_task_performance()
# per_task_performance_useremb()
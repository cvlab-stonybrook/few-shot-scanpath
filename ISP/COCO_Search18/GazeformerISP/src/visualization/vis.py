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

def draw_scanpath(gt_X, gt_Y, gt_T, pred_X, pred_Y, pred_T, img_name, img_root, save_path):
    img = mpimg.imread(f'{img_root}/{img_name}')
    fig, ax = plt.subplots(1, 2, figsize=(30, 10))
    # draw gt scanpath
    ax[0].imshow(img)
    for fix_idx in range(len(gt_Y)):
        x, y, t = gt_X[fix_idx], gt_Y[fix_idx], gt_T[fix_idx]
        t = clamp(t/8, 15, 30)
        if fix_idx == len(gt_X) - 1:
            circle = patches.Circle((x, y), radius=t, edgecolor='black', facecolor='red', alpha=0.5)
        else:
            circle = patches.Circle((x, y), radius=t, edgecolor='black', facecolor='yellow', alpha=0.5)
        ax[0].add_patch(circle)
        ax[0].annotate(str(fix_idx), xy=(x, y + 3), fontsize=30, ha="center", va="center")
        if fix_idx < len(gt_X)-1:
            next_x, next_y = gt_X[fix_idx+1], gt_Y[fix_idx+1]
            ax[0].plot([x, next_x], [y, next_y], color='yellow', alpha=0.5)
    ax[0].set_axis_off()
    # draw prediction
    ax[1].imshow(img)
    for fix_idx in range(len(pred_X)):
        x, y, t = pred_X[fix_idx], pred_Y[fix_idx], pred_T[fix_idx]
        t = clamp(t/10, 10, 30)
        if fix_idx == len(pred_X) - 1:
            circle = patches.Circle((x, y), radius=t, edgecolor='black', facecolor='red', alpha=0.5)
        else:
            circle = patches.Circle((x, y), radius=t, edgecolor='black', facecolor='yellow', alpha=0.5)
        ax[1].add_patch(circle)
        ax[1].annotate(str(fix_idx), xy=(x, y + 3), fontsize=30, ha="center", va="center")
        if fix_idx < len(pred_X)-1:
            next_x, next_y = pred_X[fix_idx+1], pred_Y[fix_idx+1]
            ax[1].plot([x, next_x], [y, next_y], color='yellow', alpha=0.5)
    ax[1].set_axis_off()
    print(save_path)
    plt.savefig(f'{save_path}/{img_name}')
    plt.close()


def visual_results():
    log_dir = 'TP-original-ex-789-only-subject-emb-fewshot-1-random=4'
    subject_id = 2

    img_root = '../../../PHAT/data/images'
    with open(f'src/data/TPTA/TP_fixations.json') as  f1:
        gt_data = json.load(f1)
    with open(f'result/{log_dir}/log/prediction.json') as  f2:
        pred_data = json.load(f2)
    print(len(gt_data))
    print(len(pred_data))
    gt_data = list(filter(lambda x: x['subject'] == subject_id, gt_data))
    gt_data = list(filter(lambda x: x['split'] == 'test', gt_data))

    pred_data = list(filter(lambda x: x['subject'] == subject_id, pred_data))
    
    for data_idx, d in enumerate(pred_data):
        # if data_idx > 50:
        #     break
        img_name, task, pred_X, pred_Y, pred_T = d['img_names'], d['task'], d['X'], d['Y'], d['T']
        if img_name.split('/')[-1] not in ['000000459825.jpg', '000000042526.jpg', '000000331475.jpg', 
                            '000000165257.jpg', '000000546934.jpg', '000000341933.jpg',
                            '000000460312.jpg', '000000218811.jpg', '000000118108.jpg',
                            '000000411093.jpg', '000000418623.jpg']:
            continue
        # img_name, pred_X, pred_Y = d['name'], d['X'], d['Y']
        gt_X, gt_Y, gt_T = gt_data[data_idx]['X'], gt_data[data_idx]['Y'], gt_data[data_idx]['T']
        save_path = f'result/{log_dir}/{subject_id}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(os.path.join(save_path, task)):
            os.makedirs(os.path.join(save_path, task))
        print(save_path)
        draw_scanpath(gt_X, gt_Y, gt_T, pred_X, pred_Y, pred_T, img_name, img_root, save_path)


def visual_attn_task():
    split = 'val'
    # log_dir = 'TP-useremb-ex-789-task-img-encoder'
    log_dir = 'TP-useremb-ex-789-img-task-encoder-bbox-loss-new'
    # img_root = '../saliency/data/OSIE/train'
    img_root = '../../../PHAT/data/images'
    with open(
        f'/data10/shared/ruoyu/project/PHAT/assets/{log_dir}/{split}_attn_weights.pkl', \
            'rb') as pickle_file:
        attn_weights = pickle.load(pickle_file)
    print(len(attn_weights))
    
    for data_idx, d in enumerate(attn_weights):
        img_name, subject, task, fixs, duration, img_attn, fix_attn = \
            d['name'], d['subject'], d['task'], d['fixs'], d['duration'], d['img_attn'], d['fix_attn']
        task = task.replace(' ', '_')
        # task = 'test'
        X, Y = np.array(fixs[:,0], dtype=float), np.array(fixs[:,1], dtype=float)
        save_path = f'result/{log_dir}/{split}_attn_weights'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        img = Image.open(f'{img_root}/{task}/{img_name}').resize((512, 320))
        fig, ax = plt.subplots(1, 2, figsize=(30, 10))
        # draw img and attn_weights
        ax[0].imshow(img)
        print(img_attn.shape)
        # img_attn = img_attn[0, 1:]  # vis for task encoder
        img_attn = img_attn[0, 2:]  # vis for bbox attn weights
        img_attn = (img_attn - img_attn.min())/(img_attn.max() - img_attn.min())
        img_attn = img_attn.reshape(10, 16)
        img_attn = torch.nn.functional.interpolate(
            img_attn.unsqueeze(0).unsqueeze(0), size=(320, 512),
            mode='bilinear',align_corners=False).squeeze(0).squeeze(0).numpy()
        img_attn = (img_attn - img_attn.min()) / (img_attn.max() - img_attn.min())
        ax[0].imshow(img_attn, alpha=0.6, cmap='jet', interpolation='nearest')
        ax[0].set_axis_off()
        # ax[0].text(0.95, 0.95, d['win'].item(), fontsize=20, ha='right', va='top', color='yellow', transform=ax[0].transAxes)
        ax[0].text(0, 0.95, task, fontsize=30, ha='right', va='top', color='black', transform=ax[0].transAxes)
        ax[1].imshow(img)
        max_fix_attn_idx = torch.argmax(fix_attn)
        for fix_idx in range(len(X)):
            x, y, fix_a = X[fix_idx], Y[fix_idx], round(fix_attn[0][fix_idx].item(), 1)
            t = duration[fix_idx]
            t = clamp(t/8, 15, 40) * 1.8
            if fix_idx == len(X) - 1:
                circle = patches.Circle((x, y), radius=t, edgecolor='black', facecolor='red', alpha=0.5)
                ax[1].annotate(str(fix_idx), xy=(x, y + 3), fontsize=20, ha="center", va="center", alpha=0.7)
                ax[1].add_patch(circle)
            elif (X[fix_idx+1] == x and Y[fix_idx+1] == y):
                circle = patches.Circle((x, y), radius=t, edgecolor='black', facecolor='red', alpha=0.5)
                ax[1].annotate(str(fix_idx), xy=(x, y + 3), fontsize=20, ha="center", va="center", alpha=0.7)
                ax[1].add_patch(circle)
                break
                # circle = patches.Circle((x, y), radius=20, edgecolor='black', facecolor='red', alpha=0.5)
            else:
                circle = patches.Circle((x, y), radius=t, edgecolor='black', facecolor='yellow', alpha=0.5)
                # circle = patches.Circle((x, y), radius=20, edgecolor='black', facecolor='yellow', alpha=0.5)
                ax[1].add_patch(circle)
                ax[1].annotate(str(fix_idx), xy=(x, y + 3), fontsize=20, ha="center", va="center", alpha=0.7)

            if fix_idx < len(X)-1:
                next_x, next_y = X[fix_idx+1], Y[fix_idx+1]
                ax[1].plot([x, next_x], [y, next_y], color='yellow', alpha=0.5)
        ax[1].set_axis_off()


        plt.savefig('{}/{}_{}.jpg'.format(save_path, img_name.split('.')[0], subject))
        # plt.savefig('{}/{}_{}.jpg'.format(save_path, img_name.split('.')[0].split('/')[1], subject))
        plt.close()

def clamp(number, min_value, max_value):
    return max(min(number, max_value), min_value)

if __name__ == '__main__':
    visual_results()
    # visual_attn_task()


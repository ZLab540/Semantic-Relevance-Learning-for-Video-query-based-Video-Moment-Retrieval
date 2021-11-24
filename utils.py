" Some useful functions "

import numpy as np
from six.moves import xrange
import time
import pickle
import operator
import torch


def chosen_ten_unit(N):
    idx = []

    if N == 1:
        idx = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif N == 2:
        idx = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    elif N == 3:
        idx = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
    elif N == 4:
        idx = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3]
    elif N == 5:
        idx = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    elif N == 6:
        idx = [0, 1, 1, 2, 2, 3, 3, 4, 4, 5]
    elif N == 7:
        idx = [0, 1, 2, 2, 3, 3, 4, 4, 5, 6]
    elif N == 8:
        idx = [0, 1, 2, 3, 4, 4, 5, 5, 6, 7]
    elif N == 9:
        idx = [0, 1, 2, 3, 4, 4, 5, 6, 7, 8]
    else:
        five_unit = N / float(11)
        for i in range(10):
            idx.append(int(np.floor((i + 1) * five_unit)))

    return idx


def chosen_feature(N):
    max_video_length = 40
    idx = []
    chosen_unit = N / float(max_video_length + 1)
    for i in range(max_video_length):
        idx.append(int(np.floor((i + 1) * chosen_unit)))
    return idx


def calculate_reward_batch_withstop(Previous_IoU, current_IoU, t):
    batch_size = len(Previous_IoU)
    reward = torch.zeros(batch_size)

    for i in range(batch_size):
        if current_IoU[i] > Previous_IoU[i] and Previous_IoU[i] >= 0:
            reward[i] = 1 - 0.001 * t
        elif current_IoU[i] <= Previous_IoU[i] and current_IoU[i] >= 0:
            reward[i] = -0.001 * t
        else:
            reward[i] = -1 - 0.001 * t
    return reward


def calculate_reward(Previous_IoU, current_IoU, t):
    if current_IoU > Previous_IoU and Previous_IoU >= 0:
        reward = 1 - 0.001 * t
    elif current_IoU <= Previous_IoU and current_IoU >= 0:
        reward = -0.001 * t
    else:
        reward = -1 - 0.001 * t
    return reward


def calculate_RL_IoU_batch(i0, i1):
    # calculate temporal intersection over union
    batch_size = len(i0)
    iou_batch = torch.zeros(batch_size)

    for i in range(len(i0)):
        union = (min(i0[i][0], i1[i][0]), max(i0[i][1], i1[i][1]))
        inter = (max(i0[i][0], i1[i][0]), min(i0[i][1], i1[i][1]))
        iou = 1.0 * (inter[1] - inter[0]) / (union[1] - union[0])
        iou_batch[i] = iou
    return iou_batch


def calculate_IoU(i0, i1):
    # calculate temporal intersection over union
    union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
    inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))
    iou = 1.0 * (inter[1] - inter[0]) / (union[1] - union[0])
    return iou


def compute_IoU_recall_top_n_forreg_rl(top_n, iou_thresh, video_query_reg_mat, gt):
    correct_num = 0.0
    gt_start = gt[0][0]
    gt_end = gt[0][1]

    pred_start = video_query_reg_mat[0]
    pred_end = video_query_reg_mat[1]
    iou = calculate_IoU((gt_start, gt_end), (pred_start, pred_end))

    if iou >= iou_thresh:
        correct_num += 1

    return correct_num

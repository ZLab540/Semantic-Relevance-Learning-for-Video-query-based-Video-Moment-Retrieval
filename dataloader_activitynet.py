import torch
import torch.utils.data
import os
import pickle
import numpy as np
import math
from utils import *
import random
import glob
import json

class Activitynet_Train_dataset(torch.utils.data.Dataset):

    def __init__(self):

        self.data_dir = "./data/"
        with open('./data/train.json', 'r') as f:
            self.data = json.load(f)
        self.num_data = len(self.data)
        print(self.num_data, "train data are readed")

        videos = [[] for _ in range(4)]
        for i in self.data:
            videos[0].append(i['id'])
            videos[1].append(i['groundtruth'])
            videos[2].append(i['label'])
            videos[3].append(i['location'])

        self.all_ids = videos[0]
        self.all_gts = torch.Tensor(videos[1])
        self.all_labels = torch.Tensor(videos[2])
        self.all_locs = torch.Tensor(videos[3])

    def read_video_reference_feats(self, video_id, video_loc):

        data_dir = self.data_dir

        feat = np.load('%s/feat/v_%s.npy' % (data_dir, video_id))
        length = video_loc[1] - video_loc[0] + 1  # N
        ten_len = length / 10
        four_len = length / 4
        oneinfour_len = four_len
        threeinfour_len = length - four_len

        original_feat = np.zeros([500, 500], dtype=np.float32)
        original_feat_1 = feat[int(video_loc[0]):int(video_loc[1]) + 1]
        for i in range(original_feat_1.shape[0]):
            original_feat[i] = original_feat_1[i]
        global_feature = np.mean(original_feat_1, axis=0)

        initial_feat = original_feat[int(oneinfour_len):int(threeinfour_len)]
        initial_feature = np.mean(initial_feat, axis=0)

        initial_offset_start = oneinfour_len
        initial_offset_end = threeinfour_len - 1

        initial_offset_start_norm = initial_offset_start / float(length - 1)
        initial_offset_end_norm = initial_offset_end / float(length - 1)

        return original_feat, global_feature, initial_feature, initial_offset_start, \
               initial_offset_end, initial_offset_start_norm, initial_offset_end_norm, ten_len, length

    def read_video_query_feats(self, video_id, video_gt, video_loc):

        data_dir = self.data_dir

        feat = np.load('%s/feat/v_%s.npy' % (data_dir, video_id))
        query_feat_1 = feat[int(video_loc[0]):int(video_loc[1]) + 1]
        query_feat = query_feat_1[int(video_gt[0]):int(video_gt[1]) + 1]
        query_feature = np.mean(query_feat, axis=0)

        return query_feature

    def sample(self, query_id, query_gt, query_label, query_loc,
               all_ids, all_gts, all_labels, all_locs):
        """
        For each query video, random sample a reference video
        :params:
        query_id: Query video id.
        query_gt: The action segment in the query video.
        query_label: The action category of the query video.
        query_loc: The location of the video segment in the original whole video.
        all_ids: All video ids.
        all_gts: The action segments in all the videos.
        all_labels: The action categories of all the videos.
        all_locs: The locations of the all videos in their corresponding original
            whole video.
        :return:
        chosen_id: The reference video id.
        chosen_gt: The action segment in the reference video.
        chosen_loc: The location of the reference video in the original whole video.
        """
        same = torch.eq(all_labels, query_label)
        query_gt = torch.Tensor(query_gt)
        longer = torch.le(query_gt[1] - query_gt[0], all_locs[:, 1] - all_locs[:, 0])
        same = same.mul(longer)
        same = torch.nonzero(same)

        num = same.size()[0]

        idx = torch.randint(num, [], dtype=torch.int32)
        idx = same[idx, 0]
        chosen_id = all_ids[idx]
        chosen_gt = all_gts[idx]
        chosen_loc = all_locs[idx]

        # Data augmentation during training
        max1 = chosen_gt[0]
        high1 = int(max1.data) + 1
        off_st = torch.randint(high1, [], dtype=torch.int32)
        max2 = chosen_loc[1] - chosen_loc[0] - chosen_gt[1] + 1
        high2 = int(max2.data)
        off_en = torch.randint(high2, [], dtype=torch.int32)
        use_off = torch.randint(1, [])
        if use_off > 0.9:
            off_st = 0
            off_en = 0
        off_gt = torch.stack([-off_st, -off_st])
        off_loc = torch.stack([off_st, -off_en])
        chosen_gt += off_gt
        chosen_loc += off_loc
        return chosen_id, chosen_gt, chosen_loc

    def __getitem__(self, index):

        samples = self.data[index]

        query_id = samples['id']
        query_loc = samples['location']
        query_gt = samples['groundtruth']
        query_label = samples['label']

        query_feature = self.read_video_query_feats(query_id, query_gt, query_loc)

        # reference
        chosen_id, chosen_gt, chosen_loc = self.sample(query_id, query_gt, query_label, query_loc, self.all_ids,
                                                       self.all_gts,
                                                       self.all_labels, self.all_locs)
        offset = np.zeros(2, dtype=np.float32)
        offset_norm = np.zeros(2, dtype=np.float32)
        initial_offset = np.zeros(2, dtype=np.float32)
        initial_offset_norm = np.zeros(2, dtype=np.float32)

        original_feat, global_feature, initial_feature, initial_offset_start, \
        initial_offset_end, initial_offset_start_norm, initial_offset_end_norm, ten_len, length = self.read_video_reference_feats(
            chosen_id, chosen_loc)

        # offset
        offset[0] = chosen_gt[0]
        offset[1] = chosen_gt[1]

        offset_norm[0] = offset[0] / float(length - 1)
        offset_norm[1] = offset[1] / float(length - 1)

        initial_offset[0] = initial_offset_start
        initial_offset[1] = initial_offset_end

        initial_offset_norm[0] = initial_offset_start_norm
        initial_offset_norm[1] = initial_offset_end_norm

        return original_feat, global_feature, initial_feature, query_feature, offset_norm, initial_offset, initial_offset_norm, ten_len, length

    def __len__(self):
        return self.num_data


class Activitynet_Test_dataset(torch.utils.data.Dataset):

    def __init__(self):

        self.data_dir = "./data/"
        with open('data/test.json', 'r') as f:
            self.data = json.load(f)
        self.num_data = len(self.data)
        print(self.num_data, "test data are readed")
        videos = [[] for _ in range(4)]

        for i in self.data:
            videos[0].append(i['id'])
            videos[1].append(i['groundtruth'])
            videos[2].append(i['label'])
            videos[3].append(i['location'])

        self.all_ids = videos[0]
        self.all_gts = torch.Tensor(videos[1])
        self.all_labels = torch.Tensor(videos[2])
        self.all_locs = torch.Tensor(videos[3])

    def read_video_reference_feats(self, video_id, video_loc):

        data_dir = self.data_dir
        feat = np.load('%s/feat/v_%s.npy' % (data_dir, video_id))

        length = video_loc[1] - video_loc[0] + 1
        ten_len = length / 10
        four_len = length / 4
        oneinfour_len = four_len
        threeinfour_len = length - four_len

        original_feat = np.zeros([500, 500], dtype=np.float32)
        original_feat_1 = feat[int(video_loc[0]):int(video_loc[1]) + 1]
        for i in range(original_feat_1.shape[0]):
            original_feat[i] = original_feat_1[i]
        global_feature = np.mean(original_feat_1, axis=0)

        initial_feat = original_feat[int(oneinfour_len):int(threeinfour_len)]
        initial_feature = np.mean(initial_feat, axis=0)

        initial_offset_start = oneinfour_len
        initial_offset_end = threeinfour_len - 1

        initial_offset_start_norm = initial_offset_start / float(length - 1)
        initial_offset_end_norm = initial_offset_end / float(length - 1)

        return original_feat, global_feature, initial_feature, initial_offset_start, initial_offset_end, \
               initial_offset_start_norm, initial_offset_end_norm, ten_len, length

    def read_video_query_feats(self, video_id, video_gt, video_loc):

        data_dir = self.data_dir

        feat = np.load('%s/feat/v_%s.npy' % (data_dir, video_id))
        query_feat_1 = feat[int(video_loc[0]):int(video_loc[1]) + 1]
        query_feat = query_feat_1[int(video_gt[0]):int(video_gt[1]) + 1]
        query_feature = np.mean(query_feat, axis=0)

        return query_feature

    def sample(self, query_id, query_gt, query_label, query_loc, all_ids, all_gts,
               all_labels, all_locs):

        same = torch.eq(all_labels, query_label)
        query_gt = torch.Tensor(query_gt)
        longer = torch.le(query_gt[1] - query_gt[0], all_locs[:, 1] - all_locs[:, 0])
        same = same.mul(longer)
        same = torch.nonzero(same)

        num = same.size()[0]

        idx = torch.randint(num, [], dtype=torch.int32)
        idx = same[idx, 0]
        chosen_id = all_ids[idx]
        chosen_gt = all_gts[idx]
        chosen_loc = all_locs[idx]

        return chosen_id, chosen_gt, chosen_loc

    def __getitem__(self, index):

        # query
        samples = self.data[index]
        query_id = samples['id']
        query_loc = samples['location']
        query_gt = samples['groundtruth']
        query_label = samples['label']

        query_feature = self.read_video_query_feats(query_id, query_gt, query_loc)

        # reference
        chosen_id, chosen_gt, chosen_loc = self.sample(query_id, query_gt, query_label, query_loc, self.all_ids,
                                                       self.all_gts,
                                                       self.all_labels, self.all_locs)

        original_feat, global_feature, initial_feature, initial_offset_start, \
        initial_offset_end, initial_offset_start_norm, initial_offset_end_norm, ten_len, length = self.read_video_reference_feats(
            chosen_id, chosen_loc)

        # offset
        offset = np.zeros(2, dtype=np.float32)
        initial_offset = np.zeros(2, dtype=np.float32)
        initial_offset_norm = np.zeros(2, dtype=np.float32)

        # offset
        offset[0] = chosen_gt[0]
        offset[1] = chosen_gt[1]

        initial_offset[0] = initial_offset_start
        initial_offset[1] = initial_offset_end

        initial_offset_norm[0] = initial_offset_start_norm
        initial_offset_norm[1] = initial_offset_end_norm

        return original_feat, global_feature, initial_feature, query_feature, offset, initial_offset, initial_offset_norm, ten_len, length

    def __len__(self):
        return self.num_data

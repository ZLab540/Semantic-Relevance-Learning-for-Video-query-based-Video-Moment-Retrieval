from absl import app
from absl import flags
import json
import numpy as np
import torch
import torchvision

def sample(query_id, query_gt, query_label, query_loc, all_ids, all_gts,
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
    chosen_label = all_labels[idx]
    chosen_loc = all_locs[idx]

    return chosen_id, chosen_gt, chosen_label, chosen_loc


def get_all_data(data):

    videos = [[] for _ in range(4)]
    for i in data:
        videos[0].append(i['id'])
        videos[1].append(i['groundtruth'])
        videos[2].append(i['label'])
        videos[3].append(i['location'])
    all_ids = videos[0]
    all_gts = torch.Tensor(videos[1])
    all_labels = torch.Tensor(videos[2])
    all_locs = torch.Tensor(videos[3])
    return all_ids, all_gts, all_labels, all_locs

def get_data():
    query_reference = []
    with open('data/val.json', 'r') as f:
        data = json.load(f)
    for i in data:
        query_0 = i['id']
        query_1 = i['groundtruth']
        query_2 = i['label']
        query_3 = i['location']
        all_ids, all_gts, all_labels, all_locs = get_all_data(data)
        chosen_id, chosen_gt, chosen_label, chosen_location = sample(query_0, query_1, query_2, query_3, all_ids,
                                                                     all_gts, all_labels, all_locs)
        chosen_gt = chosen_gt.numpy().tolist()
        chosen_label = chosen_label.numpy().tolist()
        chosen_location = chosen_location.numpy().tolist()
        video = dict()
        # 创建一个新的字典
        video['q_id'] = query_0
        video['q_groundtruth'] = query_1
        video['q_label'] = query_2
        video['q_location'] = query_3
        video['p_id'] = chosen_id
        video['p_groundtruth'] = chosen_gt
        video['p_label'] = chosen_label
        video['p_location'] = chosen_location
        query_reference.append(video)
    return query_reference

def main(_):
    test_list = get_data()
    with open('data/test_list.json', 'w') as f:
      json.dump(test_list, f)

if __name__ == '__main__':
  app.run(main)
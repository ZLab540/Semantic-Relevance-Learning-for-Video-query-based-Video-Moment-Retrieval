# coding:utf-8
from __future__ import print_function
import torch.nn.functional as F
import os
import argparse
from utils import *
from dataloader_activitynet import Activitynet_Train_dataset, Activitynet_Test_dataset
from model1 import A2C
import random
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (8.0, 4.0)

from optparse import OptionParser

parser = argparse.ArgumentParser(description='Video Grounding of PyTorch')
parser.add_argument('--model', type=str, default='A2C', help='model type')
parser.add_argument('--dataset', type=str, default='Activitynet', help='dataset type')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--num_steps', type=int, default=10, help='number of forward steps in A2C (default: 10)')
parser.add_argument('--gamma', type=float, default=0.3, help='discount factor for rewards (default: 0.99)')
parser.add_argument('--entropy-coef', type=float, default=0.1, help='entropy term coefficient (default: 0.01)')
opt = parser.parse_args()
path = os.path.join(opt.dataset + '_' + opt.model)  # Activitynet_A2C

if not os.path.exists(path):
    os.makedirs(path)

train_dataset = Activitynet_Train_dataset()
test_dataset = Activitynet_Test_dataset()

all_number = len(test_dataset)
num_train_batches = int(len(train_dataset) / opt.batch_size)
print("num_train_batches:", num_train_batches)

trainloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# Model
if opt.model == 'A2C':
    net = A2C().to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


setup_seed(0)
best_R1_IOU9 = 0
best_R1_IOU8 = 0
best_R1_IOU7 = 0
best_R1_IOU6 = 0
best_R1_IOU5 = 0
best_R5_IOU7 = 0
best_R5_IOU5 = 0

best_R1_IOU9_epoch = 0
best_R1_IOU8_epoch = 0
best_R1_IOU7_epoch = 0
best_R1_IOU6_epoch = 0
best_R1_IOU5_epoch = 0
best_R5_IOU7_epoch = 0
best_R5_IOU5_epoch = 0


def determine_range_test(action, current_offset, ten_len, length):
    abnormal_done = False
    update_offset = np.zeros(2, dtype=np.float32)
    update_offset_norm = np.zeros(2, dtype=np.float32)
    current_offset_start_batch = np.zeros(1, dtype=np.int32)
    current_offset_end_batch = np.zeros(1, dtype=np.int32)
    current_offset_start = int(current_offset[0][0])
    current_offset_end = int(current_offset[0][1])

    ten_len = int(ten_len)

    if action == 0:
        current_offset_start = current_offset_start + ten_len
        current_offset_end = current_offset_end + ten_len
    elif action == 1:
        current_offset_start = current_offset_start - ten_len
        current_offset_end = current_offset_end - ten_len
    elif action == 2:
        current_offset_start = current_offset_start + ten_len
    elif action == 3:
        current_offset_start = current_offset_start - ten_len
    elif action == 4:
        current_offset_end = current_offset_end + ten_len
    elif action == 5:
        current_offset_end = current_offset_end - ten_len

    if current_offset_start < 0:
        current_offset_start = 0
        if current_offset_end < 0:
            abnormal_done = True

    if current_offset_end > length - 1:
        current_offset_end = length - 1
        if current_offset_start > length - 1:
            abnormal_done = True

    if current_offset_end <= current_offset_start:
        abnormal_done = True

    current_offset_start_batch[0] = current_offset_start
    current_offset_end_batch[0] = current_offset_end

    current_offset_start_norm = current_offset_start / float(length - 1)
    current_offset_end_norm = current_offset_end / float(length - 1)

    update_offset_norm[0] = current_offset_start_norm
    update_offset_norm[1] = current_offset_end_norm

    update_offset[0] = current_offset_start
    update_offset[1] = current_offset_end

    update_offset = torch.from_numpy(update_offset)
    update_offset = update_offset.unsqueeze(0).to(device)
    # 对数据维度进行扩充，'0'为行方向扩充

    update_offset_norm = torch.from_numpy(update_offset_norm)
    update_offset_norm = update_offset_norm.unsqueeze(0).to(device)

    return current_offset_start_batch, current_offset_end_batch, update_offset, update_offset_norm, abnormal_done


def determine_range(action, current_offset, ten_len, length):
    batch_size = action.size
    current_offset_start_batch = np.zeros(batch_size, dtype=np.int32)
    current_offset_end_batch = np.zeros(batch_size, dtype=np.int32)
    abnormal_done_batch = torch.ones(batch_size)
    update_offset = torch.zeros(batch_size, 2)
    update_offset_norm = torch.zeros(batch_size, 2)

    for i in range(batch_size):
        abnormal_done = False

        current_offset_start = int(current_offset[i][0])
        current_offset_end = int(current_offset[i][1])

        ten_len_index = int(ten_len[i])
        len_index = length[i]
        action_index = action[i]

        if current_offset_end < 0 or current_offset_start > len_index - 1 or current_offset_end <= current_offset_start or action_index == 6:
            abnormal_done = True
        else:
            if action_index == 0:
                current_offset_start = current_offset_start + ten_len_index
                current_offset_end = current_offset_end + ten_len_index
            elif action_index == 1:
                current_offset_start = current_offset_start - ten_len_index
                current_offset_end = current_offset_end - ten_len_index
            elif action_index == 2:
                current_offset_start = current_offset_start + ten_len_index
            elif action_index == 3:
                current_offset_start = current_offset_start - ten_len_index
            elif action_index == 4:
                current_offset_end = current_offset_end + ten_len_index
            elif action_index == 5:
                current_offset_end = current_offset_end - ten_len_index
            else:
                abnormal_done = True  # stop

            if current_offset_start < 0:
                current_offset_start = 0
                if current_offset_end < 0:
                    abnormal_done = True

            if current_offset_end > len_index - 1:
                current_offset_end = len_index - 1
                if current_offset_start > len_index - 1:
                    abnormal_done = True

            if current_offset_end <= current_offset_start:
                abnormal_done = True

        current_offset_start_batch[i] = current_offset_start
        current_offset_end_batch[i] = current_offset_end

        current_offset_start_norm = current_offset_start / float(len_index - 1)
        current_offset_end_norm = current_offset_end / float(len_index - 1)

        update_offset_norm[i][0] = current_offset_start_norm
        update_offset_norm[i][1] = current_offset_end_norm

        update_offset[i][0] = current_offset_start
        update_offset[i][1] = current_offset_end

        update_offset = update_offset.to(device)
        update_offset_norm = update_offset_norm.to(device)

        if abnormal_done == True:
            abnormal_done_batch[i] = 0

    return current_offset_start_batch, current_offset_end_batch, update_offset, update_offset_norm, abnormal_done_batch


# Training
def train(epoch):
    net.train()
    train_loss = 0
    policy_loss_epoch = []  # list
    value_loss_epoch = []  # list
    total_rewards_epoch = []  # list

    for batch_idx, (original_feat, global_feature, initial_feature, query_feature, offset_norm, initial_offset,
                    initial_offset_norm, ten_len, length) in enumerate(trainloader):
        original_feat = original_feat.to(device)  # 64*500*500
        global_feature = global_feature.to(device)  # 64*500
        initial_feature = initial_feature.to(device)  # 64*500
        query_feature = query_feature.to(device)  # 64*500
        offset_norm = offset_norm.to(device)  # 64*2
        initial_offset = initial_offset.to(device)  # 64*2
        initial_offset_norm = initial_offset_norm.to(device)  # 64*2
        ten_len = ten_len.to(device)  # 64 torch.float32
        length = length.to(device)  # 64 torch.float32

        batch_size = global_feature.size()[0]  # 64
        entropies = torch.zeros(opt.num_steps, batch_size)  # 10*64
        values = torch.zeros(opt.num_steps, batch_size)  # 10*64
        log_probs = torch.zeros(opt.num_steps, batch_size)  # 10*64
        rewards = torch.zeros(opt.num_steps, batch_size)  # 10*64
        Previous_IoUs = torch.zeros(opt.num_steps, batch_size)  # 10*64
        Predict_IoUs = torch.zeros(opt.num_steps, batch_size)  # 10*64
        locations = torch.zeros(opt.num_steps, batch_size, 2)  # 10*64*2
        mask = torch.zeros(opt.num_steps, batch_size)  # 10*64

        # network forward
        for step in range(opt.num_steps):

            if step == 0:
                hidden_state = torch.zeros(batch_size, 512).to(device)  # 64*512
                current_feature = initial_feature  # 64*500
                current_offset = initial_offset  # 64*2
                current_offset_norm = initial_offset_norm  # 64*2

            hidden_state, logit, value, tIoU, location = net(global_feature, current_feature, query_feature,
                                                             current_offset_norm, hidden_state)
            # 64*512 64*7 64*1 64*1 64*2
            prob = F.softmax(logit, dim=1)  # 64*7
            log_prob = F.log_softmax(logit, dim=1)  # 64*7
            entropy = -(log_prob * prob).sum(1)  # 64
            entropies[step, :] = entropy  # 10*64

            # print(prob)
            action = prob.multinomial(num_samples=1).data  # 64*1
            log_prob = log_prob.gather(1, action)  # 64*1
            action = action.cpu().numpy()[:, 0]  # ndarray 64

            current_offset_start_batch, current_offset_end_batch, current_offset, current_offset_norm, abnormal_done_batch = determine_range(
                action, current_offset, ten_len, length)

            if step == 0:
                Previou_IoU = calculate_RL_IoU_batch(initial_offset, offset_norm)  # 64*1
            else:
                Previou_IoU = current_IoU

            Previous_IoUs[step, :] = Previou_IoU
            mask[step, :] = abnormal_done_batch

            current_IoU = calculate_RL_IoU_batch(current_offset, offset_norm)

            current_feature = torch.zeros_like(initial_feature).to(device)  # 64*500

            for i in range(batch_size):
                abnormal = abnormal_done_batch[i]
                if abnormal == 1:
                    current_feature_med = original_feat[i][
                                          (current_offset_start_batch[i]):(current_offset_end_batch[i] + 1)]
                    current_feature_med = torch.mean(current_feature_med, dim=0)
                    current_feature[i] = current_feature_med

            reward = calculate_reward_batch_withstop(Previou_IoU, current_IoU, step + 1)  # 64
            values[step, :] = value.squeeze(1)  # 64*1 → 64
            log_probs[step, :] = log_prob.squeeze(1)  # 64*1 → 64
            rewards[step, :] = reward
            locations[step, :] = location
            Predict_IoUs[step, :] = tIoU.squeeze(1)  # 64*1 → 64

        total_rewards_epoch.append(rewards.sum())

        policy_loss = 0
        value_loss = 0
        idx = 0
        for j in range(batch_size):
            mask_one = mask[:, j]  # 10
            index = opt.num_steps  # index为走的总步数（最大步数/采取stop动作/出现异常情况）
            # 到达最大步数前停止
            for i in range(opt.num_steps):
                if mask_one[i] == 0:
                    index = i + 1
                    break

            for k in reversed(range(index)):
                if k == index - 1:
                    R = opt.gamma * values[k][j] + rewards[k][j]
                else:
                    R = opt.gamma * R + rewards[k][j]

                advantage = R - values[k][j]

                value_loss = value_loss + advantage.pow(2)
                policy_loss = policy_loss - log_probs[k][j] * advantage - opt.entropy_coef * entropies[k][j]
                idx += 1

        policy_loss /= idx
        value_loss /= idx

        policy_loss_epoch.append(policy_loss.item())
        value_loss_epoch.append(value_loss.item())

        # iou loss
        iou_loss = 0
        iou_id = 0
        mask_1 = np.zeros_like(Previous_IoUs)  # ndarray (10,64)
        for i in range(Previous_IoUs.size()[0]):  # 10
            for j in range(Previous_IoUs[i].size()[0]):  # 64
                iou_id += 1
                iou_loss += torch.abs(Previous_IoUs[i, j] - Predict_IoUs[i, j])
                mask_1[i, j] = Previous_IoUs[i, j] > 0.4

        iou_loss /= iou_id

        # loc loss
        loc_loss = 0
        loc_id = 0
        num = mask_1.shape[0]

        for i in range(num):  # 10
            for j in range(mask_1[i].size):  # 64
                if mask_1[i, j] == 1:
                    loc_loss += (torch.abs(offset_norm[j][0].cpu() - locations[i][j][0]) + torch.abs(
                        offset_norm[j][1].cpu() - locations[i][j][1]))
                    loc_id += 1

        if loc_id > 0:
            loc_loss /= loc_id
        else:
            loc_loss = 0

        optimizer.zero_grad()
        (policy_loss + value_loss + iou_loss + loc_loss).backward(retain_graph=True)
        optimizer.step()

        print("Train Epoch: %d | Index: %d | policy loss: %f" % (epoch, batch_idx + 1, policy_loss.data.item()))
        print("Train Epoch: %d | Index: %d | value_loss: %f" % (epoch, batch_idx + 1, value_loss.item()))

        print("Train Epoch: %d | Index: %d | iou_loss: %f" % (epoch, batch_idx + 1, iou_loss.item()))
        if loc_loss > 0:
            print("Train Epoch: %d | Index: %d | location_loss: %f" % (epoch, batch_idx + 1, loc_loss.item()))

    ave_policy_loss = np.mean(policy_loss_epoch)
    ave_policy_loss_all.append(ave_policy_loss)
    print("Average Policy Loss for Train Epoch %d : %f" % (epoch, ave_policy_loss))

    ave_value_loss = np.mean(value_loss_epoch)
    ave_value_loss_all.append(ave_value_loss)
    print("Average Value Loss for Train Epoch %d : %f" % (epoch, ave_value_loss))

    ave_total_rewards_epoch = np.mean(total_rewards_epoch)
    ave_total_rewards_all.append(ave_total_rewards_epoch)
    print("Average Total reward for Train Epoch %d: %f" % (epoch, ave_total_rewards_epoch))

    with open(path + "/iteration_ave_reward.pkl", "wb") as file:
        pickle.dump(ave_total_rewards_all, file)
    # plot the val loss vs epoch and save to disk:
    num_total = len(ave_total_rewards_all)
    x = np.arange(1, num_total + 1)
    plt.figure(1)
    plt.plot(x, ave_total_rewards_all, "r-")
    plt.ylabel("Rewards")
    plt.xlabel("Iteration")
    plt.title("Average Reward iteration")
    plt.xticks(fontsize=8)
    plt.savefig(path + "/iteration_ave_reward.png")
    plt.close(1)

    with open(path + "/iteration_ave_policy_loss.pkl", "wb") as file:
        pickle.dump(ave_policy_loss_all, file)
    # plot the val loss vs epoch and save to disk:
    plt.figure(1)
    plt.plot(x, ave_policy_loss_all, "r-")
    plt.ylabel("Loss")
    plt.xlabel("Iteration")
    plt.title("Average Policy Loss iteration")
    plt.xticks(fontsize=8)
    plt.savefig(path + "/iteration_ave_policy_loss.png")
    plt.close(1)

    with open(path + "/iteration_ave_value_loss.pkl", "wb") as file:
        pickle.dump(ave_value_loss_all, file)
    # plot the val loss vs epoch and save to disk:
    plt.figure(1)
    plt.plot(x, ave_value_loss_all, "r-")
    plt.ylabel("Loss")
    plt.xlabel("Iteration")
    plt.title("Average Value Loss iteration")
    plt.xticks(fontsize=8)
    plt.savefig(path + "/iteration_ave_value_loss.png")
    plt.close(1)


def test(epoch):
    global best_R1_IOU9
    global best_R1_IOU8
    global best_R1_IOU7
    global best_R1_IOU6
    global best_R1_IOU5
    global best_R1_IOU9_epoch
    global best_R1_IOU8_epoch
    global best_R1_IOU7_epoch
    global best_R1_IOU6_epoch
    global best_R1_IOU5_epoch
    global all_number

    net.eval()

    IoU_thresh = [0.5, 0.6, 0.7, 0.8, 0.9]
    all_correct_num_1 = [0.0] * 5
    all_retrievd = 0.0

    for idx, (
    original_feat, global_feature, initial_feature, query_feature, offset, initial_offset, initial_offset_norm, ten_len,
    length) in enumerate(testloader):

        original_feat = original_feat.to(device)  # 1*500*500
        global_feature = global_feature.to(device)  # 1*500
        initial_feature = initial_feature.to(device)  # 1*500
        query_feature = query_feature.to(device)  # 1*500
        offset = offset.to(device)  # 1*2
        initial_offset = initial_offset.to(device)  # 1*2
        initial_offset_norm = initial_offset_norm.to(device)  # 1*2
        ten_len = ten_len.to(device)  # 1
        length = length.to(device)  # 1

        query_video_reg_mat = np.zeros(2)  # ndarray(2)

        # network forward
        for step in range(opt.num_steps):
            if step == 0:
                hidden_state = torch.zeros(1, 512).to(device)  # 1*512
                current_feature = initial_feature  # 1*500
                current_offset = initial_offset  # 1*2
                current_offset_norm = initial_offset_norm  # 1*2
            hidden_state, logit, value, tIoU, location = net(global_feature, current_feature, query_feature,
                                                             current_offset_norm, hidden_state)
            # 1*512 1*7 1*1 1*1 1*2
            prob = F.softmax(logit, dim=1)  # 1*7
            action = prob.max(1, keepdim=True)[1].data.cpu().numpy()[0, 0]  # int64

            current_offset_start, current_offset_end, current_offset, current_offset_norm, abnormal_done = determine_range_test(
                action, current_offset, ten_len, length)
            # int int 1*2 1*2 bool

            if not abnormal_done:
                current_feature = original_feat[0][(current_offset_start[0]):(current_offset_end[0] + 1)]
                current_feature = torch.mean(current_feature, dim=0)
                current_feature = current_feature.unsqueeze(0).to(device)

            if action == 6 or abnormal_done == True:
                break
        query_video_reg_mat[0] = current_offset_start
        query_video_reg_mat[1] = current_offset_end

        # calculate Recall@m,IOU=n
        for k in range(len(IoU_thresh)):
            IoU = IoU_thresh[k]
            correct_num_1 = compute_IoU_recall_top_n_forreg_rl(1, IoU, query_video_reg_mat, offset)
            all_correct_num_1[k] += correct_num_1

        all_retrievd += 1

    for k in range(len(IoU_thresh)):
        print(" IoU=" + str(IoU_thresh[k]) + ", R@1: " + str(all_correct_num_1[k] / all_retrievd))
        test_result_output.write("Epoch " + str(epoch) + ": IoU=" + str(IoU_thresh[k]) + ", R@1: " + str(
            all_correct_num_1[k] / all_retrievd) + "\n")

    R1_IOU9 = all_correct_num_1[4] / all_retrievd
    R1_IOU8 = all_correct_num_1[3] / all_retrievd
    R1_IOU7 = all_correct_num_1[2] / all_retrievd
    R1_IOU6 = all_correct_num_1[1] / all_retrievd
    R1_IOU5 = all_correct_num_1[0] / all_retrievd

    R1_IOU9_all.append(R1_IOU9)
    print("R1_IOU9 for Train Epoch %d : %f" % (epoch, R1_IOU9))

    R1_IOU8_all.append(R1_IOU8)
    print("R1_IOU8 for Train Epoch %d : %f" % (epoch, R1_IOU8))

    R1_IOU7_all.append(R1_IOU7)
    print("R1_IOU7 for Train Epoch %d : %f" % (epoch, R1_IOU7))

    R1_IOU6_all.append(R1_IOU6)
    print("R1_IOU6 for Train Epoch %d : %f" % (epoch, R1_IOU6))

    R1_IOU5_all.append(R1_IOU5)
    print("R1_IOU5 for Train Epoch %d : %f" % (epoch, R1_IOU5))

    with open(path + "/R1_IOU7_all.pkl", "wb") as file:
        pickle.dump(R1_IOU7_all, file)
    # plot the val loss vs epoch and save to disk:
    x = np.arange(1, len(R1_IOU7_all) + 1)
    plt.figure(1)
    plt.plot(x, R1_IOU7_all, "r-")
    plt.ylabel("Recall")
    plt.xlabel("epoch")
    plt.title("R1_IOU7_all")
    plt.xticks(fontsize=8)
    plt.savefig(path + "/R1_IOU7_all.png")
    plt.close(1)

    with open(path + "/R1_IOU5_all.pkl", "wb") as file:
        pickle.dump(R1_IOU5_all, file)
    # plot the val loss vs epoch and save to disk:
    plt.figure(1)
    plt.plot(x, R1_IOU5_all, "r-")
    plt.ylabel("Recall")
    plt.xlabel("epoch")
    plt.title("R1_IOU5_all")
    plt.xticks(fontsize=8)
    plt.savefig(path + "/R1_IOU5_all.png")
    plt.close(1)

    if R1_IOU9 > best_R1_IOU9:
        print("best_R1_IOU9: %0.3f" % R1_IOU9)
        state = {
            'net': net.state_dict(),
            'best_R1_IOU9': R1_IOU9,
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path, 'best_R1_IOU9_model.t7'))
        best_R1_IOU9 = R1_IOU9
        best_R1_IOU9_epoch = epoch

    if R1_IOU8 > best_R1_IOU8:
        print("best_R1_IOU8: %0.3f" % R1_IOU8)
        state = {
            'net': net.state_dict(),
            'best_R1_IOU8': R1_IOU8,
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path, 'best_R1_IOU8_model.t7'))
        best_R1_IOU8 = R1_IOU8
        best_R1_IOU8_epoch = epoch

    if R1_IOU7 > best_R1_IOU7:
        print("best_R1_IOU7: %0.3f" % R1_IOU7)
        state = {
            'net': net.state_dict(),
            'best_R1_IOU7': R1_IOU7,
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path, 'best_R1_IOU7_model.t7'))
        best_R1_IOU7 = R1_IOU7
        best_R1_IOU7_epoch = epoch

    if R1_IOU6 > best_R1_IOU6:
        print("best_R1_IOU6: %0.3f" % R1_IOU6)
        state = {
            'net': net.state_dict(),
            'best_R1_IOU6': R1_IOU6,
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path, 'best_R1_IOU6_model.t7'))
        best_R1_IOU6 = R1_IOU6
        best_R1_IOU6_epoch = epoch

    if R1_IOU5 > best_R1_IOU5:
        print("best_R1_IOU5: %0.3f" % R1_IOU5)
        state = {
            'net': net.state_dict(),
            'best_R1_IOU5': R1_IOU5,
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path, 'best_R1_IOU5_model.t7'))
        best_R1_IOU5 = R1_IOU5
        best_R1_IOU5_epoch = epoch


if __name__ == '__main__':
    start_epoch = 0
    total_epoch = 100
    ave_policy_loss_all = []
    ave_value_loss_all = []
    ave_total_rewards_all = []

    R1_IOU9_all = []
    R1_IOU8_all = []
    R1_IOU7_all = []
    R1_IOU6_all = []
    R1_IOU5_all = []

    test_result_output = open(os.path.join(path, "A2C_results.txt"), "w")
    for epoch in range(start_epoch, total_epoch):
        train(epoch)
        test(epoch)

print("best_R1_IOU9: %0.3f in epoch: %d " % (best_R1_IOU9, best_R1_IOU9_epoch))
print("best_R1_IOU8: %0.3f in epoch: %d " % (best_R1_IOU8, best_R1_IOU8_epoch))
print("best_R1_IOU7: %0.3f in epoch: %d " % (best_R1_IOU7, best_R1_IOU7_epoch))
print("best_R1_IOU6: %0.3f in epoch: %d " % (best_R1_IOU6, best_R1_IOU6_epoch))
print("best_R1_IOU5: %0.3f in epoch: %d " % (best_R1_IOU5, best_R1_IOU5_epoch))

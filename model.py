import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True).expand_as(out))

    return out


def weights_init(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, mean=0, std=0.01)
        # N(mean,std)
        m.bias.data.fill_(0)

    elif classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data)
        m.bias.data.fill_(0)


class A2C(nn.Module):
    def __init__(self):
        super(A2C, self).__init__()
        self.visual_feature_dim = 500
        self.global_fc = nn.Linear(self.visual_feature_dim, 512)
        self.local_fc = nn.Linear(self.visual_feature_dim, 512)
        self.loc_fc = nn.Linear(2, 512)
        self.query_fc = nn.Linear(self.visual_feature_dim, 512)
        self.state_fc = nn.Linear(512 + 512 + 512, 512)

        self.gru = nn.GRUCell(512, 512)
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, 7)

        self.tiou_fc = nn.Linear(512, 1)
        self.location_fc = nn.Linear(512, 2)

        # Initializing weights
        self.apply(weights_init)

        # actor
        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)

        # critic
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

    def forward(self, global_feature, local_feature, query_feature, location_feature, hidden_state):

        global_feature = self.global_fc(global_feature)
        global_feature_norm = F.normalize(global_feature, p=2, dim=1)
        global_feature_norm = F.relu(global_feature_norm)

        local_feature = self.local_fc(local_feature)
        local_feature_norm = F.normalize(local_feature, p=2, dim=1)
        local_feature_norm = F.relu(local_feature_norm)

        query_feature = self.query_fc(query_feature)
        query_feature_norm = torch.sigmoid(query_feature)

        location_feature = self.loc_fc(location_feature)
        location_feature_norm = F.normalize(location_feature, p=2, dim=1)
        location_feature_norm = F.relu(location_feature_norm)

        # local gate-attention
        assert local_feature_norm.size() == query_feature_norm.size()
        local_attention_feature = local_feature_norm * query_feature_norm

        # global gate-attention
        assert global_feature_norm.size() == query_feature_norm.size()
        global_attention_feature = global_feature_norm * query_feature_norm

        # location gate-attention
        assert location_feature_norm.size() == query_feature_norm.size()
        location_attention_feature = location_feature_norm * query_feature_norm

        state_feature = torch.cat([global_attention_feature, local_attention_feature, location_attention_feature], 1)

        state_feature = self.state_fc(state_feature)
        state_feature = F.relu(state_feature)

        hidden_state = self.gru(state_feature, hidden_state)

        value = self.critic_linear(hidden_state)
        actions = self.actor_linear(hidden_state)

        tIoU = self.tiou_fc(state_feature)
        location = self.location_fc(state_feature)

        return hidden_state, actions, value, tIoU, location

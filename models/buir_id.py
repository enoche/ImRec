# -*- coding: utf-8 -*-
# @Time   : 2021/05/17
# @Author : Zhou xin
# @Email  : enoche.chow@gmail.com

r"""
BUIR_ID
################################################
Bootstrapping User and Item Representations for One-Class Collaborative Filtering, SIGIR21
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common.abstract_recommender import GeneralRecommender


class BUIR_ID(GeneralRecommender):
    def __init__(self, config, dataset):
        super(BUIR_ID, self).__init__(config, dataset)
        self.user_count = self.n_users
        self.item_count = self.n_items
        self.latent_size = config['embedding_size']
        self.momentum = config['momentum']

        self.user_online = nn.Embedding(self.user_count, self.latent_size)
        self.user_target = nn.Embedding(self.user_count, self.latent_size)
        self.item_online = nn.Embedding(self.item_count, self.latent_size)
        self.item_target = nn.Embedding(self.item_count, self.latent_size)

        self.predictor = nn.Linear(self.latent_size, self.latent_size)

        self._init_model()
        self._init_target()

    def _init_model(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.normal_(m.bias.data)

            if isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight.data)

    def _init_target(self):
        for param_o, param_t in zip(self.user_online.parameters(), self.user_target.parameters()):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False

        for param_o, param_t in zip(self.item_online.parameters(), self.item_target.parameters()):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False

    def _update_target(self):
        for param_o, param_t in zip(self.user_online.parameters(), self.user_target.parameters()):
            param_t.data = param_t.data * self.momentum + param_o.data * (1. - self.momentum)

        for param_o, param_t in zip(self.item_online.parameters(), self.item_target.parameters()):
            param_t.data = param_t.data * self.momentum + param_o.data * (1. - self.momentum)

    def forward(self, users, items):
        self._update_target()

        u_online = self.predictor(self.user_online(users))
        u_target = self.user_target(users)
        i_online = self.predictor(self.item_online(items))
        i_target = self.item_target(items)

        return u_online, u_target, i_online, i_target

    @torch.no_grad()
    def get_embedding(self):
        u_online = self.user_online.weight
        i_online = self.item_online.weight
        return self.predictor(u_online), u_online, self.predictor(i_online), i_online

    def calculate_loss(self, interaction):
        users = interaction[0]
        items = interaction[1]
        u_online, u_target, i_online, i_target = self.forward(users, items)

        u_online = F.normalize(u_online, dim=-1)
        u_target = F.normalize(u_target, dim=-1)
        i_online = F.normalize(i_online, dim=-1)
        i_target = F.normalize(i_target, dim=-1)

        # Euclidean distance between normalized vectors can be replaced with their negative inner product
        loss_ui = 2 - 2 * (u_online * i_target).sum(dim=-1)
        loss_iu = 2 - 2 * (i_online * u_target).sum(dim=-1)

        return (loss_ui + loss_iu).mean()

    def full_sort_predict(self, interaction):
        user = interaction[0]
        u_online, u_target, i_online, i_target = self.get_embedding()
        score_mat_ui = torch.matmul(u_online[user], i_target.transpose(0, 1))
        score_mat_iu = torch.matmul(u_target[user], i_online.transpose(0, 1))
        scores = score_mat_ui + score_mat_iu

        return scores


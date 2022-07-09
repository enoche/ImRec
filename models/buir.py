# -*- coding: utf-8 -*-
# @Time   : 2021/05/17
# @Author : Zhou xin
# @Email  : enoche.chow@gmail.com

r"""
BUIR with LigntGCN encoder.
################################################
Bootstrapping User and Item Representations for One-Class Collaborative Filtering, SIGIR21
"""

import scipy.sparse as sp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common.abstract_recommender import GeneralRecommender
from models.common.encoders import LightGCN_Encoder


class BUIR(GeneralRecommender):
    def __init__(self, config, dataset):
        super(BUIR, self).__init__(config, dataset)
        self.user_count = self.n_users
        self.item_count = self.n_items
        self.latent_size = config['embedding_size']
        self.momentum = config['momentum']

        self.online_encoder = LightGCN_Encoder(config, dataset)
        self.target_encoder = LightGCN_Encoder(config, dataset)

        self.predictor = nn.Linear(self.latent_size, self.latent_size)

        self._init_target()

    def _init_target(self):
        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False

    def _update_target(self):
        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data = param_t.data * self.momentum + param_o.data * (1. - self.momentum)

    def forward(self, inputs):
        self._update_target()

        u_online, i_online = self.online_encoder(inputs)
        u_target, i_target = self.target_encoder(inputs)
        return self.predictor(u_online), u_target, self.predictor(i_online), i_target

    @torch.no_grad()
    def get_embedding(self):
        u_online, i_online = self.online_encoder.get_embedding()
        return self.predictor(u_online), u_online, self.predictor(i_online), i_online

    def calculate_loss(self, interaction):
        u_online, u_target, i_online, i_target = self.forward(interaction)

        u_online = F.normalize(u_online, dim=-1)
        u_target = F.normalize(u_target, dim=-1)
        i_online = F.normalize(i_online, dim=-1)
        i_target = F.normalize(i_target, dim=-1)

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

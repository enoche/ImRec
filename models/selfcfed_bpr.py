# -*- coding: utf-8 -*-
# @Time   : 2021/05/17
# @Author : Zhou xin
# @Email  : enoche.chow@gmail.com
#
# updated: 2022/05/07

r"""
################################################
Self-supervised CF
Using current implementation of LightGCN
SELFCF_{ed}: embedding dropout
"""

import scipy.sparse as sp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common.init import xavier_normal_initialization
from models.bpr import BPR
from models.common.loss import BPRLoss, EmbLoss, L2Loss
from models.common.abstract_recommender import GeneralRecommender

class SELFCFED_BPR(GeneralRecommender):
    def __init__(self, config, dataset):
        super(SELFCFED_BPR, self).__init__(config, dataset)
        self.user_count = self.n_users
        self.item_count = self.n_items
        self.latent_size = config['embedding_size']
        self.dropout = config['dropout']
        self.reg_weight = config['reg_weight']
        self.linear_weight = config['linear_weight']

        self.online_encoder = BPR(config, dataset)
        self.predictor = nn.Linear(self.latent_size, self.latent_size)
        self.reg_loss = L2Loss()

    def forward(self):
        u_online, i_online = self.online_encoder()
        with torch.no_grad():
            u_target, i_target = self.online_encoder()
            u_target.detach()
            i_target.detach()
            u_target = F.dropout(u_target, self.dropout)
            i_target = F.dropout(i_target, self.dropout)

        return u_online, u_target, i_online, i_target

    @torch.no_grad()
    def get_embedding(self):
        u_online, i_online = self.online_encoder()
        return self.predictor(u_online), u_online, self.predictor(i_online), i_online

    def loss_fn(self, p, z):  # negative cosine similarity
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()

    def calculate_loss(self, interaction):
        u_all_online, u_all_target, i_all_online, i_all_target = self.forward()

        users, items = interaction[0], interaction[1]
        u_online = u_all_online[users, :]
        i_online = i_all_online[items, :]
        u_target = u_all_target[users, :]
        i_target = i_all_target[items, :]

        reg_loss = self.reg_loss(u_online, i_online)

        u_online, i_online = self.predictor(u_online), self.predictor(i_online)

        loss_ui = self.loss_fn(u_online, i_target)/2
        loss_iu = self.loss_fn(i_online, u_target)/2

        linear_loss = .0
        for param in self.predictor.parameters():
            linear_loss += torch.norm(param, 1) ** 2

        return loss_ui + loss_iu + self.reg_weight * reg_loss + self.linear_weight * linear_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        u_online, u_target, i_online, i_target = self.get_embedding()
        score_mat_ui = torch.matmul(u_online[user], i_target.transpose(0, 1))
        score_mat_iu = torch.matmul(u_target[user], i_online.transpose(0, 1))
        scores = score_mat_ui + score_mat_iu

        return scores

# -*- coding: utf-8 -*-
# @Time   : 2021/05/17
# @Author : Zhou xin
# @Email  : enoche.chow@gmail.com

r"""
################################################
Self-supervised CF

Using the same implementation of LightGCN in BUIR
With regularization


SELFCF_{ep}: edge pruning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common.encoders import LightGCN_Encoder
from models.common.abstract_recommender import GeneralRecommender
from models.common.loss import L2Loss


class SELFCFEP_LGN(GeneralRecommender):
    def __init__(self, config, dataset):
        super(SELFCFEP_LGN, self).__init__(config, dataset)
        self.user_count = self.n_users
        self.item_count = self.n_items
        self.latent_size = config['embedding_size']
        self.dropout = config['dropout']
        self.reg_weight = config['reg_weight']

        self.online_encoder = LightGCN_Encoder(config, dataset)
        self.predictor = nn.Linear(self.latent_size, self.latent_size)
        self.norm_adj_matrix = self.online_encoder.sparse_norm_adj.clone().to(self.device)

        # to get all embeddings (u+i)
        self.ui_all_inputs = (torch.arange(self.user_count).to(self.device),
                              torch.arange(self.item_count).to(self.device))
        self.reg_loss = L2Loss()

    def sparse_dropout(self, x):
        kprob = 1 - self.dropout
        randx = torch.rand(x._values().size()).to(self.device)
        mask = ((randx + kprob).floor()).type(torch.bool)
        rc = x._indices()[:, mask]
        val = x._values()[mask] * (1.0 / kprob)
        return torch.sparse.FloatTensor(rc, val, x.shape).to(self.device)

    def forward(self, inputs):
        u_online, i_online = self.online_encoder(self.ui_all_inputs)
        with torch.no_grad():
            u_target, i_target = u_online.clone(), i_online.clone()
            # edge pruning
            x = self.sparse_dropout(self.norm_adj_matrix)
            all_embeddings = torch.cat([u_target, i_target], 0)
            all_embeddings = torch.sparse.mm(x, all_embeddings)
            u_target = all_embeddings[:self.user_count, :]
            i_target = all_embeddings[self.user_count:, :]
            u_target = u_target[inputs[0], :]
            i_target = i_target[inputs[1], :]

        u_online = u_online[inputs[0], :]
        i_online = i_online[inputs[1], :]
        return u_online, u_target, i_online, i_target

    @torch.no_grad()
    def get_embedding(self):
        u_online, i_online = self.online_encoder.get_embedding()
        return self.predictor(u_online), u_online, self.predictor(i_online), i_online

    def loss_fn(self, p, z):  # negative cosine similarity
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()

    def calculate_loss(self, interaction):
        u_online, u_target, i_online, i_target = self.forward(interaction)
        reg_loss = self.reg_loss(u_online, i_online)

        u_online, i_online = self.predictor(u_online), self.predictor(i_online)

        loss_ui = self.loss_fn(u_online, i_target)/2
        loss_iu = self.loss_fn(i_online, u_target)/2

        return loss_ui + loss_iu + self.reg_weight * reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        u_online, u_target, i_online, i_target = self.get_embedding()
        score_mat_ui = torch.matmul(u_online[user], i_target.transpose(0, 1))
        score_mat_iu = torch.matmul(u_target[user], i_online.transpose(0, 1))
        scores = score_mat_ui + score_mat_iu

        return scores


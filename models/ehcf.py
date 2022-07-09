# -*- coding: utf-8 -*-

r"""
EHCF
################################################

Reference:
    Chong Chen, Min Zhang, Weizhi Ma, Yongfeng Zhang, Yiqun Liu and Shaoping Ma. 2020.
    Efﬁcient Heterogeneous Collaborative Filtering without Negative Sampling for Recommendation. In AAAI'20.

Reference code:
    https://github.com/chenchongthu/EHCF

"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from models.common.abstract_recommender import GeneralRecommender
from models.common.loss import EmbLoss
import torch.nn.functional as F


class EHCF(GeneralRecommender):
    def __init__(self, config, dataset):
        super(EHCF, self).__init__(config, dataset)

        # load parameters info
        self.latent_dim = config['embedding_size']  # int type:the embedding size of lightGCN
        self.dropout_prob = 0.0
        self.weight1 = config['reg_weight']

        # define layers and loss
        self.user_embedding = nn.Parameter(self.truncated_normal(torch.empty(self.n_users, self.latent_dim)))
        self.item_embedding = nn.Parameter(self.truncated_normal(torch.empty(self.n_items, self.latent_dim)))

        self.l2loss = nn.MSELoss(reduction='sum')

        self.pos_items, self.pos_items_mask = self.get_pos_items(dataset._get_history_items_u())
        self.H_i = nn.Parameter(torch.full((self.latent_dim, 1), 0.01)).to(self.device)
        self.lambda_bilinear = [0.0, 0.0]

        # parameters initialization
        #self.apply(xavier_uniform_initialization)

    def get_pos_items(self, pos_items_per_u):
        u_ids = sorted(pos_items_per_u.keys())
        max_key = max(pos_items_per_u, key=lambda k: len(pos_items_per_u[k]))
        max_len = len(pos_items_per_u[max_key])
        items_ls, items_mask = [], []
        for u_id in u_ids:
            u_len = len(pos_items_per_u[u_id])
            items_ls.append(pos_items_per_u[u_id].tolist() + [0] * (max_len - u_len))
            items_mask.append([1.0] * u_len + [0.0] * (max_len - u_len))
        return torch.tensor(items_ls).to(self.device), torch.tensor(items_mask).to(self.device)

    def truncated_normal(self, t, mean=0.0, std=0.01):
        torch.nn.init.normal_(t, mean=mean, std=std)
        while True:
            cond = torch.logical_or(t < mean - 2 * std, t > mean + 2 * std)
            if not torch.sum(cond):
                break
            t = torch.where(cond, torch.nn.init.normal_(torch.ones(t.shape), mean=mean, std=std), t)
        return t

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        ego_embeddings = torch.cat([self.user_embedding, self.item_embedding], dim=0)
        return ego_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings()

        user_all_embeddings = all_embeddings[:self.n_users, :]
        item_all_embeddings = all_embeddings[self.n_users:, :]

        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, interaction):
        user = interaction
        # batch_size * max_items_len_per_u(max_len)
        pos_item = self.pos_items[user, :]
        pos_item_mask = self.pos_items_mask[user, :]

        user_all_embeddings, item_all_embeddings = self.forward()

        # batch_size * 1 * dim
        u_embeddings = user_all_embeddings[user, :]
        u_embeddings = F.dropout(u_embeddings, self.dropout_prob)
        # batch_size * max_len * dim
        pos_embeddings = item_all_embeddings[pos_item, :]
        pos_embeddings = torch.einsum('ab,abc->abc', pos_item_mask, pos_embeddings)
        pos_pred = torch.einsum('ac,abc->abc', u_embeddings, pos_embeddings)
        pos_pred = torch.einsum('ajk,kl->ajl', pos_pred, self.H_i)
        pos_pred = torch.reshape(pos_pred, (-1, pos_item.shape[1]))

        # Loss
        loss1 = self.weight1 * torch.sum(
            torch.sum(torch.sum(torch.einsum('ab,ac->abc', item_all_embeddings, item_all_embeddings), 0)
                      * torch.sum(torch.einsum('ab,ac->abc', u_embeddings, u_embeddings), 0)
                      * torch.matmul(self.H_i, torch.t(self.H_i)), 0), 0)
        loss1 += torch.sum((1.0 - self.weight1) * torch.square(pos_pred) - 2.0 * pos_pred)

        emb_loss0 = self.l2loss(user_all_embeddings, torch.zeros_like(user_all_embeddings)) * 0.5
        emb_loss1 = self.l2loss(item_all_embeddings, torch.zeros_like(item_all_embeddings)) * 0.5
        loss = loss1 + self.lambda_bilinear[0] * emb_loss0 + self.lambda_bilinear[1] * emb_loss1

        return loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        restore_user_e, restore_item_e = self.forward()
        u_embeddings = restore_user_e[user, :]

        # dot with all item embedding to accelerate
        #scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        dot = torch.einsum('ac,bc->abc', u_embeddings, restore_item_e)
        pre = torch.einsum('ajk,kl->ajl', dot, self.H_i)
        scores = torch.reshape(pre, [-1, self.n_items])

        return scores

# -*- coding: utf-8 -*-

r"""
UltraGCN
################################################
Reference:
    UltraGCN: Ultra Simplification of Graph Convolutional Networks for Recommendation, CIKM'21
"""

import torch
import torch.nn as nn
import numpy as np

from models.common.abstract_recommender import GeneralRecommender
from models.common.loss import BPRLoss, EmbLoss
from models.common.init import xavier_normal_initialization
import torch.nn.functional as F


class UltraGCN(GeneralRecommender):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way.

    """
    def __init__(self, config, dataset):
        super(UltraGCN, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config['embedding_dim']
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalizaton
        self.w1 = config['w1']
        self.w2 = config['w2']
        self.w3 = config['w3']
        self.w4 = config['w4']

        self.negative_weight = config['negative_weight']
        self.gamma = config['gamma']
        self.lambda_ = config['lambda']
        self.initial_weight = config['initial_weight']
        self.ii_neighbor_num = config['ii_neighbor_num']

        # define layers and loss
        self.user_embeds = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embeds = nn.Embedding(self.n_items, self.embedding_size)
        self.loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # parameters initialization
        nn.init.normal_(self.user_embeds.weight, std=self.initial_weight)
        nn.init.normal_(self.item_embeds.weight, std=self.initial_weight)

        self.constraint_mat = dict()
        self.ii_constraint_mat, self.ii_neighbor_mat = None, None
        interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        inter_dok_mat = interaction_matrix.todok()
        self.build_mats(inter_dok_mat)

    def build_mats(self, train_mat):
        items_D = np.sum(train_mat, axis=0).reshape(-1)
        users_D = np.sum(train_mat, axis=1).reshape(-1)
        beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
        beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)

        self.constraint_mat = {"beta_uD": torch.from_numpy(beta_uD).reshape(-1),
                          "beta_iD": torch.from_numpy(beta_iD).reshape(-1)}
        self.ii_neighbor_mat, self.ii_constraint_mat = self.get_ii_constraint_mat(train_mat, self.ii_neighbor_num)

    def get_ii_constraint_mat(self, train_mat, num_neighbors, ii_diagonal_zero=False):
        #print('Computing \\Omega for the item-item graph... ')
        A = train_mat.T.dot(train_mat)  # I * I
        n_items = A.shape[0]
        res_mat = torch.zeros((n_items, num_neighbors))
        res_sim_mat = torch.zeros((n_items, num_neighbors))
        if ii_diagonal_zero:
            A[range(n_items), range(n_items)] = 0
        items_D = np.sum(A, axis=0).reshape(-1)
        users_D = np.sum(A, axis=1).reshape(-1)

        beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
        beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)
        all_ii_constraint_mat = torch.from_numpy(beta_uD.dot(beta_iD))
        for i in range(n_items):
            row = all_ii_constraint_mat[i] * torch.from_numpy(A.getrow(i).toarray()[0])
            row_sims, row_idxs = torch.topk(row, num_neighbors)
            res_mat[i] = row_idxs
            res_sim_mat[i] = row_sims
            #if i % 15000 == 0:
            #    print('i-i constraint matrix {} ok'.format(i))

        #print('Computation \\Omega OK!')
        return res_mat.long(), res_sim_mat.float()

    def get_omegas(self, users, pos_items, neg_items):
        device = self.device
        if self.w2 > 0:
            pos_weight = torch.mul(self.constraint_mat['beta_uD'][users], self.constraint_mat['beta_iD'][pos_items]).to(
                device)
            pos_weight = self.w1 + self.w2 * pos_weight
        else:
            pos_weight = self.w1 * torch.ones(len(pos_items)).to(device)

        # users = (users * self.item_num).unsqueeze(0)
        if self.w4 > 0:
            neg_weight = torch.mul(torch.repeat_interleave(self.constraint_mat['beta_uD'][users], neg_items.size(1)),
                                   self.constraint_mat['beta_iD'][neg_items.flatten()]).to(device)
            neg_weight = self.w3 + self.w4 * neg_weight
        else:
            neg_weight = self.w3 * torch.ones(neg_items.size(0) * neg_items.size(1)).to(device)

        weight = torch.cat((pos_weight, neg_weight))
        return weight

    def norm_loss(self):
        loss = 0.0
        for parameter in self.parameters():
            loss += torch.sum(parameter ** 2)
        return loss / 2

    def cal_loss_L(self, users, pos_items, neg_items, omega_weight):
        device = self.device
        user_embeds = self.user_embeds(users)
        pos_embeds = self.item_embeds(pos_items)
        neg_embeds = self.item_embeds(neg_items)

        pos_scores = (user_embeds * pos_embeds).sum(dim=-1)  # batch_size
        user_embeds = user_embeds.unsqueeze(1)
        neg_scores = (user_embeds * neg_embeds).sum(dim=-1)  # batch_size * negative_num

        neg_labels = torch.zeros(neg_scores.size()).to(device)
        neg_loss = F.binary_cross_entropy_with_logits(neg_scores, neg_labels,
                                                      weight=omega_weight[len(pos_scores):].view(neg_scores.size()),
                                                      reduction='none').mean(dim=-1)

        pos_labels = torch.ones(pos_scores.size()).to(device)
        pos_loss = F.binary_cross_entropy_with_logits(pos_scores, pos_labels, weight=omega_weight[:len(pos_scores)],
                                                      reduction='none')

        loss = pos_loss + neg_loss * self.negative_weight
        return loss.sum()

    def cal_loss_I(self, users, pos_items):
        device = self.device
        neighbor_embeds = self.item_embeds(self.ii_neighbor_mat[pos_items].to(device))  # len(pos_items) * num_neighbors * dim
        sim_scores = self.ii_constraint_mat[pos_items].to(device)  # len(pos_items) * num_neighbors
        user_embeds = self.user_embeds(users).unsqueeze(1)

        loss = -sim_scores * (user_embeds * neighbor_embeds).sum(dim=-1).sigmoid().log()
        # loss = loss.sum(-1)
        return loss.sum()

    def calculate_loss(self, interaction):
        users, pos_items, neg_items = interaction[0], interaction[1], interaction[2]

        omega_weight = self.get_omegas(users, pos_items, neg_items)
        loss = self.cal_loss_L(users, pos_items, neg_items, omega_weight)
        loss += self.gamma * self.norm_loss()
        loss += self.lambda_ * self.cal_loss_I(users, pos_items)
        return loss

    def forward(self):
        pass

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_e = self.user_embeds(user)
        all_item_e = self.item_embeds.weight
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score

# -*- coding: utf-8 -*-

r"""
LightGCN
################################################

Reference:
    Interest-aware Message-Passing GCN for Recommendation, WWW'21

Reference code:
    https://github.com/liufancs/IMP_GCN
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from models.common.abstract_recommender import GeneralRecommender
from models.common.loss import BPRLoss, EmbLoss
from models.common.init import xavier_uniform_initialization
import torch.nn.functional as F


class IMP_GCN(GeneralRecommender):
    def __init__(self, config, dataset):
        super(IMP_GCN, self).__init__(config, dataset)

        self.n_fold = 20
        # load parameters info
        self.groups = config['groups']
        self.adj_type = config['adj_type']
        self.emb_dim = config['embedding_size']  # int type:the embedding size of lightGCN
        self.reg_weight = config['regs']  # float32 type: the weight decay for l2 normalizaton
        self.weight_size = config['layer_size']
        self.n_layers = len(self.weight_size)

        # load dataset info
        interaction_matrix = dataset.inter_matrix(
            form='coo').astype(np.float32)
        # generate intermediate data
        self.norm_adj = self.get_norm_adj_mat(interaction_matrix)

        # init parameters
        initializer = nn.init.xavier_uniform_
        self.embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.n_users, self.emb_dim))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.n_items, self.emb_dim)))
        })
        self.W_gc_1 = nn.Parameter(initializer(torch.empty(self.emb_dim, self.emb_dim)))
        self.b_gc_1 = nn.Parameter(initializer(torch.empty(1, self.emb_dim)))
        self.W_gc_2 = nn.Parameter(initializer(torch.empty(self.emb_dim, self.emb_dim)))
        self.b_gc_2 = nn.Parameter(initializer(torch.empty(1, self.emb_dim)))
        self.W_gc = nn.Parameter(initializer(torch.empty(self.emb_dim, self.groups)))
        self.b_gc = nn.Parameter(initializer(torch.empty(1, self.groups)))

        self.A_fold_hat = self._split_A_hat(self.norm_adj)
        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

    def pre_epoch_processing(self):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

    def get_norm_adj_mat(self, interaction_matrix):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = interaction_matrix
        inter_M_t = interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col+self.n_users),
                             [1]*inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row+self.n_users, inter_M_t.col),
                                  [1]*inter_M_t.nnz)))
        adj_mat._update(data_dict)
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()

        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat_inv)
        #print('generate pre adjacency matrix.')
        pre_adj_mat = norm_adj.tocsr()
        return pre_adj_mat

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col])#.transpose()
        indices = torch.from_numpy(indices).type(torch.LongTensor)
        data = torch.from_numpy(coo.data)
        return torch.sparse.FloatTensor(indices, data, torch.Size((coo.shape[0], coo.shape[1])))

    def _split_A_hat(self, X):
        A_fold_hat = []
        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]).to(self.device))
        return A_fold_hat

    def sparse_dense_mul(self, s, d):
        i = s._indices()
        v = s._values()
        dv = d[i[0, :], i[1, :]]  # get values from relevant entries of dense matrix
        ret_tensor = torch.sparse.FloatTensor(i, v * dv, s.size())
        return ret_tensor.to(self.device)

    def _split_A_hat_group(self, X, group_embedding):
        group_embedding = group_embedding.T
        A_fold_hat_group = []
        A_fold_hat_group_filter = []
        A_fold_hat = self.A_fold_hat

        # # split L in fold
        fold_len = (self.n_users + self.n_items) // self.n_fold
        # for i_fold in range(self.n_fold):
        #     start = i_fold * fold_len
        #     if i_fold == self.n_fold - 1:
        #         end = self.n_users + self.n_items
        #     else:
        #         end = (i_fold + 1) * fold_len
        #     A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))

        # k groups
        for k in range(0, self.groups):
            A_fold_item_filter = []
            A_fold_hat_item = []

            # n folds in per group (filter user)
            for i_fold in range(self.n_fold):
                start = i_fold * fold_len
                if i_fold == self.n_fold - 1:
                    end = self.n_users + self.n_items
                else:
                    end = (i_fold + 1) * fold_len

                temp_g = self.sparse_dense_mul(A_fold_hat[i_fold], group_embedding[k].expand(A_fold_hat[i_fold].shape))
                temp_slice = self.sparse_dense_mul(temp_g, torch.unsqueeze(group_embedding[k][start:end], dim=1).expand(temp_g.shape))
                #A_fold_hat_item.append(A_fold_hat[i_fold].__mul__(group_embedding[k]).__mul__(torch.unsqueeze(group_embedding[k][start:end], dim=1)))
                A_fold_hat_item.append(temp_slice)
                item_filter = torch.sparse.sum(A_fold_hat_item[i_fold], dim=1).to_dense()
                item_filter = torch.where(item_filter > 0., torch.ones_like(item_filter), torch.zeros_like(item_filter))
                A_fold_item_filter.append(item_filter)

            A_fold_item = torch.concat(A_fold_item_filter, dim=0)
            A_fold_hat_group_filter.append(A_fold_item)
            A_fold_hat_group.append(A_fold_hat_item)

        return A_fold_hat_group, A_fold_hat_group_filter

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.
        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        return ego_embeddings

    def forward(self):
        # _create_imp_gcn_embed in original IMP_GCN/IMP_GCN.py
        A_fold_hat = self.A_fold_hat
        ego_embeddings = self.get_ego_embeddings()
        # group users
        temp_embed = []
        for f in range(self.n_fold):
            temp_embed.append(torch.sparse.mm(A_fold_hat[f], ego_embeddings))
        user_group_embeddings_side = torch.concat(temp_embed, dim=0) + ego_embeddings

        user_group_embeddings_hidden_1 = F.leaky_relu(torch.matmul(user_group_embeddings_side, self.W_gc_1) + self.b_gc_1)
        user_group_embeddings_hidden_d1 = F.dropout(user_group_embeddings_hidden_1, 0.6)

        user_group_embeddings_sum = torch.matmul(user_group_embeddings_hidden_d1, self.W_gc) + self.b_gc
        # user 0-1
        a_top, a_top_idx = torch.topk(user_group_embeddings_sum, 1, sorted=False)
        user_group_embeddings = torch.eq(user_group_embeddings_sum, a_top).type(torch.float32)
        u_group_embeddings, i_group_embeddings = torch.split(user_group_embeddings, [self.n_users, self.n_items], 0)
        i_group_embeddings = torch.ones_like(i_group_embeddings)
        user_group_embeddings = torch.concat([u_group_embeddings, i_group_embeddings], dim=0)
        # Matrix mask
        A_fold_hat_group, A_fold_hat_group_filter = self._split_A_hat_group(self.norm_adj, user_group_embeddings)
        # embedding transformation
        all_embeddings = [ego_embeddings]
        temp_embed = []
        for f in range(self.n_fold):
            temp_embed.append(torch.sparse.mm(A_fold_hat[f], ego_embeddings))

        side_embeddings = torch.concat(temp_embed, dim=0)
        all_embeddings += [side_embeddings]

        ego_embeddings_g = []
        for g in range(0, self.groups):
            ego_embeddings_g.append(ego_embeddings)
        ego_embeddings_f = []
        for k in range(1, self.n_layers):
            for g in range(0, self.groups):
                temp_embed = []
                for f in range(self.n_fold):
                    temp_embed.append(torch.sparse.mm(A_fold_hat_group[g][f], ego_embeddings_g[g]))
                side_embeddings = torch.concat(temp_embed, dim=0)
                ego_embeddings_g[g] = ego_embeddings_g[g] + side_embeddings
                temp_embed = []
                for f in range(self.n_fold):
                    temp_embed.append(torch.sparse.mm(A_fold_hat[f], side_embeddings))
                if k == 1:
                    ego_embeddings_f.append(torch.concat(temp_embed, dim=0))
                else:
                    ego_embeddings_f[g] = torch.concat(temp_embed, dim=0)
            ego_embeddings = torch.sum(torch.stack(ego_embeddings_f, dim=0), dim=0)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, 1)
        all_embeddings = torch.sum(all_embeddings, dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings
        #return u_g_embeddings, i_g_embeddings, A_fold_hat_group_filter, user_group_embeddings_sum

    def bpr_loss(self, users, pos_items, neg_items, users_pre, pos_items_pre, neg_items_pre):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1./2*(users_pre**2).sum() + 1./2*(pos_items_pre**2).sum() + 1./2*(neg_items_pre**2).sum()
        regularizer = regularizer / self.batch_size

        maxi = F.softplus(-(pos_scores - neg_scores))
        mf_loss = torch.mean(maxi)

        emb_loss = self.reg_weight * regularizer
        reg_loss = 0.0
        return mf_loss + emb_loss + reg_loss

    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_item = interaction[1]
        neg_item = interaction[2]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user, :]
        pos_embeddings = item_all_embeddings[pos_item, :]
        neg_embeddings = item_all_embeddings[neg_item, :]
        u_embeddings_pre = self.embedding_dict['user_emb'][user, :]
        pos_embeddings_pre = self.embedding_dict['item_emb'][pos_item, :]
        neg_embeddings_pre = self.embedding_dict['item_emb'][neg_item, :]

        loss = self.bpr_loss(u_embeddings, pos_embeddings, neg_embeddings,
                             u_embeddings_pre, pos_embeddings_pre, neg_embeddings_pre)

        return loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        u_embeddings = self.restore_user_e[user, :]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores

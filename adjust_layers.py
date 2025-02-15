# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : layers.py
# Time       :2024/4/2 14:28
# Author     :Liu Ziyue
# version    :python 3.8
# Description:
"""





import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GraphConv, TopKPooling, SAGPooling, EdgePooling
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from sklearn.model_selection import KFold
from adjust_common import get_activation_function, get_conv_layer, get_pool_layer, get_global_pool_layer


import torch
import torch.nn as nn
import torch.nn.functional as F

# class GraphAttentionLayer(nn.Module):
#     """
#     Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
#     """
#     def __init__(self, in_features, out_features, dropout, Alpha, activation_name, concat=True, alpha=None):
#         super(GraphAttentionLayer, self).__init__()

#         self.dropout = dropout
#         self.in_features = in_features
#         self.out_features = out_features
#         self.Alpha = Alpha
#         self.concat = concat
#         self.activation = get_activation_function(activation_name, alpha=alpha)

#         self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
#         nn.init.xavier_uniform_(self.W.data, gain=1.414)
#         self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
#         nn.init.xavier_uniform_(self.a.data, gain=1.414)

#         self.leakyrelu = nn.LeakyReLU(self.Alpha)

#     def forward(self, x, adj):
#         Wh = torch.mm(x, self.W)
#         e = self._prepare_attentional_mechanism_input(Wh)

#         # Ensure adj and e have the same dimensions
#         num_nodes = adj.size(0)
#         if e.size(0) != num_nodes or e.size(1) != num_nodes:
#             raise ValueError(f"Size mismatch: e shape {e.shape} does not match adj shape {adj.shape}")

#         # Adjust e and adj to ensure they have the same shape
#         e = e[:num_nodes, :num_nodes]
#         zero_vec = -9e15 * torch.ones_like(e)
#         attention = torch.where(adj > 0, e, zero_vec)
#         attention = F.softmax(attention, dim=1)
#         attention = F.dropout(attention, self.dropout, training=self.training)
#         h_prime = torch.matmul(attention, Wh)

#         if self.concat:
#             return self.activation(h_prime)
#         else:
#             return h_prime

#     def _prepare_attentional_mechanism_input(self, Wh):
#         Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
#         Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
#         e = Wh1 + Wh2.T

#         return self.leakyrelu(e)

#     def __repr__(self):
#         return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'




# class GraphAttentionLayer(nn.Module):
#     """
#     Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
#     """
#     def __init__(self, in_features, out_features, dropout, Alpha, concat=True, activation):
#         super(GraphAttentionLayer, self).__init__()
#         self.dropout = dropout
#         self.in_features = in_features
#         self.out_features = out_features
#         self.Alpha = Alpha
#         self.concat = concat
#         self.activation = get_activation_function(activation, alpha)


#         self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
#         nn.init.xavier_uniform_(self.W.data, gain=1.414)
#         self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
#         nn.init.xavier_uniform_(self.a.data, gain=1.414)

#         self.leakyrelu = nn.LeakyReLU(self.Alpha)

#     def forward(self, x, adj):
#         # x.shape: (batch_size, num_nodes, in_features)
#         # adj.shape: (batch_size, num_nodes, num_nodes)
#         Wh = torch.mm(x, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
#         e = self._prepare_attentional_mechanism_input(Wh)

#         zero_vec = -9e15 * torch.ones_like(e)
#         attention = torch.where(adj > 0, e, zero_vec)
#         attention = F.softmax(attention, dim=1)
#         attention = F.dropout(attention, self.dropout, training=self.training)
#         h_prime = torch.matmul(attention, Wh)

#         if self.concat:
#             return self.activation(h_prime)  # Use the activation function passed as a parameter
#         else:
#             return h_prime

#     def _prepare_attentional_mechanism_input(self, Wh):
#         # Wh.shape (N, out_feature)
#         # self.a.shape (2 * out_feature, 1)
#         # Wh1&2.shape (N, 1)
#         # e.shape (N, N)
#         Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
#         Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
#         # broadcast add
#         e = Wh1 + Wh2.T
#         return self.leakyrelu(e)

#     def __repr__(self):
#         return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, Alpha, activation_name, concat=True, alpha=None):
        super(GraphAttentionLayer, self).__init__()
        
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.Alpha = Alpha
        self.concat = concat
        self.activation = get_activation_function(activation_name, alpha=alpha)

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.Alpha)

    def forward(self, x, adj):
        Wh = torch.mm(x, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return self.activation(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
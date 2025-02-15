import math

import optuna
import torch
import torch
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
from adjust_common import get_activation_function, get_conv_layer, get_pool_layer, get_global_pool_layer
from adjust_layers import GraphAttentionLayer
from loss import *
from sklearn.model_selection import KFold
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GraphConv, TopKPooling, SAGPooling, EdgePooling
from torch_geometric.nn import TopKPooling, GCNConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn import global_mean_pool as gap


# from adjust_train117 import get_activation_function
# from adjust_train117 import get_conv_layer
# from adjust_train117 import get_pool_layer
# from adjust_train117 import get_global_pool_layer

# 这是针对1_1_3实验改的model 主要在hidden_dim以及卷积gcn那进行了改动 最高维度到2048
# class Net(torch.nn.Module):
#     def __init__(self, node_input_dim, hidden_dim, num_layers, dropout, gat_heads, activation_name, conv_layer,
#                  pool_layer, global_pool_layer, Alpha, alpha=None):
#         super(Net, self).__init__()
#         self.dropout = dropout
#         self.input_bn = nn.BatchNorm1d(node_input_dim)
#         self.NodeMLP = EmbeddingMLP(input_dim=node_input_dim, output_dim=hidden_dim, num_layers=num_layers,
#                                     activation_name=activation_name, alpha=alpha)
#         self.GAT = GAT(in_features=hidden_dim, hidden_features=hidden_dim, dropout=dropout, Alpha=Alpha,
#                        gat_heads=gat_heads, activation_name=activation_name, alpha=alpha)
#         self.gcn = GCN(input_dim=hidden_dim, hidden_dim=hidden_dim, activation_name=activation_name,
#                        conv_layer=conv_layer, alpha=alpha, pool_layer=pool_layer,
#                        global_pool_layer=global_pool_layer, dropout=dropout)

#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)

#     def forward(self, x, edge_index, batch):
#         x = self.input_bn(x)
#         x = self.NodeMLP(x)
#         adj_matrix = self._calculate_adjacency_matrix_(edge_index).to(x.device)
#         output_h = self.GAT(x, adj_matrix)
#         x = self.gcn(x=output_h, edge_index=edge_index, batch=batch)
#         return x

#     def _calculate_adjacency_matrix_(self, edge_index):
#         num_nodes = edge_index.max().item() + 1  # 确保 num_nodes 足够大
#         adjacency_matrix = torch.zeros((num_nodes, num_nodes), device=edge_index.device)
#         row = edge_index[0]
#         col = edge_index[1]
#         adjacency_matrix[row, col] = 1
#         adjacency_matrix[col, row] = 1  # 对于无向图，边是双向的，所以填充两次
#         return adjacency_matrix

class Net(torch.nn.Module):
    def __init__(self, node_input_dim, hidden_dim, num_layers, dropout, gat_heads, activation_name, conv_layer,
                 pool_layer, global_pool_layer, Alpha, alpha=None):
        super(Net, self).__init__()
        self.dropout = dropout
        self.input_bn = nn.BatchNorm1d(node_input_dim)
        self.NodeMLP = EmbeddingMLP(input_dim=node_input_dim, output_dim=hidden_dim, num_layers=num_layers,
                                    activation_name=activation_name, alpha=alpha)
        self.GAT = GAT(in_features=hidden_dim, hidden_features=hidden_dim, dropout=dropout, Alpha=Alpha,
                       gat_heads=gat_heads, activation_name=activation_name, alpha=alpha)
        self.gcn = GCN(input_dim=hidden_dim, hidden_dim=hidden_dim, activation_name=activation_name,
                       conv_layer=conv_layer, alpha=alpha, pool_layer=pool_layer,
                       global_pool_layer=global_pool_layer, dropout=dropout)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, edge_index, batch):
        x = self.input_bn(x)
        x = self.NodeMLP(x)
        adj_matrix = self._calculate_adjacency_matrix_(edge_index).cuda()
        output_h = self.GAT(x, adj_matrix)
        x = self.gcn(x=output_h, edge_index=edge_index, batch=batch)
        return x

    def _calculate_adjacency_matrix_(self, edge_index):
        num_nodes = edge_index.max().item() + 1
        adjacency_matrix = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
        for src, dst in edge_index.t().tolist():
            adjacency_matrix[src, dst] = 1
            adjacency_matrix[dst, src] = 1  # 对于无向图，边是双向的，所以填充两次
        return adjacency_matrix


class EmbeddingMLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, activation_name, alpha=None):
        super(EmbeddingMLP, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, output_dim, bias=True))
            layers.append(nn.BatchNorm1d(output_dim))
            layers.append(get_activation_function(activation_name, alpha=alpha))  # 使用传递的激活函数
            input_dim = output_dim
        self.hidden_layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.hidden_layers(x)


class GAT(nn.Module):
    def __init__(self, in_features, hidden_features, dropout, Alpha, gat_heads, activation_name, alpha=None):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.Alpha = Alpha
        self.activation = get_activation_function(activation_name, alpha=alpha)
        self.attentions = [GraphAttentionLayer(in_features, hidden_features, dropout=dropout, Alpha=Alpha, concat=True,
                                               activation_name=activation_name, alpha=alpha)
                           for _ in range(gat_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(hidden_features * gat_heads, hidden_features, dropout=dropout, Alpha=Alpha,
                                           concat=False, activation_name=activation_name, alpha=alpha)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.activation(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, activation_name, conv_layer, pool_layer, global_pool_layer,
                 alpha=None):
        super(GCN, self).__init__()
        self.heads = 1
        self.dropout = dropout
        self.activation = get_activation_function(activation_name, alpha=alpha)
        self.conv1 = get_conv_layer(conv_layer, input_dim, hidden_dim)
        self.conv2 = get_conv_layer(conv_layer, hidden_dim, hidden_dim // 2)
        self.conv3 = get_conv_layer(conv_layer, hidden_dim // 2, hidden_dim // 4)
        self.conv4 = get_conv_layer(conv_layer, hidden_dim // 4, hidden_dim // 8)
        self.pool1 = get_pool_layer(pool_layer, hidden_dim)
        self.pool2 = get_pool_layer(pool_layer, hidden_dim // 2)
        self.pool3 = get_pool_layer(pool_layer, hidden_dim // 4)
        self.global_pool = get_global_pool_layer(global_pool_layer)
        self.lin1 = nn.Linear(hidden_dim // 8, 1)

    def forward(self, x, edge_index, batch):
        x = self.activation(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch=batch)
        x = self.activation(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch=batch)
        x = self.activation(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch=batch)
        x = self.activation(self.conv4(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.global_pool(x, batch)
        x = self.lin1(x)
        x = x.squeeze(1) * 100.0
        return x

# class Net(torch.nn.Module):
#     def __init__(self, node_input_dim, hidden_dim, num_layers, dropout, gat_heads, activation, conv_layer, pool_layer, global_pool_layer):
#         super(Net, self).__init__()
#         '''
#         __init__
#         :param node_input_dim: 结点嵌入矩阵的维度
#         :param hidden_dim: 隐藏层的维度
#         :param num_layers: 隐藏层的层数
#         :param dropout: dropout的概率值
#         :param gat_heads: 图注意力机制的的multi-head数
#         :param activation: 激活函数
#         :param conv_layer: 图卷积层
#         :param pool_layer: 图池化层
#         :param global_pool_layer: 图全局池化层

#         forward
#         :input x: 图的结点矩阵
#         :input edge_index: 图的边索引
#         :input batch: 图的batch标签
#         :return: 输出值
#         '''
#         self.dropout = dropout
#         self.input_bn = nn.BatchNorm1d(node_input_dim)
#         self.NodeMLP = EmbeddingMLP(input_dim=node_input_dim, output_dim=hidden_dim, num_layers=num_layers, activation=activation)
#         self.GAT = GAT(in_features=hidden_dim, hidden_features=hidden_dim, dropout=dropout, Alpha=Alpha, gat_heads=gat_heads,activation=activation)
#         self.gcn = GCN(input_dim=hidden_dim, hidden_dim=hidden_dim,  activation=activation, conv_layer=conv_layer, pool_layer=pool_layer, global_pool_layer=global_pool_layer)

#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)

#     def forward(self, x, edge_index, batch):
#         x = self.input_bn(x)
#         x = self.NodeMLP(x)  # 经过mlp处理的优化embedding
#         adj_matrix = self._calculate_adjacency_matrix_(edge_index).cuda()
#         output_h = self.GAT(x, adj_matrix)
#         x = self.gcn(x=output_h, edge_index=edge_index, batch=batch)
#         return x

#     def _calculate_adjacency_matrix_(self, edge_index):

#         num_nodes = edge_index.max().item() + 1

#         # 初始化邻接矩阵为零矩阵
#         adjacency_matrix = torch.zeros(num_nodes, num_nodes, device=edge_index.device)

#         # 填充邻接矩阵
#         for src, dst in edge_index.t().tolist():
#             adjacency_matrix[src, dst] = 1
#             adjacency_matrix[dst, src] = 1  # 对于无向图，边是双向的，所以填充两次

#         return adjacency_matrix


# class EmbeddingMLP(torch.nn.Module):
#     def __init__(self, input_dim, output_dim, num_layers, activation):
#         super(EmbeddingMLP, self).__init__()
#         '''
#         __init__
#         :parma input_dim: 嵌入层的输入维度
#         :parma num_layers: 嵌入层MLP的隐藏层层数
#         :parma hidden_dim: 嵌入层MLP的隐藏层维度

#         forward
#         : input x: 图节点矩阵
#         '''
#         layers = []
#         for _ in range(num_layers):
#             layers.append(nn.Linear(input_dim, output_dim, bias=True))
#             layers.append(nn.BatchNorm1d(output_dim))
#             layers.append(get_activation_function(activation, alpha))  # Use the activation function passed as a parameter
#             #在多层感知机（MLP）中，每一层的输出维度是下一层的输入维度。因此，在构建每一层时，需要更新
#             #input_dim 为上一层的 output_dim，以便正确连接每一层。
#             input_dim = output_dim #当numlayers不为1时就用上了
#         self.hidden_layers = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.hidden_layers(x)


#         return x


# class GAT(nn.Module):
#     def __init__(self, in_features, hidden_features, dropout, Alpha, gat_heads, activation):
#         super(GAT, self).__init__()
#         '''
#         __init__
#         :param in_features: 输入特征维度
#         :param hidden_features: 隐藏特征维度
#         :param dropout: Dropout rate
#         :param alpha: LeakyReLU的alpha参数
#         :param gat_heads: GAT的head数
#         :param activation: 激活函数
#         '''
#         self.dropout = dropout
#         self.Alpha = Alpha
#         self.activation = get_activation_function(activation, alpha)

#         #self.attention是一个包含gat_heads个layer实例的列表
#         #每个layer实例中，输入维度为infeature 输出维度是outfeature
#         #concat=true是这几个layer实例中的注意力输出将被拼接起来
#         #比如每个输出是1x1的矩阵 那么拼接起来以后就是1x3
#         #这里是输入注意力层
#         self.attentions = [GraphAttentionLayer(in_features, hidden_features, dropout=dropout, Alpha=Alpha, concat=True, activation=activation)
#                            for _ in range(gat_heads)]

#         for i, attention in enumerate(self.attentions):
#             self.add_module('attention_{}'.format(i), attention)

#         #输出注意力层 用于将多个注意力头输出整合
#         self.out_att = GraphAttentionLayer(hidden_features * gat_heads, hidden_features, dropout=dropout, Alpha=Alpha, concat=False, activation=activation)

#     def forward(self, x, adj):
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.activation(self.out_att(x, adj))
#         return F.log_softmax(x, dim=1)


# # class GCN(torch.nn.Module):
# #     def __init__(self, input_dim, hidden_dim):
# #         super(GCN, self).__init__()
# #         '''
# #         __init__
# #         :param input_dim: 输入特征维度
# #         :param hidden_dim: 隐藏层特征维度

# #         forward
# #         :input x: 图节点矩阵
# #         :input edge_index: 图边索引

# #         '''
# #         self.dropout = 0.3 #这里变成学习超参数
# #         self.conv1 = GCNConv(input_dim, 1024)
# #         self.conv2 = GCNConv(1024, 512)
# #         self.conv3=  GCNConv(512, 256)
# #         self.pool1 = TopKPooling(1024, ratio=0.8)
# #         self.pool2=TopKPooling(512, ratio=0.8)
# #         self.lin1 = nn.Linear(256, 1)

# #     def forward(self, x, edge_index, batch):
# #         x = F.relu(self.conv1(x, edge_index))  # Apply BN after the first convolution layer out=1024
# #         x = F.dropout(x, self.dropout, training = self.training)
# #         x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch) # out = 1024
# #         x = F.relu(self.conv2(x, edge_index))  # Apply ReLU after the second convolution layer
# #         x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
# #         x= F.relu(self.conv3(x, edge_index))

# #         x = gap(x, batch)  # Global average pooling
# #         x = self.lin1(x).squeeze(1)

# #         #x = torch.sigmoid(x)

# #         x = x * 100.0
# #         return x

# # class GCN(torch.nn.Module):
# #     def __init__(self, input_dim, hidden_dim):
# #         super(GCN, self).__init__()
# #         self.dropout = 0.3
# #         self.conv1 = GCNConv(input_dim, 1600)
# #         #self.res_add=
# #         self.conv2 = GCNConv(1600, 800)
# #         self.conv3 = GCNConv(800, 512)
# #         self.conv4 = GCNConv(512, 256)
# #         self.pool1 = TopKPooling(1600, ratio=0.8)
# #         self.pool2 = TopKPooling(800, ratio=0.8)
# #         self.pool3 = TopKPooling(512 , ratio=0.8)
# #         self.lin1 = nn.Linear(256, 1)
# #         #self.lin2 = nn.Linear(128, 1)

# #     def forward(self, x, edge_index, batch):
# #         x = F.relu(self.conv1(x, edge_index))
# #         x = F.dropout(x, p=self.dropout, training=self.training)
# #         x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
# #         x = F.relu(self.conv2(x, edge_index))
# #         x = F.dropout(x, p=self.dropout, training=self.training)
# #         x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
# #         x = F.relu(self.conv3(x, edge_index))
# #         x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
# #         x = F.relu(self.conv4(x, edge_index))
# #         x = F.dropout(x, p=self.dropout, training=self.training)
# #         x = gap(x, batch)
# #         x = self.lin1(x)
# #         #x=  self.lin2(x)
# #         x = x.squeeze(1) * 100.0
# #         return x

# class GCN(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim, dropout, activation, conv_layer, pool_layer, global_pool_layer):
#         super(GCN, self).__init__()
#         self.dropout = dropout
#         self.activation = get_activation_function(activation, alpha)
#         self.conv1 = get_conv_layer(conv_layer_name, input_dim, hidden_dim)
#         self.conv2 = get_conv_layer(conv_layer.__class__.__name__, hidden_dim, hidden_dim // 2)
#         self.conv3 = get_conv_layer(conv_layer.__class__.__name__, hidden_dim // 2, hidden_dim // 4)
#         self.conv4 = get_conv_layer(conv_layer.__class__.__name__, hidden_dim // 4, hidden_dim // 8)
#         self.pool1 = get_pool_layer(pool_layer_name, hidden_dim)
#         self.pool2 = get_pool_layer(pool_layer.__class__.__name__, hidden_dim // 2)
#         self.pool3 = get_pool_layer(pool_layer.__class__.__name__, hidden_dim // 4)
#         self.global_pool = get_global_pool_layer(global_pool_layer_name)
#         self.lin1 = nn.Linear(hidden_dim // 8, 1)

#     def forward(self, x, edge_index, batch):
#         x = self.activation(self.conv1(x, edge_index))
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
#         x = self.activation(self.conv2(x, edge_index))
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
#         x = self.activation(self.conv3(x, edge_index))
#         x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
#         x = self.activation(self.conv4(x, edge_index))
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.global_pool(x, batch)
#         x = self.lin1(x)
#         x = x.squeeze(1) * 100.0
#         return x

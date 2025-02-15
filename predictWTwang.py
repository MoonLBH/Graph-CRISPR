import math
import copy
import json
import logging
import random
import re
import warnings
import optuna.visualization as vis
import numpy as np
import optuna
import pandas as pd
import torch
import openpyxl
import torch.nn as nn
import torch.nn.functional as F
from adjust_model import Net
from datasetWT import get_data, get_test_data, get_testspilt_data
from loguru import logger
from loss import crit_r2, crit_pearson, crit_spearman
from sklearn.model_selection import KFold
from adjust_common import get_activation_function, get_conv_layer, get_pool_layer, get_global_pool_layer
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GraphConv, TopKPooling, SAGPooling, EdgePooling
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
import os

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_file_path):
    with open(config_file_path, 'r') as f:
        config = json.load(f)
    return config

def predict(config, model_path, graph_test_loader, crit_r2, crit_pearson, crit_spearman, excel_path):

    # config_file_path = '/data/data2/liuxiuqin/JYJ/pygall/gat/config_117testset.json'
    # config = load_config(config_file_path)

    # #初始化模型

    # model = Net(
    #     node_input_dim=config['embed_dim'], hidden_dim=config['hidden_dim'], num_layers=config['layers'],
    #     dropout=config['dropout'], gat_heads=config['heads'], activation_name=activation_name,
    #     conv_layer=conv_layer_name, pool_layer=config['pool_layer'], global_pool_layer=config['global_pool_layer'], Alpha=Alpha, alpha=alpha
    # ).cuda()

    # 加载整个模型对象 避免了初始化模型
    # 加载检查点
    checkpoint = torch.load(model_path)
    model = checkpoint['best_model']
    model = model.cuda()  # 将模型移动到GPU（如果使用GPU）
    model.eval()

    all_label = []
    all_outputs = []

    with torch.no_grad():
        for graph_data in graph_test_loader:
            graph_data = graph_data.cuda()
            output = model(x=graph_data.x, edge_index=graph_data.edge_index, batch=graph_data.batch)
            label = graph_data.y.cuda()

            all_label.extend(label.cpu().numpy().flatten())
            all_outputs.extend(output.cpu().numpy().flatten())

    all_label = np.array(all_label)
    all_outputs = np.array(all_outputs)

    # 调试：打印数组长度
    print(f"Length of all_label: {len(all_label)}")
    print(f"Length of all_outputs: {len(all_outputs)}")

    # 计算指标
    r2_score = crit_r2(all_outputs, all_label).item()
    pearson_score = crit_pearson(all_outputs, all_label).item()
    spearman_score = crit_spearman(all_outputs, all_label).item()

    # 将结果保存到 Excel 文件
    results = pd.DataFrame({
        'True Labels': all_label,
        'Predicted Outputs': all_outputs
    })

    # 将指标添加到 DataFrame 中
    metrics = pd.DataFrame({
        'R2 Score': [r2_score],
        'Pearson Score': [pearson_score],
        'Spearman Score': [spearman_score]
    })

    if not os.path.exists(excel_path):
        with pd.ExcelWriter(excel_path) as writer:
            results.to_excel(writer, sheet_name='Results', index=False)
            metrics.to_excel(writer, sheet_name='Metrics', index=False)
    else:
        with pd.ExcelWriter(excel_path, mode='a', if_sheet_exists='overlay', engine='openpyxl') as writer:
            startrow = writer.sheets['Results'].max_row
            results.to_excel(writer, sheet_name='Results', index=False, startrow=startrow, header=False)
            metrics.to_excel(writer, sheet_name='Metrics', index=False)
    
    return r2_score, pearson_score, spearman_score

if __name__ == "__main__":

    config_file_path = '/data/data2/liuxiuqin/JYJ/pygall/gat/config_WTset.json'
    config = load_config(config_file_path)

    hidden_dim = config['hidden_dim']
    dropout = config['dropout']
    activation_name = config['activation']
    conv_layer_name = config['conv_layer']
    pool_layer_name = config['pool_layer']
    global_pool_layer_name = config['global_pool_layer']
    lr = config['lr']
    Alpha = config['Alpha']
    alpha = config['alpha']
    batch_size = config['batch_size']  # 新增

    device_index = 2
    torch.cuda.set_device(device_index)
    torch.cuda.empty_cache()
# 复刻wt训练过程关于数据集的划分方式，保持测试集与WT里划分的一致。
    set_random_seed(config['seed'])

    indices = np.arange(config['length'])
    # np.random.shuffle(indices)

    # 将数据集划分为80%训练和验证集，20%测试集
    # split_point = int(0.8 * len(indices))
    # train_val_indices = indices[:split_point]
    test_indices = indices

    # set_random_seed(config['seed'])



    # # 加载测试数据集
    # test_indices = np.arange(config['length'])  # 假设在 config 中定义了测试集长度
 #   graph_test_loader = get_test_data(root=config['root'], batch_size=config['batch_size'])
    graph_test_loader = get_testspilt_data(root=config['root'],
                                                       batch_size=batch_size,
                                                       test_index=test_indices
                                                       )

    # 预测并保存结果
    model_path = '/data/data2/liuxiuqin/JYJ/pygall/gat/adjust_model/bestmodel_official/best_model_fold5_epoch90.pt'
    checkpoint = torch.load(model_path)
    excel_path = '/data/data2/liuxiuqin/JYJ/pygall/gat/adjust_output/prediction_WT5W_useofficialmodel_results.xlsx'

    r2_score, pearson_score, spearman_score = predict(config, model_path, graph_test_loader, crit_r2, crit_pearson, crit_spearman, excel_path)

    print(f"R2 Score: {r2_score}")
    print(f"Pearson Score: {pearson_score}")
    print(f"Spearman Score: {spearman_score}")
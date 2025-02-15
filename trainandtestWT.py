import os
import math

import copy
import json
import logging
import os
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
from config import config
from dataset500 import get_data, get_test_data
from loguru import logger
from loss import *
from sklearn.model_selection import KFold
from sklearn.model_selection import KFold
from adjust_common import get_activation_function, get_conv_layer, get_pool_layer, get_global_pool_layer
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GraphConv, TopKPooling, SAGPooling, EdgePooling
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def load_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

def save_config(config, file_path):
    with open(file_path, 'w') as file:
        json.dump(config, file, indent=4)

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(model, optimizer, crit, graph_train_loader, regulization, epoch):
    model.train()
    logger.info(f"训练轮数:{epoch + 1}")
    total_mse_loss, total_r2_score, total_spearman_score, total_pearson_score = 0, 0, 0, 0
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=30, gamma=0.5, last_epoch=-1)
    all_label = []
    all_outputs = []

    for graph_data in graph_train_loader:
        graph_data = graph_data.cuda()
        output = model(x=graph_data.x, edge_index=graph_data.edge_index, batch=graph_data.batch)
        label = graph_data.y.cuda()
        loss = crit(output, label)
        regularization_loss = 0

        if regulization == 'l1':
            l1_lambda = 1e-2
            for param in model.parameters():
                regularization_loss += torch.sum(torch.abs(param))
            loss += l1_lambda * regularization_loss  # 添加 L1 正则化项到损失函数

        elif regulization == 'l2':
            l2_lambda = 1e-2
            for param in model.parameters():
                regularization_loss += torch.sum(param.pow(2))
            loss += l2_lambda * regularization_loss  # 添加 L2 正则化项到损失函数

        optimizer.zero_grad()
        loss.backward()
        all_label.extend(label.detach().cpu().numpy().flatten())
        all_outputs.extend(output.detach().cpu().numpy().flatten())
        optimizer.step()

        total_mse_loss += loss.item()  # 每个batch的损失值累加

    scheduler.step()

    epoch_mse_loss = crit_mse(all_outputs, all_label).item()
    epoch_r2_score = crit_r2(all_outputs, all_label).item()
    epoch_spearman_score = crit_spearman(all_outputs, all_label).item()
    epoch_pearson_score = crit_pearson(all_outputs, all_label).item()

    if np.any(np.isnan(all_outputs)) or np.any(np.isinf(all_outputs)):
        raise ValueError("Epoch outputs contain NaN or infinity values")
        
    if np.any(np.isnan(all_label)) or np.any(np.isinf(all_label)):
        raise ValueError("Epoch labels contain NaN or infinity values")

    return epoch_mse_loss, epoch_r2_score, epoch_spearman_score, epoch_pearson_score

def evaluate(model, crit, graph_val_loader):
    model.eval()  # 将模型设置为评估模式
    total_loss = 0.0
    all_label = []
    all_outputs = []
    with torch.no_grad():  # 禁用梯度计算
        for graph_data in graph_val_loader:
            graph_data = graph_data.cuda()
            output = model(x=graph_data.x, edge_index=graph_data.edge_index, batch=graph_data.batch)
            label = graph_data.y.cuda()
            all_label.extend(label.detach().cpu().numpy().flatten())
            all_outputs.extend(output.detach().cpu().numpy().flatten())
    avg_loss = crit(all_outputs, all_label).item()
    return avg_loss, all_outputs, all_label

def load_data(config, batch_size, train_idx, val_idx):
    graph_train_loader, graph_val_loader = get_data(
        root=config['root'],
        batch_size=batch_size,
        train_index=train_idx,
        val_index=val_idx
    )
    return graph_train_loader, graph_val_loader

def initialize_model(config, hidden_dim, dropout, activation_name, conv_layer, pool_layer, global_pool_layer, lr,
                     Alpha, alpha):
    model = Net(
        node_input_dim=config['embed_dim'], hidden_dim=config['hidden_dim'], num_layers=config['layers'],
        dropout=dropout, gat_heads=config['heads'], activation_name=activation_name,
        conv_layer=conv_layer, pool_layer=pool_layer, global_pool_layer=global_pool_layer, Alpha=Alpha, alpha=alpha
    ).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, min_lr=1e-6)
    return model, optimizer, scheduler

def save_checkpoint(state, filename):
    torch.save(state, filename)
    logging.info(f'Model saved at {filename}')

def load_checkpoint(filename):
    return torch.load(filename)

if __name__ == "__main__":
    device_index = 2
    torch.cuda.set_device(device_index)
    torch.cuda.empty_cache()
    
    config_file_path = '/data/data2/liuxiuqin/JYJ/pygall/gat/config_117official.json' 
    config = load_config(config_file_path)
    set_random_seed(config['seed'])

    indices = np.arange(config['length'])
    np.random.shuffle(indices)

    # 将数据集划分为80%训练和验证集，20%测试集
    split_point = int(0.8 * len(indices))
    train_val_indices = indices[:split_point]
    test_indices = indices[split_point:]

    # 设置 KFold 交叉验证
    kf = KFold(n_splits=config['kfold'], shuffle=True, random_state=42)

    best_model = None
    best_model_dir = '/data/data2/liuxiuqin/JYJ/pygall/gat/adjust_model/bestmodelwang'
    
    criterion = torch.nn.MSELoss(reduction='sum')
    lowest_val_loss = float('inf')

    hidden_dim = config['hidden_dim']
    dropout = config['dropout']
    activation_name = config['activation']
    conv_layer_name = config['conv_layer']
    pool_layer_name = config['pool_layer']
    global_pool_layer_name = config['global_pool_layer']
    lr = config['lr']
    Alpha = config['Alpha']
    alpha = config['alpha']
    batch_size = config['batch_size']

    model, optimizer, scheduler = initialize_model(config, hidden_dim, dropout, activation_name, conv_layer_name,
                                                   pool_layer_name, global_pool_layer_name, lr, Alpha, alpha)

    val_mse_losses = []
    val_spearman_losses = []
    val_pearson_losses = []
    val_r2_losses = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_indices)):
        logging.info(f'Starting fold {fold + 1}/{config["kfold"]}')
        graph_train_loader, graph_val_loader = load_data(config, batch_size, train_idx, val_idx)

        for epoch in range(config['epochs']):
            train(model=model, optimizer=optimizer, crit=criterion, graph_train_loader=graph_train_loader,
                  regulization='l2', epoch=epoch)

            val_mse, _, _ = evaluate(model=model, graph_val_loader=graph_val_loader, crit=crit_mse)
            scheduler.step(val_mse)

            if val_mse < lowest_val_loss:
                lowest_val_loss = val_mse
                best_model = copy.deepcopy(model)
                best_model_path = os.path.join(best_model_dir, f'best_model_fold{fold + 1}_epoch{epoch + 1}.pt')
                save_checkpoint({'best_model': best_model}, best_model_path)
                logging.info(f'Best model updated at fold {fold + 1}, epoch {epoch + 1}')
                
        best_model.eval()

        mse_loss_val, val_outputs, val_labels = evaluate(model=best_model, crit=crit_mse, graph_val_loader=graph_val_loader)
        r2_loss_val = crit_r2(val_outputs, val_labels).item()
        spearman_loss_val = crit_spearman(val_outputs, val_labels).item()
        pearson_loss_val = crit_pearson(val_outputs, val_labels).item()

        val_mse_losses.append(mse_loss_val)
        val_spearman_losses.append(spearman_loss_val)
        val_pearson_losses.append(pearson_loss_val)
        val_r2_losses.append(r2_loss_val)

    pearson_mean = np.mean(val_pearson_losses)
    spearman_mean = np.mean(val_spearman_losses)
    r2_mean = np.mean(val_r2_losses)
    mse_mean = np.mean(val_mse_losses)

    result_mean = pd.DataFrame({
        'mse_mean': [mse_mean],
        'pearson_mean': [pearson_mean],
        'spearman_mean': [spearman_mean],
        'r2_mean': [r2_mean]
    })
    excel_path_mean = '/data/data2/liuxiuqin/JYJ/pygall/gat/adjust_output/6fold_5wWT_mean_results.xlsx'

    if not os.path.exists(excel_path_mean):
        result_mean.to_excel(excel_path_mean, index=False)
    else:
        with pd.ExcelWriter(excel_path_mean, mode='a', if_sheet_exists='overlay', engine='openpyxl') as writer:
            startrow = writer.sheets['Sheet1'].max_row
            result_mean.to_excel(writer, index=False, header=False, startrow=startrow)

    pearson_best = np.max(val_pearson_losses)
    spearman_best = np.max(val_spearman_losses)
    r2_best = np.max(val_r2_losses)
    mse_best = np.min(val_mse_losses)

    result_best = pd.DataFrame({
        'mse_min': [mse_best],
        'pearson_max': [pearson_best],
        'spearman_max': [spearman_best],
        'r2_max': [r2_best]
    })
    excel_path_best = '/data/data2/liuxiuqin/JYJ/pygall/gat/adjust_output/6fold_5wWT_best_results.xlsx'

    if not os.path.exists(excel_path_best):
        result_best.to_excel(excel_path_best, index=False)
    else:
        with pd.ExcelWriter(excel_path_best, mode='a', if_sheet_exists='overlay', engine='openpyxl') as writer:
            startrow = writer.sheets['Sheet1'].max_row
            result_best.to_excel(writer, index=False, header=False, startrow=startrow)

    logging.info('Training completed.')

    # 加载最佳模型并对测试集进行预测
    best_model_path = os.path.join(best_model_dir, f'best_model_fold{fold + 1}_epoch{epoch + 1}.pt')
    best_model_checkpoint = load_checkpoint(best_model_path)
    best_model.load_state_dict(best_model_checkpoint['best_model'])

    test_loader = DataLoader(test_indices, batch_size=batch_size, shuffle=False)
    _, test_outputs, test_labels = evaluate(model=best_model, crit=crit_mse, graph_val_loader=test_loader)

    spearman_corr, _ = spearmanr(test_outputs, test_labels)
    pearson_corr, _ = pearsonr(test_outputs, test_labels)

    print(f'Spearman correlation on test set: {spearman_corr}')
    print(f'Pearson correlation on test set: {pearson_corr}')

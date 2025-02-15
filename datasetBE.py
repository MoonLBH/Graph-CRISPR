import pickle
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.data import Dataset
from tqdm import tqdm
from torch_geometric.data import DataLoader
from config import config

config = config('/data/data2/liuxiuqin/JYJ/pygall/gat/config_BE.json')

def load_datafile(embed_path):
    with open(embed_path, "rb") as f:
        embeddings = pickle.load(f)
    return embeddings


def one_hot_encoding(sequence):
    bases = 'AUGC'
    one_hot = []
    for base in sequence:
        encoding = [1 if base == b else 0 for b in bases]
        one_hot.append(encoding)
    return one_hot


def positional_encoding(max_len, d_model):
    """
    Generate positional encodings
    """
    position_enc = np.zeros((max_len, d_model))
    for pos in range(max_len):
        for i in range(0, d_model, 2):
            position_enc[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            position_enc[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))
    return position_enc


def basic_structure(sequence):
    edges = []
    for i in range(len(sequence) - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])

    edge_indexes = torch.tensor(edges).t().contiguous()
    return edge_indexes


def extract_RNA(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # 使用列表推导式提取RNA以及相应的二级结构
        rna = [lines[i + 1].strip() for i in range(0, len(lines), 3) if i + 1 < len(lines)]
        rna_structure = [lines[i + 2].strip() for i in range(0, len(lines), 3) if i + 2 < len(lines)]
    return rna, rna_structure


def second_structure(structure):
    # 如果结构字符串中不包含左右括号，则返回空的稀疏矩阵
    if '(' not in structure:
        return np.zeros((2, 0), dtype=int)
    else:
        # 初始化行和列的列表
        row = []
        col = []
        stack = []
        # 遍历二级结构字符串，将左右括号对应位置的碱基视为相连的节点
        for i, char in enumerate(structure):
            if char == '(':
                # 左括号入栈
                stack.append(i)
            elif char == ')':
                # 右括号与栈顶左括号对应位置的碱基相连
                if stack:
                    left_bracket_position = stack.pop()
                    row.append(left_bracket_position)
                    col.append(i)
                    row.append(i)
                    col.append(left_bracket_position)
            elif char == '.':
                # 点号表示没有连接，不添加边
                pass
            else:
                # 其他字符可能是碱基，也不添加边
                pass

        # 构建稀疏矩阵
        sparse_matrixes = np.zeros((2, len(row)), dtype=int)
        sparse_matrixes[0, :] = row
        sparse_matrixes[1, :] = col
        return sparse_matrixes


def get_degree_matrix(edge_index):
    # 获取节点的数量
    num_nodes = edge_index.max().item() + 1

    # 初始化度矩阵为零矩阵
    degree_matrix = torch.zeros(num_nodes, num_nodes)

    # 计算每个节点的度数
    for src, dst in edge_index.t().tolist():
        degree_matrix[src, src] += 1

    return degree_matrix


def process_data(rna_strc_path, embed_path):
    rna, rna_structure = extract_RNA(rna_strc_path)

    node_one_hot = [torch.tensor(one_hot_encoding(seq), dtype=torch.float32) for seq in rna]
    node_embed = load_datafile(embed_path)
    basic_matrices = [torch.tensor(basic_structure(seq)) for seq in rna]
    structure_matrices = [torch.tensor(second_structure(structure)) for structure in rna_structure]

    # 生成图矩阵
    graph_matrix = []
    for tensor1, tensor2 in zip(basic_matrices, structure_matrices):
        if torch.any(tensor2 != 0):
            graph_matrix.append(torch.cat((tensor1, tensor2), dim=1))
        else:
            graph_matrix.append(tensor1)

    # 生成最终的结点矩阵
    final_tensor_list = []
    position_matrix = torch.tensor(
        positional_encoding(max_len=20, d_model=640), dtype=torch.float32)
    for i, (one_hot, embed) in enumerate(zip(node_one_hot, node_embed)):
        #onehot矩阵是为了判断要不要填充，不与其拼接
        if one_hot.shape[0] == embed.shape[0]:
            combined_tensor = embed
        else:
            # 计算embed需要扩展的行数
            rows_to_add = one_hot.shape[0] - embed.shape[0]

            # 创建一个新的张量，行数为rows_to_add，列数与embed相同，填充0
            padding = torch.zeros((rows_to_add, embed.shape[1]))

            combined_tensor = torch.cat((embed, padding), dim=0)
        
        combined_tensor = torch.tensor(combined_tensor + position_matrix, dtype=torch.float32)
        final_tensor_list.append(combined_tensor)

    # 将列表转换为张量
    node_feature = torch.stack(final_tensor_list)
    return node_feature, graph_matrix


def get_data(root, batch_size, train_index, val_index):
    graph_dataset = RnaPredictionDataset(root)
    graph_train_loader = DataLoader(graph_dataset, batch_size=batch_size, shuffle=False,drop_last=True,
                                    sampler=torch.utils.data.sampler.SubsetRandomSampler(train_index))
    graph_val_loader = DataLoader(graph_dataset, batch_size=batch_size, shuffle=False,drop_last=True,
                                  sampler=torch.utils.data.sampler.SubsetRandomSampler(val_index))
                                
    return graph_train_loader, graph_val_loader

def evaldata(root):
    return RnaPredictionDataset(root)

#这个是当测试集是一个完整Excel表格的时候使用的   
def get_test_data(root, batch_size):
    dataset = RnaPredictionDataset(root)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return test_loader
#这个是当测试集是从一个Excel表格里划分出来的时候使用的
def get_testspilt_data(root, batch_size,test_index):
    dataset = RnaPredictionDataset(root)
    graph_test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,drop_last=True,
                            sampler=torch.utils.data.sampler.SubsetRandomSampler(test_index))
    
    return graph_test_loader

class RnaPredictionDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return None

    @property
    def processed_file_names(self):
        return ['wideRNA.pt']
        

    def process(self):
        data_list = []
        df = pd.read_csv("/data/data2/liuxiuqin/JYJ/dataset/update_data/BE/BE_1210RNA.csv", nrows=config.length)
        # data_df = pd.read_csv(self.raw_paths[0])  # Modify this line to your actual data reading logic
        # 输出读取的前几行数据和列名
        print("First few rows of the dataframe:")
        print(df.head())

        print("Available columns:")
        print(df.columns)
        #print("Available columns:", df.columns)  # Print all available columns
        node_features, graph_matrix = process_data(rna_strc_path=config.rna_strc_path,
                                                   embed_path=config.embed_path)
        for index, row in tqdm(df.iterrows(), total=len(df)):
            edge_index = graph_matrix[index]         
            node_feature = torch.tensor((node_features[index]), dtype=torch.float32)
            g_label = row['Efficiencies(%)']
            x = node_feature
            y = torch.tensor(g_label, dtype=torch.float)
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)
        data, slices = self.collate(data_list)
        # 将我们的list形式里面存储的一个个小data图数据，汇总，转换成大的data数据集的格式。
        # 这是调用了人家本地的方法，将我们生成的datalist转化为能够直接保存到本地的形式
        torch.save((data, slices), self.processed_paths[0])




acreate_graph_dataset = evaldata(root=config.root)
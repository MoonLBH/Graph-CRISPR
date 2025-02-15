# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import pickle

# 设置文件夹路径
folder_path = '/data/liuxiuqin/jyj2/articles/RNA-FM/redevelop/results/embed500test/representations'  # 替换为你的文件夹路径
# 设置pickle文件的名称
pickle_filename = '/data/data2/liuxiuqin/JYJ/dataset/update_data/kim2019/RNA-FMemb/500RNA20_embeddings_tensors.pkl'  # 可以自定义pickle文件的名称

# 获取所有.npy文件并按文件名中的序号排序
npy_files = sorted(
    [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npy')],
    key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('RNA')[0])
)

# 创建一个空列表来存储Tensor
tensors = []

# 加载每个.npy文件并转换为PyTorch Tensor，然后添加到列表中
for npy_file in npy_files:
    try:
        array = np.load(npy_file)
        tensor = torch.from_numpy(array)
        tensors.append(tensor)
    except Exception as e:
        print(f"Error loading {npy_file}: {e}")

# 将Tensor列表保存到pickle文件中
try:
    with open(pickle_filename, 'wb') as pickle_file:
        pickle.dump(tensors, pickle_file)
    print(f'All npy files have been sorted, converted to tensors, and saved to {pickle_filename}')
except Exception as e:
    print(f"Error saving to pickle file: {e}")

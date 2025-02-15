# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import pickle

# �����ļ���·��
folder_path = '/data/liuxiuqin/jyj2/articles/RNA-FM/redevelop/results/embed500test/representations'  # �滻Ϊ����ļ���·��
# ����pickle�ļ�������
pickle_filename = '/data/data2/liuxiuqin/JYJ/dataset/update_data/kim2019/RNA-FMemb/500RNA20_embeddings_tensors.pkl'  # �����Զ���pickle�ļ�������

# ��ȡ����.npy�ļ������ļ����е��������
npy_files = sorted(
    [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npy')],
    key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('RNA')[0])
)

# ����һ�����б����洢Tensor
tensors = []

# ����ÿ��.npy�ļ���ת��ΪPyTorch Tensor��Ȼ����ӵ��б���
for npy_file in npy_files:
    try:
        array = np.load(npy_file)
        tensor = torch.from_numpy(array)
        tensors.append(tensor)
    except Exception as e:
        print(f"Error loading {npy_file}: {e}")

# ��Tensor�б��浽pickle�ļ���
try:
    with open(pickle_filename, 'wb') as pickle_file:
        pickle.dump(tensors, pickle_file)
    print(f'All npy files have been sorted, converted to tensors, and saved to {pickle_filename}')
except Exception as e:
    print(f"Error saving to pickle file: {e}")

# -*- coding: utf-8 -*-
# @Time    :  2020年09月01日 0001 17:05:57 
# @Author  :  Chen Xiaoyu
# @contact :  xiaoyu981228@163.com
# @Desc    :  生成数据集
# @File    :  generate_dataset.py

import numpy as np

def gen_dataset(vertices, data_path, seq_len, train_test_split_rate):
    """
    :param vertices:检测点编号（0~307）
    :param data_path: 数据集保存的路径
    :param seq_len: 序列长度
    :param train_test_split_rate:训练集和测试集划分比例
    :return:
    """
    raw_data = np.load(data_path)['data']
    all_data = raw_data[:,vertices,:]
    mean = all_data.mean(axis=0, keepdims=True)
    std = all_data.std(axis=0, keepdims=True)
    def normalize(x):
        return (x - mean) / std

    all_data_norm = normalize(all_data)
    train_test = []
    for i in range(seq_len, len(all_data_norm)):
        train_test.append(all_data_norm[i - seq_len: i + 1])
    train_test = np.array(train_test)
    data_index = [i for i in range(len(train_test))]
    np.random.shuffle(data_index)
    train = train_test[data_index[:int(len(train_test)*train_test_split_rate)], :, :]
    test = train_test[data_index[int(len(train_test) * train_test_split_rate):], :, :]
    return {'mean': mean, 'std': std}, train, test



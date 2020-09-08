# -*- coding: utf-8 -*-
# @Time    :  2020年09月01日 0001 17:47:39 
# @Author  :  Chen Xiaoyu
# @contact :  xiaoyu981228@163.com
# @Desc    :  数据迭代
# @File    :  dataset.py

import math

def next_batch(data, batch_size):
    data_length = len(data)
    num_batches = math.ceil(data_length / batch_size)
    for batch_index in range(num_batches):
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, data_length)
        yield data[start_index:end_index]
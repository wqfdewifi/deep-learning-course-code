# -*- coding: utf-8 -*-
# @Time    :  2020年09月01日 0001 10:31:38 
# @Author  :  Chen Xiaoyu
# @contact :  xiaoyu981228@163.com
# @Desc    :  评价指标
# @File    :  metric.py


import numpy as np


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # non_zero_index = (y_true > 0)
    # y_true = y_true[non_zero_index]
    # y_pred = y_pred[non_zero_index]

    mape = np.abs((y_true - y_pred) / y_true)
    # mape[np.isinf(mape)] = 0
    return np.mean(mape) * 100


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
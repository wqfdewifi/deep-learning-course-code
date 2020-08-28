# -*- coding: utf-8 -*-
# @Time    :  2020年08月15日 0015 21:06:28 
# @Author  :  Chen Xiaoyu
# @contact :  xiaoyu981228@163.com
# @Desc    :  手动实现卷积需要的函数
# @File    :  conv_function.py

import torch
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def corr2d(X,K):
    """
    X：输入，shape（batch_size,H,W）
    K：卷积核，shape（k_h，k_w）
    """
    batch_size, H, W = X.shape
    k_h, k_w = K.shape
    #初始化结果矩阵
    Y = torch.zeros((batch_size, H - k_h + 1, W - k_w + 1)).to(device)
    for i in range(Y.shape[1]):
        for j in range(Y.shape[2]):
            Y[:, i, j] = (X[:,i: i + k_h, j: j+k_w]* K).sum()
    return Y

def corr2d_multi_in(X, K):
    #输入x：维度（batch_size,C_in, H, W）
    #卷积核K：维度（C_in, k_h, k_w）
    res = corr2d(X[:,0, :, :], K[0, :, :])
    for i in range(1, X.shape[1]):
        #按通道相加
        res += corr2d(X[:, i, :, :], K[i, :, :])
    return res

def corr2d_multi_in_out(X, K):
    """
    X：shape (batch_size,C_in,H,W)
    K:  shape(C_out,C_in,h,w)
    Y: shape(batch_size,C_out,H_out,W_out)
    """
    return torch.stack([corr2d_multi_in(X, k) for k in K],dim=1)


class MyConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(MyConv2D, self).__init__()
        # 初始化卷积层的2个参数：卷积核、偏差
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.randn((out_channels, in_channels) + kernel_size))
        self.bias = nn.Parameter(torch.randn((out_channels, 1, 1)))

    def forward(self, X):
        """
        x：输入图片，维度（batch_size,C_in,H,W）
        """
        return corr2d_multi_in_out(X, self.weight) + self.bias
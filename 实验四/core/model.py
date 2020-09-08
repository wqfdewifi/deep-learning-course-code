# -*- coding: utf-8 -*-
# @Time    :  2020年09月01日 0001 17:55:52 
# @Author  :  Chen Xiaoyu
# @contact :  xiaoyu981228@163.com
# @Desc    :  模型
# @File    :  model.py

import torch
import torch.nn as nn

class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        :param input_size: 指定输入数据的维度。例如，对于简单的时间序列预测问题，每一步的输入均为一个采样值，因此input_size=1.
        :param hidden_size： 指定隐藏状态的维度。这个值并不受输入和输出控制，但会影响模型的容量.
        :param output_size: 指定输出数据的维度，此值取决于具体的预测要求。例如，对简单的时间序列预测问题， output_size=1.
        """
        super(MyRNN, self).__init__()
        self.hidden_size = hidden_size

        # 可学习参数的维度设置， 可以类比一下全连接网络的实现，其维度取决于输入数据的维度，以及指定的隐藏状态维度。
        self.w_h = nn.Parameter(torch.rand(input_size, hidden_size))
        self.u_h = nn.Parameter(torch.rand(hidden_size, hidden_size))
        self.b_h = nn.Parameter(torch.zeros(hidden_size))

        self.w_y = nn.Parameter(torch.rand(hidden_size, output_size))
        self.b_y = nn.Parameter(torch.zeros(output_size))

        # 准备激活函数。Dropout函数可选
        self.tanh = nn.Tanh()
        self.leaky_relu = nn.LeakyReLU()

        # 可选：使用性能更好的参数初始化函数
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, x):
        """
        :param x: 输入序列。一般来说，此输入包含三个维度：batch，序列长度，以及每条数据的特征
        """
        batch_size = x.size(0)
        seq_len = x.size(1)

        # 初始化隐藏状态，一般设为全0。由于是内部新建的变量，需要同步设备位置
        h = torch.zeros(batch_size, self.hidden_size).to(x.device)
        # RNN实际上只能一步一步处理序列。因此需要用循环迭代。
        y_list = []
        for i in range(seq_len):
            h = self.tanh(torch.matmul(x[:, i, :], self.w_h) +
                          torch.matmul(h, self.u_h) + self.b_h)
            y = self.leaky_relu(torch.matmul(h, self.w_y) + self.b_y)
            y_list.append(y)
        return h, torch.stack(y_list, dim=1)


class torch_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(torch_RNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        r_out, h_state = self.rnn(x)

        outs = []
        for time in range(r_out.size(1)):
            outs.append(self.out(r_out[:, time, :]))
        return h_state, torch.stack(outs, dim=1)


class LSTM_RNN(nn.Module):
    """搭建LSTM"""
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM_RNN, self).__init__()
        # LSTM层
        self.lstm = nn.LSTM(input_size=input_size,    # 输入单元个数
                            hidden_size=hidden_size,  # 隐藏单元个数
                            num_layers= 1,    # 隐藏层数
                            batch_first=True)
        # 输出层
        self.output_layers = nn.Linear(in_features=hidden_size,    # 输入特征个数
                                       out_features=output_size)  # 输出特征个数

    def forward(self, x):
        lstm_out, hidden = self.lstm(x, None)
        outs = []
        for time in range(lstm_out.size(1)):
            outs.append(self.output_layers(lstm_out[:, time, :]))

        return hidden, torch.stack(outs, dim=1)


class GRU_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRU_RNN, self).__init__()
        self.gru = nn.GRU(input_size=input_size,    # 输入单元个数
                            hidden_size=hidden_size,  # 隐藏单元个数
                            num_layers= 1,    # 隐藏层数
                            batch_first=True)
        # 输出层
        self.output_layers = nn.Linear(in_features=hidden_size,    # 输入特征个数
                                       out_features=output_size)  # 输出特征个数

    def forward(self, x):
        gru_out, hidden = self.gru(x, None)
        outs = []
        for time in range(gru_out.size(1)):
            outs.append(self.output_layers(gru_out[:, time, :]))

        return hidden, torch.stack(outs, dim=1)

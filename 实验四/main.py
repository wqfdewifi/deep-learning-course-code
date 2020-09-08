# -*- coding: utf-8 -*-
# @Time    :  2020年09月01日 0001 16:41:29 
# @Author  :  Chen Xiaoyu
# @contact :  xiaoyu981228@163.com
# @Desc    :  主程序
# @File    :  main.py

import torch
from core.generate_dataset import gen_dataset
from core.dataset import next_batch
from core.model import MyRNN, torch_RNN, LSTM_RNN, GRU_RNN
import torch.nn as nn
import numpy as np
import math
from core.metric import mape, mae, mse
import time
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Hyperparams')
parser.add_argument('--data_path', nargs='?', type=str, default='./data/高速公路传感器数据/PEMS04/PEMS04.npz',
                    help='data path')
parser.add_argument('--arch', nargs='?', type=str, default='LSTM_RNN',
                    help='Architecture to use [\'MyRNN, torch_RNN, LSTM_RNN, GRU_RNN\']')
parser.add_argument('--batch_size', nargs='?', type=int, default=64,
                    help='define batch_size')
parser.add_argument('--lr', nargs='?', type=float, default=0.0005,
                    help='define learning rate')
parser.add_argument('--epochs', nargs='?', type=int, default=3,
                    help='define max epochs')
parser.add_argument('--vertices', nargs='?', type=int, default=5,
                    help='choose the number of vertices')
parser.add_argument('--seq_len', nargs='?', type=int, default=12,
                    help='sequence len')
parser.add_argument('--train_test_split_rate', nargs='?', type=float, default=0.8,
                    help='split the train_set and test_set rate')
parser.add_argument('--hidden_size', nargs='?', type=int, default=32,
                    help='hidden size of the model')
args = parser.parse_args()

assert args.vertices >= 0 and args.vertices < 307
vertices = args.vertices  # 选择测量点编号（0~306）
data_path = args.data_path
seq_len = args.seq_len
train_test_split_rate = args.train_test_split_rate  # train : test = 0.8 :0.2
batch_size = args.batch_size
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
record, train_set, test_set = gen_dataset(vertices, data_path, seq_len, train_test_split_rate)

if args.arch == 'MyRNN':
    model = MyRNN(input_size=3, hidden_size=args.hidden_size, output_size=1).to(device)
elif args.arch == 'torch_RNN':
    model = torch_RNN(input_size=3, hidden_size=args.hidden_size, output_size=1).to(device)
elif args.arch == 'LSTM_RNN':
    model = LSTM_RNN(input_size=3, hidden_size=args.hidden_size, output_size=1).to(device)
elif args.arch == 'GRU_RNN':
    model = GRU_RNN(input_size=3, hidden_size=args.hidden_size, output_size=1).to(device)
else:
    raise Exception(f'网络结构可选范围:MyRNN,torch_RNN,LSTM_RNN,GRU_RNN'
                    f'当前网络为{args.arch}')


loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


loss_log = []
score_log = []
trained_batches = 0

train_start_time = time.perf_counter()
for epoch in range(args.epochs):
    for batch in next_batch(train_set, batch_size):
        batch = torch.from_numpy(batch).float().to(device)
        # 使用短序列最后一个值作为预测值。
        x, label = batch[:, :-1, :], batch[:, -1, 0]

        hidden, out = model(x)
        prediction = out[:, -1, :].squeeze(-1)

        loss = loss_func(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_log.append(loss.detach().cpu().numpy().tolist())
        trained_batches += 1

        # 每训练一定数量的batch，就在测试集上测试模型效果
        if trained_batches % 200 == 0:
            all_prediction = []
            for batch in next_batch(test_set, batch_size=64):
                batch = torch.from_numpy(batch).float().to(device)
                x, label = batch[:, :-1, :], batch[:, -1, 0]

                hidden, out = model(x)
                prediction = out[:, -1, :].squeeze(-1)
                all_prediction.append(prediction.detach().cpu().numpy())
            all_prediction = np.concatenate(all_prediction)
            all_label = test_set[:, -1, 0]

            # 计算测试指标
            rmse_score = math.sqrt(mse(all_label, all_prediction))
            mae_score = mae(all_label, all_prediction)
            mape_score = mape(all_label, all_prediction)
            score_log.append([rmse_score, mae_score, mape_score])
            print('RMSE: %.4f, MAE: %.4f, MAPE: %.4f' % (rmse_score, mae_score, mape_score))

train_end_time = time.perf_counter()
with open(f'{args.arch}_time.txt', 'w') as f:
    f.write(str(train_end_time - train_start_time))
print(f'time: {train_end_time - train_start_time:.2f}')

# save result
# print('saving result ...')
# plt.plot(loss_log)
# plt.xlabel('Number of batches')
# plt.ylabel('Loss value')
# plt.savefig(f'{args.arch}_loss.jpg')
# np.save(f'{args.arch}_loss_log.npy', loss_log)
# np.save(f'{args.arch}_score_log.npy', score_log)

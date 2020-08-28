# -*- coding: utf-8 -*-
# @Time    :  2020年08月15日 0015 14:05:23 
# @Author  :  Chen Xiaoyu
# @contact :  xiaoyu981228@163.com
# @Desc    :  训练和测试的主函数
# @File    :  main.py

import os
from os.path import join as pjoin
import argparse
from core.data_loader import my_dataloader
from core.model import *
import torchvision.transforms as transforms
import torch.utils.data as data
import torch
import torch.nn as nn
import time
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Hyperparams')
parser.add_argument('--Dataset', nargs='?', type=str, default='车辆分类数据集',
                    help='Dataset to Use:[\'车辆分类数据集, 去雾数据集\']')
parser.add_argument('--arch', nargs='?', type=str, default='MyConvModule',
                    help='Architecture to use [\'MyConvModule, ConvModule, AlexNet, LinearNet, DilatedConvModule, DilatedConvModule2, ResNet18, DilatedResNet18\']')
parser.add_argument('--batch_size', nargs='?', type=int, default='32',
                    help='define batch_size')
parser.add_argument('--lr', nargs='?', type=float, default='0.001',
                    help='define learning rate')
parser.add_argument('--epochs', nargs='?', type=int, default='100',
                    help='define max epochs')
args = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.Dataset == '车辆分类数据集':
    file_path = './Datasets/车辆分类数据集'
    file_list = os.listdir(file_path)
    sample = []
    if os.path.exists(pjoin('./Datasets', 'cars_train.txt')):
        os.remove(pjoin('./Datasets', 'cars_train.txt'))
    if os.path.exists(pjoin('./Datasets', 'cars_train_label.txt')):
        os.remove(pjoin('./Datasets', 'cars_train_label.txt'))
    if os.path.exists(pjoin('./Datasets', 'cars_test.txt')):
        os.remove(pjoin('./Datasets', 'cars_test.txt'))
    if os.path.exists(pjoin('./Datasets', 'cars_test_label.txt')):
        os.remove(pjoin('./Datasets', 'cars_test_label.txt'))
    for i in range(len(file_list)):
        sample = os.listdir(pjoin(file_path,file_list[i]))
        train_len = int(len(sample) * 0.8)
        with open(pjoin('./Datasets', 'cars_train.txt'), 'a') as f:
            for j in range(train_len):
                if i == 0 and j == 0:
                    f.write(file_list[i] + '/' + sample[j])
                else:
                    f.write('\n')
                    f.write(file_list[i] + '/' +sample[j])
        with open(pjoin('./Datasets', 'cars_train_label.txt'), 'a') as f:
            for j in range(train_len):
                if i == 0 and j == 0:
                    f.write(str(i))
                else:
                    f.write('\n')
                    f.write(str(i))
        with open(pjoin('./Datasets', 'cars_test.txt'), 'a') as f:
            for j in range(train_len+1, len(sample)):
                if i == 0 and j == train_len+1:
                    f.write(file_list[i] + '/' + sample[j])
                else:
                    f.write('\n')
                    f.write(file_list[i] + '/' +sample[j])
        with open(pjoin('./Datasets', 'cars_test_label.txt'), 'a') as f:
            for j in range(train_len+1, len(sample)):
                if i == 0 and j == train_len+1:
                    f.write(str(i))
                else:
                    f.write('\n')
                    f.write(str(i))
    num_classes = 3
    cars_transforms = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = my_dataloader(args.Dataset, transforms=cars_transforms, train=True)
    test_dataset = my_dataloader(args.Dataset, transforms=cars_transforms, train=False)

    train_dataloader = data.DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=0)
    test_dataloader = data.DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=0)

elif args.Dataset == '去雾数据集':
    pass
else:
    raise Exception(f'数据集可选范围:车辆分类数据集,去雾数据集. 当前数据集为{args.Dataset}')


def train_epoch(net, data_loader, device, criterion, optimizer):
    net.train()  # 指定当前为训练模式
    train_batch_num = len(data_loader)  # 记录共有多少个batch
    total_loss = 0  # 记录Loss
    correct = 0  # 记录共有多少个样本被正确分类
    sample_num = 0  # 记录样本总数

    # 遍历每个batch进行训练
    for batch_idx, (data, target) in enumerate(data_loader):
        # 将图片放入指定的device中
        data = data.to(device).float()
        # 将图片标签放入指定的device中
        target = target.to(device).long()
        # 将当前梯度清零
        optimizer.zero_grad()
        # 使用模型计算出结果
        output = net(data)
        # 计算损失
        loss = criterion(output, target)
        # 进行反向传播
        loss.backward()
        optimizer.step()
        # 累加loss
        total_loss += loss.item()
        # 找出每个样本值最大的idx，即代表预测此图片属于哪个类别
        prediction = torch.argmax(output, 1)
        # 统计预测正确的类别数量
        correct += (prediction == target).sum().item()
        # 累加当前的样本总数
        sample_num += len(prediction)
    # 计算平均的Loss与准确率
    loss = total_loss / train_batch_num
    acc = correct / sample_num
    return loss, acc

def test_epoch(net, data_loader, device, criterion):
    net.eval()
    test_batch_num = len(data_loader)
    total_loss = 0
    correct = 0
    sample_num = 0
    # 指定不进行梯度变化
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data = data.to(device).float()
            target = target.to(device).long()
            output = net(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            prediction = torch.argmax(output, 1)
            correct += (prediction == target).sum().item()
            sample_num += len(prediction)
    loss = total_loss / test_batch_num
    acc = correct / sample_num
    return loss, acc

def train(args):
    if args.arch == 'MyConvModule':
        net = MyConvModule(num_classes).to(device)
    elif args.arch == 'ConvModule':
        net = ConvModule(num_classes).to(device)
    elif args.arch == 'AlexNet':
        net = AlexNet(num_classes).to(device)
    elif args.arch == 'LinearNet':
        net = AlexNet(num_classes).to(device)   
    elif args.arch == 'DilatedConvModule':
        net = DilatedConvModule(num_classes).to(device)
    elif args.arch == 'DilatedConvModule2':
        net = DilatedConvModule2(num_classes).to(device)
    elif args.arch == 'ResNet18':
        net = ResNet18(num_classes).to(device)
    elif args.arch == 'DilatedResNet18':
        net = DilatedResNet18(num_classes).to(device)
    else:
        raise Exception(f'网络结构可选范围:MyConvModule,ConvModule,AlexNet,LinearNet,DilatedConvModule,DilatedConvModule2,ResNet18,DilatedResNet18.  当前网络为{args.arch}')

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(),lr=args.lr)

    # 存储每一个epoch的loss与acc的变化，便于后面可视化
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []

    # 进行训练
    for epoch in tqdm(range(args.epochs)):
        # 在训练集上训练
        train_loss, train_acc = train_epoch(net, data_loader=train_dataloader,
                                            device=device, criterion=loss,
                                            optimizer=optimizer)
        # 在测试集上验证
        test_loss, test_acc = test_epoch(net, data_loader=test_dataloader,
                                         device=device, criterion=loss)
        # 保存各个指标
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)
        print(f"epoch:{epoch+1}\t train_loss:{train_loss:.4f}\t"
              f"train_acc:{train_acc}\t"
              f"test_loss:{test_loss:.4f}\t test_acc:{test_acc}")

    return train_loss_list, train_acc_list, test_loss_list, test_acc_list


if __name__ == '__main__':
    start = time.time()
    train_loss_list, train_acc_list, test_loss_list, test_acc_list = train(args)
    end = time.time()
    with open(f'{args.arch}_time.txt','w') as f:
        f.write(str(end - start))
    print(f'time: {end - start} s')
    np.save(f'{args.arch}_train_loss_list.npy',train_loss_list)
    np.save(f'{args.arch}_train_acc_list.npy',train_acc_list)
    np.save(f'{args.arch}_test_loss_list.npy',test_loss_list)
    np.save(f'{args.arch}_test_acc_list.npy',test_acc_list)







# -*- coding: utf-8 -*-
# @Time    :  2020年08月15日 0015 13:59:31 
# @Author  :  Chen Xiaoyu
# @contact :  xiaoyu981228@163.com
# @Desc    :  数据集的加载
# @File    :  data_loader.py

from PIL import Image
from torch.utils import data
from os.path import join as pjoin


class my_dataloader(data.Dataset):
    def __init__(self, load_Dataset, transforms=None, train=True):
        if load_Dataset == '车辆分类数据集':
            self.path = './Datasets/车辆分类数据集'
            self.image_path = []
            self.labels = []
            if train == True:
                with open('./Datasets/cars_train.txt', 'r') as f:
                    for i in f.readlines():
                        self.image_path.append(i.strip('\n'))
                with open('./Datasets/cars_train_label.txt', 'r') as f:
                    for i in f.readlines():
                        self.labels.append(i.strip('\n'))
            elif train == False:
                with open('./Datasets/cars_test.txt', 'r') as f:
                    for i in f.readlines():
                        self.image_path.append(i.strip('\n'))
                with open('./Datasets/cars_test_label.txt', 'r') as f:
                    for i in f.readlines():
                        self.labels.append(i.strip('\n'))
        elif load_Dataset == '去雾数据集':
            pass

        self.transforms = transforms

    def __getitem__(self, index):
        img = Image.open(pjoin(self.path, self.image_path[index]))
        label = int(self.labels[index])
        if self.transforms is not None:
            img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.image_path)
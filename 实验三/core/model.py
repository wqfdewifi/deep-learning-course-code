# -*- coding: utf-8 -*-
# @Time    :  2020年08月15日 0015 21:00:57 
# @Author  :  Chen Xiaoyu
# @contact :  xiaoyu981228@163.com
# @Desc    :  定义网络模型
# @File    :  model.py

from torch import nn
from core.conv_function import MyConv2D


class MyConvModule(nn.Module):
    def __init__(self, num_classes):
        super(MyConvModule, self).__init__()
        # 定义三层卷积
        self.conv1 = nn.Sequential(
            MyConv2D(in_channels=3, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            MyConv2D(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            MyConv2D(in_channels=64, out_channels=128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # 输出层，将特征通道数变为分类数量
#         self.fc = nn.Linear(128, num_classes)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, X):
        # 图片先经过三层卷积，输出维度（batch_size,C_out,H,W）
        out = self.conv1(X)
#         out = self.conv2(out)
#         out = self.conv3(out)
        # 使用平均池化层将图片的大小变为1×1
        out = nn.functional.avg_pool2d(out, 30)
        # 将张量out从shape batch × 32 × 1 × 1 变为 batch × 32
        out = out.view(out.size(0),-1)
        # out = out.squeeze()
        # 输入到全连接层将输出的维度变为3
        out = self.fc(out)
        return out

class ConvModule(nn.Module):
    def __init__(self, num_classes):
        super(ConvModule, self).__init__()
        # 定义一个三层卷积
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # 输出层，将通道数变为分类数量
        self.fc = nn.Linear(128, num_classes)

    def forward(self, X):
        out = self.conv(X)
        out = nn.functional.avg_pool2d(out, 26)
        out = out.squeeze()
        out = self.fc(out)
        return out


class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=3, stride=1, padding=1),  # input[3, 32, 32]  output[48, 32, 32]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[48, 15, 15]
            nn.Conv2d(48, 128, kernel_size=3, stride=1, padding=1),           # output[128, 15, 15]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 7, 7]
            nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1),          # output[192, 7, 7]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),          # output[192, 7, 7]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1),          # output[128, 7, 7]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 3, 3]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 3 * 3, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.shape[0],-1)
        out = self.classifier(out)
        return out


    
class LinearNet(nn.Module):
    def __init__(self, num_classes):
        super(LinearNet, self).__init__()
        self.linear = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(3*32*32, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        out = self.linear(x)
        return out
    

    
    
class DilatedConvModule(nn.Module):
    def __init__(self, num_classes):
        super(DilatedConvModule, self).__init__()
        # 定义一个空洞率为1，2，5的三层空洞卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32,
                      kernel_size=3, stride=1,
                      padding=0, dilation=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, stride=1,
                      padding=0, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, stride=1,
                      padding=0, dilation=5),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=3, stride=1,
                      padding=1, dilation=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=3, stride=1,
                      padding=1, dilation=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=128,
                      kernel_size=3, stride=1,
                      padding=1, dilation=5),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

        )
        # 输出层， 将通道数变为分类数量
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = nn.functional.avg_pool2d(out, 6)
        out = out.squeeze()
        out = self.fc(out)
        return out


class DilatedConvModule2(nn.Module):
    def __init__(self, num_classes):
        super(DilatedConvModule2, self).__init__()
        # 定义一个空洞率为2，4，6的三层空洞卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32,
                      kernel_size=3, stride=1,
                      padding=0, dilation=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, stride=1,
                      padding=0, dilation=4),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, stride=1,
                      padding=0, dilation=6),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=3, stride=1,
                      padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=3, stride=1,
                      padding=4, dilation=4),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=128,
                      kernel_size=3, stride=1,
                      padding=6, dilation=6),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

        )
        # 输出层， 将通道数变为分类数量
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = nn.functional.avg_pool2d(out, 8)
        out = out.squeeze()
        out = self.fc(out)
        return out    
    


    
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = nn.functional.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = nn.functional.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18(num_classes):

    return ResNet(ResidualBlock,num_classes)



class DilatedResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(DilatedResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False, dilation=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=2, bias=False, dilation=2),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=5, bias=False, dilation=5),
            nn.BatchNorm2d(outchannel),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = nn.functional.relu(out)
        return out


def DilatedResNet18(num_classes):

    return ResNet(DilatedResidualBlock,num_classes)
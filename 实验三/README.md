# 卷积神经网络实验

The code in mainly built using the [PyTorch](https://pytorch.org/) deep learning library.

## Dataset

Download the data from 链接：https://pan.baidu.com/s/1UXSlq0UExqsjsqkWKwO3wg    提取码：i4og

Make sure you have the following folder structure in the `Datasets` directory after you unzip the file: 

```bash
Datasets
├── 去雾数据集(I did not this Dataset，you can use it if you are interested.)
└── 车辆分类数据集
    ├── bus
    └── car
    └── truck
```

All data are in  .jpg format

## Getting Started

 Before you run the code, make sure you have the following packages installed:

### Prerequisites:

The version numbers are the exact ones I've used, but newer versions should works just fine. 

```
Pillow == 5.1.0
numpy == 1.16.2
torch == 1.4.0
torchvision == 0.5.0
tqdm == 4.43.0
```

### Train and Test:

To train and test the AlexNet model with a different batch size and epochs you can run:

```
python main.py --arch 'AlexNet' --batch_size 32 --epohs 100
```

Unless you specify the options you want, the default ones (listed in main.py will be used). 

Notice that there are 8 architectures you can chose.  



## Questions?

If you have any questions about the code, you can contact me via email, or even better open an issue in this repo and I'll do my best to help.


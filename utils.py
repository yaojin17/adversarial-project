import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms

import os
import sys
import time


# 检查是否存在路径，不存在的话则创建该路径
def check_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# 写日志函数，不过print函数的逻辑有点没看明白，而且时间好像也没用上
class Logger:
    def __init__(self, file, print=True):
        self.file = file
        local_time = time.strftime("%b%d_%H%M%S", time.localtime()) 
        # self.file += local_time
        self.All_file = 'logs/All.log'
        
    def print(self, content='', end='\n', file=None):
        if file is None:
            file = self.file
        with open(file, 'a') as f:
            if isinstance(content, str):
                f.write(content+end)
            else:
                old = sys.stdout
                sys.stdout = f
                print(content)
                sys.stdout = old
        if file is None:
            self.print(content, file=self.All_file)
        print(content,end=end)


# 导入测试数据集并且将其转化为tensor格式，归一化到0-1，并且转化为迭代器格式
def get_test_cifar(batch_size):
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    return test_loader


# 图片预处理，生成训练与测试数据迭代器
def prepare_cifar(batch_size, test_batch_size):
    kwargs = {'num_workers': 0, 'pin_memory': True}
    # kwargs = {'num_workers': 2, 'pin_memory': True}
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 随机裁剪成大小32*32的图片，然后再在周围pad四圈0，最后为40*40
        transforms.RandomHorizontalFlip(),  # 依概率0.5水平翻转图片
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, **kwargs)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader

if __name__ == '__main__':
    batch_size = 256
    kwargs = {'num_workers': 2, 'pin_memory': True}
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 随机裁剪成大小32*32的图片，然后再在周围pad四圈0，最后为40*40
        transforms.RandomHorizontalFlip(),  # 依概率0.5水平翻转图片
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset,batch_size=256)
    for batch_idx, (image, target) in enumerate(train_loader):
        if batch_idx == 5:
            # print(type(data),type(target))
            print(image.size(),target.size())
            img = image[80]
            img = transforms.ToPILImage()(img)
            img.show()
            # print(img.size())
            # img = transforms.ToPILImage()
            # image[5].show()
            print(target[80])
    # train_loader = train_loader


    # print (len(trainset))
    # print (sizeof(train_loader))
    # print()
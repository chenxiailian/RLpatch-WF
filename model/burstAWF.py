# -*- coding: utf-8 -*-
"""
Created on 24-2-25 下午5:26

@author: chensiyang
"""
import torch
import torch.nn as nn
import math
import numpy as np

class burstAWFNet(torch.nn.Module):
    def __init__(self, input_shape, classes):
        super(burstAWFNet,self).__init__()
        dropout = 0.1
        filters = 32
        kernel_size = 5
        stride_size = 1
        pool_size = 4
        
        self.dropout = nn.Dropout(p=dropout)
        
        self.conv1 = nn.Conv1d(input_shape[0], filters, kernel_size, stride=stride_size, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(pool_size, padding=2)

        self.conv2 = nn.Conv1d(filters, filters, kernel_size, stride=stride_size, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(pool_size, padding=2)

        self.conv3 = nn.Conv1d(filters, filters, kernel_size, stride=stride_size, padding=2)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(pool_size, padding=2)
        
        #torch.Size([3, 32, 79])三层
        # torch.Size([3, 32, 20])四层
        
        # self.conv4 = nn.Conv1d(filters, filters, kernel_size, stride=stride_size, padding=2)
        # self.relu4 = nn.ReLU()
        # self.pool4 = nn.MaxPool1d(pool_size, padding=2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32*13, classes)

    def forward(self, x):
        x = self.dropout(x)
        
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        # print(x.shape)
        # x = self.conv4(x)
        # x = self.relu4(x)
        # x = self.pool4(x)
        
        x = self.flatten(x)
        x = self.fc(x)

        return x
    
if __name__ == "__main__":
    #trans = transforms.Compose([ transforms.ToTensor(), ])
    input = torch.Tensor(3,1,800)
    print(input.shape)
    input_shape = (1, 800)  # 请替换为实际的输入形状
    classes = 95  # 请替换为实际的类别数
    net = burstAWFNet(input_shape,classes)
    
    # k=np.ones((40,30,3))
    # imageform = Image.fromarray(np.uint8(k))
    # kt  = trans(imageform)
    # print(np.array(kt).shape)
    
    #net = resnet50()
    # #print(net)

    x = net(input)
    x = x.data.cpu().numpy()
    print(x.shape)
    print(x)

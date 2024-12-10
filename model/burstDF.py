# -*- coding: utf-8 -*-
"""
Created on 24-2-25 下午5:26

@author: chensiyang
"""
import torch
import torch.nn as nn
import math
import numpy as np

class burstDFNet(torch.nn.Module):
    def __init__(self, input_shape, classes):
        super(burstDFNet,self).__init__()
        self.block_conv11=nn.Conv1d(in_channels=input_shape[0],out_channels=32,kernel_size=8,stride=1,padding=4)
        self.block_bn11=nn.BatchNorm1d(32)
        self.block_relu11=nn.ELU()
        self.block_conv12=nn.Conv1d(32,32,kernel_size=8,stride=1,padding=4)
        self.block_bn12=nn.BatchNorm1d(32)
        self.block_relu12=nn.ELU()
        self.block_maxpool1=nn.MaxPool1d(8,stride=4,padding=4)
        self.block_dropout1=nn.Dropout(p=0.1)
        
        self.block_conv21=nn.Conv1d(32,64,kernel_size=8,stride=1,padding=4)
        self.block_bn21=nn.BatchNorm1d(64)
        self.block_relu21=nn.ReLU()
        self.block_conv22=nn.Conv1d(64,64,kernel_size=8,stride=1,padding=4)
        self.block_bn22=nn.BatchNorm1d(64)
        self.block_relu22=nn.ReLU()
        self.block_maxpool2=nn.MaxPool1d(8,stride=4,padding=4)
        self.block_dropout2=nn.Dropout(p=0.1)
        
        self.block_conv31=nn.Conv1d(64,128,kernel_size=8,stride=1,padding=4)
        self.block_bn31=nn.BatchNorm1d(128)
        self.block_relu31=nn.ReLU()
        self.block_conv32=nn.Conv1d(128,128,kernel_size=8,stride=1,padding=4)
        self.block_bn32=nn.BatchNorm1d(128)
        self.block_relu32=nn.ReLU()
        self.block_maxpool3=nn.MaxPool1d(8,stride=4,padding=4)
        self.block_dropout3=nn.Dropout(p=0.1)
        
        self.block_conv41=nn.Conv1d(128,256,kernel_size=8,stride=1,padding=4)
        self.block_bn41=nn.BatchNorm1d(256)
        self.block_relu41=nn.ReLU()
        self.block_conv42=nn.Conv1d(256,256,kernel_size=8,stride=1,padding=4)
        self.block_bn42=nn.BatchNorm1d(256)
        self.block_relu42=nn.ReLU()
        self.block_maxpool4=nn.MaxPool1d(8,stride=4,padding=4)
        self.block_dropout4=nn.Dropout(p=0.1)
        
        self.flatten =nn.Flatten()
        self.fc1=nn.Linear(256*5,512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()

        self.dropout1 = nn.Dropout(0.7)

        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu2 = nn.ReLU()

        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(512, classes)
        # self.softmax = nn.Softmax(dim=1)
        
    def forward(self,x):
        x=self.block_conv11(x)
        x=self.block_bn11(x)
        x=self.block_relu11(x)
        x=self.block_conv12(x)
        x=self.block_bn12(x)
        x=self.block_relu12(x)
        x=self.block_maxpool1(x)
        x=self.block_dropout1(x)
        
        x=self.block_conv21(x)
        x=self.block_bn21(x)
        x=self.block_relu21(x)
        x=self.block_conv22(x)
        x=self.block_bn22(x)
        x=self.block_relu22(x)
        x=self.block_maxpool2(x)
        x=self.block_dropout2(x)
        
        x=self.block_conv31(x)
        x=self.block_bn31(x)
        x=self.block_relu31(x)
        x=self.block_conv32(x)
        x=self.block_bn32(x)
        x=self.block_relu32(x)
        x=self.block_maxpool3(x)
        x=self.block_dropout3(x)
        
        x=self.block_conv41(x)
        x=self.block_bn41(x)
        x=self.block_relu41(x)
        x=self.block_conv42(x)
        x=self.block_bn42(x)
        x=self.block_relu42(x)
        x=self.block_maxpool4(x)
        x=self.block_dropout4(x)
        
        # print(x.shape)
        x=self.flatten(x)
        # print(x.shape)
        x=self.fc1(x)
        x=self.bn1(x)
        x = self.relu1(x)

        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.dropout2(x)

        x = self.fc3(x)
        # x = self.softmax(x)s
        
        return(x)
if __name__ == "__main__":
    #trans = transforms.Compose([ transforms.ToTensor(), ])
    input = torch.randn(3,1,800)
    print(input.shape)
    input_shape = (1, 800)  # 请替换为实际的输入形状
    classes = 95  # 请替换为实际的类别数
    net = burstDFNet(input_shape,classes)
    
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

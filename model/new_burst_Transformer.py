"""
Created on 24-3-11 下午5:26

@author: chensiyang
"""
import torch
import torch.nn as nn
import math
import numpy as np
from vit_pytorch import ViT


# -*- coding: utf-8 -*-
"""
Created on 24-2-25 下午5:26

@author: chensiyang
"""
import torch
import torch.nn as nn
import math
import numpy as np
from vit_pytorch import ViT


# 定义Transformer神经网络模型
class burstTransformerNet(nn.Module):
    def __init__(self,input_shape, classes):
        super(burstTransformerNet, self).__init__()

        self.Trans = ViT(image_size=(40,20), # 20*256该值会随着forward里设定的窗口大小而改变
                        patch_size=(1,20),
                        num_classes=classes,
                        dim=256,
                        depth=4,
                        heads=4,
                        mlp_dim=512,
                        dropout=0.1,
                        channels=1) 
        
    def forward(self, x):
        # 分割输入序列为10个窗口
        windows = torch.chunk(x, 40, dim=2)
        

        # 针对每个窗口进行前向传播
        outputs = []
        for window in windows:
            x = window
            # x=x.reshape(1,20)
            # print(x.shape)
            outputs.append(x)
        
        # print(outputs)
        # 将窗口的输出拼接起来
        x = torch.stack(outputs, dim=1)
        # print(x.shape)
        x=torch.squeeze(x,2) 
        # print(x.shape)
        x = torch.unsqueeze(x, dim=1)
        # print(x.shape)
        # print(x)

        # 进入transformer
        x = self.Trans(x)

        return x
    
if __name__ == "__main__":
    #trans = transforms.Compose([ transforms.ToTensor(), ])
    input = torch.rand(1,1,800)
    input_shape=(1,800)
    classes=100
    net = burstTransformerNet(input_shape,classes)
    
    # k=np.ones((40,30,3))
    # imageform = Image.fromarray(np.uint8(k))
    # kt  = trans(imageform)
    # print(np.array(kt).shape)
    
    #net = resnet50()
    # #print(net)

    x = net(input)
    print(x)
    x = x.data.cpu().numpy()
    print(x.shape)
    print(x)

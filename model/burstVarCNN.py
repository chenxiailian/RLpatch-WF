import torch
from torch import nn
# 导入记好了,一维卷积，一维最大池化，展成1维，全连接层，构建网络结构辅助工具,1d网络归一化,激活函数,自适应平均池化
from torch.nn import Conv1d, MaxPool1d, Flatten, Linear, Sequential, BatchNorm1d, ReLU, AdaptiveAvgPool1d
from torchsummary import summary


class burstVarCNN(nn.Module):
    def __init__(self, input_shape,num_classes):
        super(burstVarCNN, self).__init__()
        self.model0 = Sequential(
            # 输入形状（1,5000）
            # 输入1通道、输出64通道、卷积核大小、步长、补零、
            Conv1d(in_channels=input_shape[0], out_channels=64, kernel_size=7, stride=2, padding=3),
            BatchNorm1d(64),
            ReLU(),
            MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.model1 = Sequential(
            # 1.1
            Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            BatchNorm1d(64),
            ReLU(),
            Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            BatchNorm1d(64),
            ReLU(),
        )

        self.R1 = ReLU()

        self.model2 = Sequential(
            # 1.2
            Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            BatchNorm1d(64),
            ReLU(),
            Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            BatchNorm1d(64),
            ReLU(),
        )

        self.R2 = ReLU()

        self.model3 = Sequential(
            # 2.1
            Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            BatchNorm1d(128),
            ReLU(),
            Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            BatchNorm1d(128),
            ReLU(),
        )
        self.en1 = Sequential(
            Conv1d(in_channels=64, out_channels=128, kernel_size=1, stride=2, padding=0),
            BatchNorm1d(128),
            ReLU(),
        )
        self.R3 = ReLU()

        self.model4 = Sequential(
            # 2.2
            Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            BatchNorm1d(128),
            ReLU(),
            Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            BatchNorm1d(128),
            ReLU(),
        )
        self.R4 = ReLU()

        self.model5 = Sequential(
            # 3.1
            Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            BatchNorm1d(256),
            ReLU(),
            Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            BatchNorm1d(256),
            ReLU(),
        )
        self.en2 = Sequential(
            Conv1d(in_channels=128, out_channels=256, kernel_size=1, stride=2, padding=0),
            BatchNorm1d(256),
            ReLU(),
        )
        self.R5 = ReLU()

        self.model6 = Sequential(
            # 3.2
            Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            BatchNorm1d(256),
            ReLU(),
            Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            BatchNorm1d(256),
            ReLU(),
        )
        self.R6 = ReLU()

        self.model7 = Sequential(
            # 4.1
            Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            BatchNorm1d(512),
            ReLU(),
            Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            BatchNorm1d(512),
            ReLU(),
        )
        self.en3 = Sequential(
            Conv1d(in_channels=256, out_channels=512, kernel_size=1, stride=2, padding=0),
            BatchNorm1d(512),
            ReLU(),
        )
        self.R7 = ReLU()

        self.model8 = Sequential(
            # 4.2
            Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            BatchNorm1d(512),
            ReLU(),
            Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            BatchNorm1d(512),
            ReLU(),
        )
        self.R8 = ReLU()

        # AAP 自适应平均池化
        self.aap = AdaptiveAvgPool1d(1)
        # flatten 维度展平
        self.flatten = Flatten(start_dim=1)
        # FC 全连接层
        self.fc = Linear(512, num_classes)

    def forward(self, x):
        x = self.model0(x)

        f1 = x
        x = self.model1(x)
        x = x + f1
        x = self.R1(x)

        f1_1 = x
        x = self.model2(x)
        x = x + f1_1
        x = self.R2(x)

        f2_1 = x
        f2_1 = self.en1(f2_1)
        x = self.model3(x)
        x = x + f2_1
        x = self.R3(x)

        f2_2 = x
        x = self.model4(x)
        x = x + f2_2
        x = self.R4(x)

        f3_1 = x
        f3_1 = self.en2(f3_1)
        x = self.model5(x)
        x = x + f3_1
        x = self.R5(x)

        f3_2 = x
        x = self.model6(x)
        x = x + f3_2
        x = self.R6(x)

        f4_1 = x
        f4_1 = self.en3(f4_1)
        x = self.model7(x)
        x = x + f4_1
        x = self.R7(x)

        f4_2 = x
        x = self.model8(x)
        x = x + f4_2
        x = self.R8(x)

        # 最后3个
        x = self.aap(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    #trans = transforms.Compose([ transforms.ToTensor(), ])
    input = torch.randn(3,1,750)
    print(input.shape)
    input_shape = (1, 750)  # 请替换为实际的输入形状
    classes = 95  # 请替换为实际的类别数
    net = burstVarCNN(input_shape,classes)
    print(input)
    
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
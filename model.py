import torch
from torch import nn
from torchsummary import summary


class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):  # c1-4 为各分支卷积核数量
        super(Inception, self).__init__()

        self.ReLU = nn.ReLU()  # 激活函数

        # 路线一 单个 1x1 卷积
        self.p1_1 = nn.Conv2d(in_channels=in_channels, out_channels=c1, kernel_size=1, stride=1, padding=0)

        # 路线二 单个 1x1 卷积 + 单个 3x3 卷积
        self.p2_1 = nn.Conv2d(in_channels=in_channels, out_channels=c2[0], kernel_size=1, stride=1, padding=0)
        self.p2_2 = nn.Conv2d(in_channels=c2[0], out_channels=c2[1], kernel_size=3, stride=1, padding=1)

        # 路线三 单个 1x1 卷积 + 单个 5x5 卷积
        self.p3_1 = nn.Conv2d(in_channels=in_channels, out_channels=c3[0], kernel_size=1, stride=1, padding=0)
        self.p3_2 = nn.Conv2d(in_channels=c3[0], out_channels=c3[1], kernel_size=5, stride=1, padding=2)

        # 路线四 单个 3x3 最大池化 + 单个 1x1 卷积
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels=in_channels, out_channels=c4, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        p1 = self.ReLU(self.p1_1(x))
        p2 = self.ReLU(self.p2_2(self.ReLU(self.p2_1(x))))
        p3 = self.ReLU(self.p3_2(self.ReLU(self.p3_1(x))))
        p4 = self.ReLU(self.p4_2(self.p4_1(x)))

        return torch.cat((p1, p2, p3, p4), dim=1)




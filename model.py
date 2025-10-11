import torch
from torch import nn
from torch.nn import MaxPool2d, Conv2d, Linear, Flatten, Sequential

# 搭建神经网络
class self_nn_Module(nn.Module):

    def __init__(self):
        super(self_nn_Module,self).__init__()
        self.conv1 = Conv2d(3,32,5,stride=1,padding=2)
        #输入通道，输出通道，核大小，步长，边界填充，扩大，组，偏差，边界填充模式
        #in_channel,out_channel,kernel_size,stride,padding,dilation,groups ,bias,padding_mode
        self.maxpool1 = MaxPool2d(2)
        self.conv2 = Conv2d(32, 32, 5, stride=1, padding=2)
        self.maxpool2 = MaxPool2d(2)
        self.conv3 = Conv2d(32, 64, 5, stride=1, padding=2)
        self.maxpool3 = MaxPool2d(2)
        self.flatten = Flatten()
        self.linear1 = Linear(in_features= 1024,out_features=64)
        #in_features,out_features,bias y = A*x+b
        self.linear2 = Linear(in_features=64,out_features=10)

        # 或者这么写，利用Sequential，比较直观简单
        self.model1 = Sequential(
            Conv2d(3, 32, 5, stride=1, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, stride=1, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, stride=1, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(in_features=1024, out_features=64),
            Linear(in_features=64, out_features=10)
        )
    def forward(self,x):
        x = self.model1(x)
        return x

if __name__ == "__main__":
    # 验证正确性
    s1 = self_nn_Module()
    input = torch.ones((64,3,32,32))
    output = s1(input)
    print(output.shape)
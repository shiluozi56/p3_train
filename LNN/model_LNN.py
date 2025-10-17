import torch
from torch import nn
from torch.nn import MaxPool2d, Conv2d, Linear, Flatten, Sequential

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


# ======================
# 液体时间常数单元
# ======================
class LiquidTimeConstantCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        #input_size表示每个输入样本的特征维度（特征数量）
        #表示隐藏层中神经元的数量（或隐藏状态的维度）
        super().__init__()
        self.W = nn.Parameter(torch.randn(hidden_size, input_size))

        # nn.Parameter 标记这是一个可学习参数，会在训练过程中被优化更新
        # torch.randn(hidden_size, input_size) 生成一个形状为 (hidden_size, input_size) 的张量
        # 其元素服从标准正态分布（均值为 0，标准差为 1）
        self.U = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.tau = nn.Parameter(torch.ones(hidden_size))  # 时间常数
        # tau大，神经元对历史信息记得久；tau小，神经元忘得快。
        print(f"slef.W:  {self.W}")
        print(f"slef.U:  {self.U}")
    def forward(self, x, h):
        # dx/dt = -(h - f(x,h))/tau
        # 先将输入 x 与权重矩阵 self.W 的转置相乘（x @ W^T）
        # 再将隐藏状态 h 与权重矩阵 self.U 的转置相乘（h @ U^T）
        # 最后将两个结果相加，得到预激活值 pre_act 是前一步计算的预激活值（通常是输入和上一状态的线性组合）
        # 输入和历史信息 “混合” 成pre_act（预激活信号）
        pre_act = torch.matmul(x, self.W.T) + torch.matmul(h, self.U.T)
        # tanh(pre_act)是f(x,h)  是 PyTorch 中的双曲正切激活函数，作用是将输入张量的数值压缩到 [-1, 1] 区间内
        dh = (-h + torch.tanh(pre_act)) / self.tau#状态变化量 dh，它表示当前状态应该如何调整
        # tanh把信号压缩到 - 1到 1 之间 模拟神经元的 “激活 通过tau调整更新幅度，得到新的神经元状态h_next。
        h_next = h + dh #下一时刻的隐藏状态 h_next
        print("\n")
        print(f"pre_act:  {pre_act}")
        print(f"dh:  {dh}")
        print(f"tanh(pre_act):  {torch.tanh(pre_act)}")
        print(f"h_next:  {h_next}")
        return h_next

# ======================
# 液体神经网络模型
# ======================
# 一步都用当前输入和历史状态更新神经元，再生成预测
class LiquidNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.cell = LiquidTimeConstantCell(input_size, hidden_size)#------------------------------------------
        self.fc = nn.Linear(hidden_size, output_size)#全连接层（线性层），
        #用于将神经网络的隐藏层输出转换为最终的输出结果。

    def forward(self, x):
        # h被初始化为一个形状为(batch_size, hidden_size)的全零张量
        # self.cell.W.size(0) 表示获取细胞（cell）中权重矩阵 W 的第 0 维大小，实际就是隐藏层维度（hidden_size）
        h = torch.zeros(x.size(0), self.cell.W.size(0),device=x.device)
        print(f"h:  {h}")
        outputs = []
        for t in range(x.size(1)): #x.size(1) 表示输入张量 x 的第 1 维大小，通常对应序列长度（seq_len），循环遍历序列中的每个时间步
            h = self.cell(x[:, t, :], h)# 通过模型的细胞单元（如 LNN 细胞），用当前时间步输入和上一时刻隐藏状态 h 计算新的隐藏状态 h
            y_t = self.fc(h)  # 每个时间步预测，通过全连接层将隐藏状态 h 映射为当前时间步的输出 y_t
            outputs.append(y_t.unsqueeze(1)) #收集每个时间步的预测结果
            # unsqueeze(1) 会在第 1 个维度（索引从 0 开始）插入一个新的维度，让 y_t 的形状变成 [batch_size, 1, output_size]
        return torch.cat(outputs, dim=1)  #  沿着第 1 维（时间步维度）拼接所有结果，最终输出形状为 [batch, seq_len, output_size]
# 通过循环迭代处理每个时间步的输入，利用隐藏状态传递历史信息，并在每个时间步生成对应的预测，最终输出完整的序列预测结果。



if __name__ == "__main__":
    # 验证正确性
    s1 = self_nn_Module()
    input = torch.ones((64,3,32,32))
    output = s1(input)
    print(output.shape)
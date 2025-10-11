import torchvision.datasets
from torch.utils.data import DataLoader

from model import *
from torch.utils.tensorboard import SummaryWriter


#加载数据集（无需修改）
train_data = torchvision.datasets.CIFAR10(root="./dataset",train=True,transform=torchvision.transforms.ToTensor(),download=True)
test_data = torchvision.datasets.CIFAR10(root="./dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)

print("训练集的长度为{}".format(len(train_data)))
print("测试集的长度为{}".format(len(test_data)))

# dataloader加载数据
train_dataloader = DataLoader(train_data,batch_size = 64)
test_dataloader = DataLoader(test_data,batch_size = 64)


# 创建网络模型
s1 = self_nn_Module()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(s1.parameters(),lr=learning_rate)


# 设置训练网络参数
# 训练次数
total_train_step = 0
# 测试次数
total_test_step = 0
# 训练轮数
epoch = 10

# 添加tensorboard
# 写入图片
writer = SummaryWriter("logs")
# 若无法将“tensorboard”项识别：执行python -m tensorboard.main --logdir logs --port=6007***************************


# 开始训练
for i in range (epoch):
    print("第{}轮训练开始----------------------------------".format(i+1))

    s1.train()
    for data in train_dataloader: # 遍历训练集中的每一批数据
        # 每批数据经过 “前向传播计算损失→反向传播求梯度→优化器更新参数” 的循环，逐步降低损失。
        imgs,targets = data # 解包：imgs为图像张量，targets为标签（0-9的整数）

        outputs = s1(imgs)# 前向传播：模型输出预测结果（形状为[batch_size, 10]）
        loss = loss_fn(outputs,targets) # 计算当前批次的损失

        #优化器优化模型
        optimizer.zero_grad()# 清零梯度（避免累积）
        loss.backward()# 反向传播，计算梯度
        optimizer.step() # 根据梯度更新模型参数

        # 记录并打印训练信息
        total_train_step +=1
        print("训练次数:{},loss:{}".format(total_train_step,loss))
        writer.add_scalar("train_loss",loss.item(),total_train_step)# 记录训练损失

    total_test_loss = 0 # 累计测试损失
    total_accuracy = 0 # 累计正确预测数

    # 开始测试步骤
    s1.eval()
    with torch.no_grad():#禁用梯度计算，减少内存占用并加速测试过程。
        for data in test_dataloader:# 遍历测试集中的每一批数据
            imgs, targets = data

            outputs = s1(imgs) # 前向传播（无梯度计算）
            loss = loss_fn(outputs, targets) # 计算测试损失
            total_test_loss += loss.item() # 累加损失
            # 计算准确率：outputs.argmax(1)取预测概率最大的类别索引，与targets比较
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy
    print("整体测试集上的loss:{}".format(total_test_loss))
    print("整体测试集上的正确率:{}".format(total_accuracy/len(test_data)))# 总准确率 = 正确数 / 测试集总样本数
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    total_test_step += 1

    torch.save(s1,"my_train_{}.pth".format(i))# 保存当前轮次的模型
    print("模型已保存 ")

writer.close()



#GPU 网络模型 数据（输入、标注） 损失函数 调用.cuda()
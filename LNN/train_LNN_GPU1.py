import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from LNN.dataloader import StockDataset
from LNN.model_LNN import LiquidNN

def s1_LNN_train(data_selected,data_selected_season):
    # 定义设备
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")  # 打印当前使用的设备，方便确认

    # spring summer autumn winter
    if not os.path.exists("best_model_1840"):
        os.makedirs("best_model_1840")

    train_data = StockDataset(
        'data_process_mult_processed/' + data_selected + '/' + data_selected + '_' + data_selected_season + '.csv', 10,
        is_test=False)
    test_data = StockDataset(
        'data_process_mult_processed/' + data_selected + '/' + data_selected + '_' + data_selected_season + '.csv', 10,
        is_test=True)

    train_loader = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=2)

    print("训练集的长度为{}".format(len(train_data)))
    print("测试集的长度为{}".format(len(test_data)))


    # 创建网络模型

    input_size = 11
    hidden_size = 8
    num_heads = 2
    s1 = LiquidNN(input_size, hidden_size, num_heads).to(device)
    print("===== Model Structure =====")
    print(f"这是{data_selected}_{data_selected_season}.")
    print(s1)
    print("===========================")
    # 损失函数
    loss_fn = nn.MSELoss().to(device)


    # 优化器
    learning_rate = 0.01
    optimizer = torch.optim.Adam(s1.parameters(),lr=learning_rate)
    #使用 Adam 优化器来更新模型参数，传入模型中所有可学习的参数（如权重矩阵、时间常数等）自动调整模型的W、U、tau等参数，让损失越来越小 学习率0.01

    total_start_time = time.time()
    best_test_loss = float("inf")
    train_losses = []
    test_losses = []
    epoch_times = []

    # 训练轮数
    epoch = 30

    # 添加tensorboard
    # 写入图片
    writer = SummaryWriter("logs")
    # 若无法将“tensorboard”项识别：执行python -m tensorboard.main --logdir logs --port=6007***************************


    # 开始训练
    for i in range (epoch):
        print("第{}轮训练开始----------------------------------".format(i+1))
        epoch_start_time = time.time()

        s1.train()
        train_loss = 0.0

        for data,label in train_loader: # 遍历训练集中的每一批数据
            # 每批数据经过 “前向传播计算损失→反向传播求梯度→优化器更新参数” 的循环，逐步降低损失。
            optimizer.zero_grad()# 清零梯度（避免累积）

            data = data.to(device)
            label = label.to(device)
            out = s1(data)
            loss = loss_fn(out, label)

            loss.backward()# 反向传播，计算梯度
            optimizer.step() # 根据梯度更新模型参数
            train_loss += loss.item()

            imgs,targets = data # 解包：imgs为图像张量，targets为标签（0-9的整数）

            imgs = imgs.to(device) #输入
            targets = targets.to(device) #标签

            outputs = s1(imgs)
            loss = loss_fn(outputs,targets) # 计算当前批次的损失
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)


        # 开始测试步骤
        s1.eval()
        test_loss = 0.0
        with torch.no_grad():#禁用梯度计算，减少内存占用并加速测试过程。
            for data in test_loader:# 遍历测试集中的每一批数据
                out = s1(data)
                loss = loss_fn(out, label)
                test_loss += loss.item()
        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)

        print(f"Epoch {epoch + 1}/{epoch}, "
              f"Train Loss: {avg_train_loss:.12f}, "
              f"Test Loss: {avg_test_loss:.12f}, "
              f"Time: {epoch_time:.2f} sec")

        # 保存最佳模型
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(s1.state_dict(), f"best_model_1840/{data_selected}/s1_best_{data_selected_season}.pth")
            print(f"Saved best model at epoch {epoch + 1}, Test Loss: {best_test_loss:.12f}")


    total_time = time.time() - total_start_time

    print(f"Training finished in {total_time:.2f} seconds")
    print(f"Best test loss: {best_test_loss:.6f}")


    writer.close()



    #GPU 网络模型 数据（输入、标注） 损失函数 这些要调用.cuda()


if __name__ == '__main__':
    s1_LNN_train("f1","spring")
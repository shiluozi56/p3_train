import os
import time

import numpy as np
import pandas as pd
import torch
import torchvision.transforms
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from torch import nn
from torch.utils.data import DataLoader

from LNN.dataloader import StockDataset
from LNN.model_LNN import LiquidNN


def test(data_selected,data_selected_season):
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")  # 打印当前使用的设备，方便确认

    # spring summer autumn winter
    if not os.path.exists("best_model_1840"):
        os.makedirs("best_model_1840")
    if not os.path.exists("result_picture_1840"):
        os.makedirs("result_picture_1840")


    # 加载测试数据
    test_data = StockDataset(
        'data_process_mult_processed/' + data_selected + '/' + data_selected + '_' + data_selected_season + '.csv', 10,
        is_test=True)
    test_loader = DataLoader(test_data, batch_size=11, shuffle=False, num_workers=2)

    print("测试集的长度为{}".format(len(test_data)))

    # params = torch.load(f"best_model_1840/{data_selected}/s1_best_spring.pth")
    params = torch.load(f"best_model_1840/f1/s1_best_spring.pth")
    input_size = 11
    hidden_size = 11
    output_size  = 1
    s1 = LiquidNN(input_size, hidden_size, output_size).to(device)

    s1.load_state_dict(params)
    s1.eval()
    loss_fn = nn.MSELoss().to(device)

    y_gt = [] #真实值（y_gt）：从test_loader中直接获取标签label，经过反归一化后得到实际业务数值（如真实发电量）
    y_pred = [] #通过加载训练好的LiquidNN模型（s1）对测试数据data进行推理，输出模型预测结果，再经过反归一化得到预测的业务数值。
    batch_times = []
    eval_loss = 0.0

    # 批量推理
    with torch.no_grad():
        for data, label in test_loader:
            batch_start_time = time.time()
            data = data.to(device)
            label = label.to(device)
            out = s1(data)
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)

            loss = loss_fn(out, label)
            eval_loss += loss.item()
            y_gt += label.numpy().squeeze(axis=1).tolist()
            y_pred += out.numpy().squeeze(axis=1).tolist()


    # 转换为 numpy
    y_gt = np.array(y_gt).reshape(-1, 1)
    y_pred = np.array(y_pred).reshape(-1, 1)


    # 根据用户选择决定是否反归一化 此处默认选择为是
    scaler = test_data.scaler
    dummy = np.zeros((y_gt.shape[0], test_data.amount_of_features))
    dummy_gt = dummy.copy()
    dummy_pred = dummy.copy()
    dummy_gt[:, -1] = y_gt[:, 0]
    dummy_pred[:, -1] = y_pred[:, 0]
    y_gt_plot = scaler.inverse_transform(dummy_gt)[:, -1]
    y_pred_plot = scaler.inverse_transform(dummy_pred)[:, -1]

    # 绘制前1000步对比图
    plt.figure(figsize=(12, 6))
    draw = pd.DataFrame({"Real": y_gt_plot, "Predicted": y_pred_plot})
    plt.plot(draw.iloc[0:1000]['Real'], label='Real', color='blue', linewidth=2.0)
    plt.plot(draw.iloc[0:1000]['Predicted'], label='Predicted', color='red', linestyle='--', linewidth=2.0)
    plt.legend(loc='upper right', fontsize=12, shadow=True, fancybox=True)
    plt.title(f"s1 Test Data Comparison", fontsize=20, fontweight='bold')
    plt.xlabel('Time', fontsize=15)
    plt.ylabel('Value', fontsize=15)
    plt.grid(True)
    plt.savefig(f"result_picture_1840/{data_selected}/s1_fic_{data_selected}_{data_selected_season}.jpg")




if __name__ in '__main__':
    test("f1", "spring")
    test("f1","summer")
    test("f1","autumn")
    test("f1", "winter")

    test("f2", "spring")
    test("f2","summer")
    test("f2","autumn")
    test("f2", "winter")

    test("f3", "spring")
    test("f3","summer")
    test("f3","autumn")
    test("f3", "winter")

    test("f4", "spring")
    test("f4","summer")
    test("f4","autumn")
    test("f4", "winter")


import torch

# 对应保存方式1 读取模型
model1 = torch.load("vgg16_method1.pth")
print(model1)

# 对应保存方式2 读取模型
model2 = torch.load("vgg16_method2.pth")
print(model2)
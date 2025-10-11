import torch
import torchvision.transforms
from PIL import Image

img_path = "./test/1.png"
img = Image.open(img_path)
img = img.convert("RGB")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize((32,32)),
     torchvision.transforms.ToTensor()
    ]
)

img = transform(img)
print(img.shape)

# 对应保存方式1 读取模型
model1 = torch.load("my_train_9.pth").to(device)
print(model1)

img = torch.reshape(img,(1,3,32,32),).to(device)
model1.eval()
with torch.no_grad():
    output = model1(img)
print(output)
print(output.argmax(1))
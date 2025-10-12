import torch
import torchvision.transforms
from PIL import Image

for i in range(7):

    try:
        img_path = "./test/" + str(i + 1) + ".png"
        img = Image.open(img_path)
    except FileNotFoundError:
        img_path = "./test/" + str(i + 1) + ".jpg"
        img = Image.open(img_path)
        # print("无png 尝试搜索jpg")
    else:
        #  如果没有异常执行这块代码
        pass


    img = img.convert("RGB")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((32,32)),
         torchvision.transforms.ToTensor()
        ]
    )

    img = transform(img)
    # print(img.shape)

    # 对应保存方式1 读取模型
    model1 = torch.load("my_train_fin.pth").to(device)
    # print(model1)
    list = ["飞机", "汽车", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车"]
    img = torch.reshape(img,(1,3,32,32),).to(device)
    model1.eval()
    with torch.no_grad():
        output = model1(img)
    # print(output)
    # print(output.argmax(1))
    print(f"第{i+1}张图片识别为{list[output.argmax(1)]}")

import torch
import copy
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.utils.data as Data
from model import GoogLeNet, Inception
from torchvision.datasets import FashionMNIST
from torchvision import transforms


def train_val_data_process():
    train_data = FashionMNIST(root='./data',
                              train=True,
                              transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]),
                              download=True)

    train_data, val_data = Data.random_split(train_data, [round(0.8 * len(train_data)), round(0.2 * len(train_data))])

    train_dataloader = Data.DataLoader(dataset=train_data,
                                       batch_size=32,
                                       shuffle=True,
                                       num_workers=2)

    val_dataloader = Data.DataLoader(dataset=val_data,
                                     batch_size=32,
                                     shuffle=True,
                                     num_workers=2)

    return train_dataloader, val_dataloader


def train_model_process(model, train_dataloader, val_dataloader, num_epochs):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    criterion = torch.nn.CrossEntropyLoss()

    model = model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())

    best_acc = 0.0

    train_loss_all = []
    train_acc_all = []

    val_loss_all = []
    val_acc_all = []

    since  = time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-----------------------------------------------')

        train_loss = 0.0
        train_correct = 0

        val_loss = 0.0
        val_correct = 0

        train_num = 0
        val_num = 0

        for step, (b_x, b_y) in enumerate(train_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            model.train()

            output = model(b_x)

            pre_lab = torch.argmax(output, dim=1)

            loss = criterion(output, b_y)

            # 将梯度初始化为 0
            optimizer.zero_grad()

            # 反向传播计算
            loss.backward()

            # 根据反向传播的梯度信息更新网络的参数
            optimizer.step()

            train_loss += loss.item() * b_x.size(0)

            train_correct += torch.sum(pre_lab == b_y.data)

            train_num += b_x.size(0)

        for step, (b_x, b_y) in enumerate(val_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            model.eval()

            output = model(b_x)

            pre_lab = torch.argmax(output, dim=1)

            loss = criterion(output, b_y)

            val_loss += loss.item() * b_x.size(0)

            val_correct += torch.sum(pre_lab == b_y.data)

            val_num += b_x.size(0)

        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_correct.double().item() / train_num)

        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_correct.double().item() / val_num)

        print("{} Train Loss:{:.4f} Train Accuracy:{:.4f}".format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print("{}   Val Loss:{:.4f}   Val Accuracy:{:.4f}".format(epoch, val_loss_all[-1], val_acc_all[-1]))

        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())

        time_used = time.time() - since
        print("训练和验证耗费的时间{:.0f}m{:.0f}s".format(time_used // 60, time_used % 60))

    model.load_state_dict(best_model_wts)
    torch.save(best_model_wts, "D:\\PytorchProject\\GoogLeNet\\best_model.pth")

    train_process = pd.DataFrame(data={"epoch": range(num_epochs),
                                       "train_loss_all": train_loss_all,
                                       "val_loss_all": val_loss_all,
                                       "train_acc_all": train_acc_all,
                                       "val_acc_all": val_acc_all})

    return train_process


def matplot_acc_loss(train_process):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process["epoch"], train_process.train_loss_all, "ro-", label="Train Loss")
    plt.plot(train_process["epoch"], train_process.val_loss_all, "bs-", label="Val Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process.train_acc_all, "ro-", label="Train Accuracy")
    plt.plot(train_process["epoch"], train_process.val_acc_all, "bs-", label=" Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


if __name__ == '__main__':

    GoogLeNet = GoogLeNet(Inception)

    train_dataloader, val_dataloader = train_val_data_process()

    train_process = train_model_process(GoogLeNet, train_dataloader, val_dataloader, num_epochs=20)

    matplot_acc_loss(train_process)



import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import GoogLeNet, Inception


def test_data_process():
    test_data = FashionMNIST(root='./data',
                             train=False,
                             transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]),
                             download=True)

    test_dataloader = Data.DataLoader(dataset=test_data,
                                      batch_size=1,
                                      shuffle=True,
                                      num_workers=2)

    return test_dataloader


def test_model_process(model, test_dataloader):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    test_correct = 0
    test_num = 0

    # 类别 FashionMNIST 数据中每个类别的标签
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)

            model.eval()

            output = model(test_data_x)

            pre_lab = torch.argmax(output, dim=1)

            pre_val = pre_lab.item()

            ground_truth = test_data_y.item()

            print("预测值：", pre_val, "------", "真实值：", ground_truth)
            print("预测值：", classes[pre_val], "------", "真实值：", classes[ground_truth])

            test_correct += torch.sum(pre_lab == test_data_y.data)

            test_num += test_data_x.size(0)

    test_acc = test_correct.double().item() / test_num
    print("测试的准确率为：", test_acc)


if __name__ == '__main__':

    GoogLeNet = GoogLeNet(Inception)

    GoogLeNet.load_state_dict(torch.load('best_model.pth'))

    test_dataloader = test_data_process()

    test_model_process(GoogLeNet, test_dataloader)














import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import ImageFolder
from cat_dog_classification.cat_dog_model import GoogLeNet, Inception


def test_data_process():
    # 数据集路径
    test_data_path = r'D:\PytorchProject\GoogLeNet\cat_dog_classification\dataset\test'

    normalize = transforms.Normalize(mean=[0.162, 0.151, 0.138], std=[0.058, 0.052, 0.048])

    # 数据集预处理组合
    test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])

    # 加载数据集
    test_data = ImageFolder(test_data_path, transform=test_transform)

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

    classes = ['猫', '狗']

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














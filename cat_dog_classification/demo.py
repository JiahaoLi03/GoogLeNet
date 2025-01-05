import torch
from torchvision import transforms
from cat_dog_classification.cat_dog_model import GoogLeNet, Inception
from PIL import Image


def image_preprocess(image):

    normalize = transforms.Normalize(mean=[0.162, 0.151, 0.138], std=[0.058, 0.052, 0.048])

    demo_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])

    image = demo_transform(image)

    # 添加批次维度
    image = image.unsqueeze(0)  # shape: torch.Size([1, 3, 224, 224]) bitch_size, c, h, w

    return image


def predict(model, image):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    classes = ['猫', '狗']

    with torch.no_grad():

        model.eval()

        image = image.to(device)

        output = model(image)

        pre_lab = torch.argmax(output, dim=1)
        pre_val = pre_lab.item();
        print("预测值：{} ------ 它是一只{}。".format(pre_val, classes[pre_val]))


if __name__ == '__main__':

    GoogLeNet = GoogLeNet(Inception)

    GoogLeNet.load_state_dict(torch.load('best_model.pth'))

    image = Image.open('dog.jpg')

    image = image_preprocess(image)

    predict(GoogLeNet, image)


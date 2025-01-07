## GoogLeNet

- 利用 **FashionMNIST** 数据集（内置数据集），如需手动下载请访问 [FashionMNIST](https://pan.baidu.com/s/1dXc8cpsefWoUGaI3_bTRVQ?pwd=6666)。
- `python train.py` 训练模型，运行后会产生 `data`文件夹，放置数据集。
- `python test.py`  测试模型



#### 训练阶段的 `Train Loss` 、 `Val Loss` 、`Train Accuracy` 、 `Val Accuracy`

![](https://res.cloudinary.com/qlyfdljh/image/upload/v1735983205/GoogLeNet/GoogLeNet_Loss_Accuracy.png)

#### 测试阶段的效果图
![](https://res.cloudinary.com/qlyfdljh/image/upload/v1735983421/GoogLeNet/test.png)

#### 猫狗分类 `cat_dog_classification`
- `data_cat_dog` 数据集获取[data_cat_dog](https://pan.baidu.com/s/1FK89a71hbXjSlHzLgDfE7g?pwd=6666)
- `python data_partitioning.py` 预处理数据集
- `python train_cat_dog.py` 训练模型
- `python test_cat_dog.py`  测试模型

#### 训练阶段的 `Train Loss` 、 `Val Loss` 、`Train Accuracy` 、 `Val Accuracy`
![](https://res.cloudinary.com/qlyfdljh/image/upload/v1736264030/GoogLeNet/GoogLeNet_Cat_Dog_Loss_Accuracy.png)

#### 测试阶段的效果图
![](https://res.cloudinary.com/qlyfdljh/image/upload/v1736264692/GoogLeNet/cat_dog_test.png)
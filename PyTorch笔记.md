

## PyTorch笔记

## 1. 基本概念：

Torch，火炬。在深度学习中，PyTorch用于nparray的gpu计算。

------

##### <font color='green'>pytorch-gpu版本下载：</font>

- 需要安装CUDA和CUDnn。CUDA（Compute Unified Device Architecture，统一计算架构），CUDnn（CUDA Deep Neural Network library），英伟达针对深度学习所打造的gpu加速库。
- **要注意的是**，pytorch-gpu不同版本号所要求的CUDA版本不同，而CUDA的版本环境跟本身电脑显卡驱动的版本相关，再者，CUdnn的版本和CUDA的版本相关。因此，我们首先要根据自己显卡的驱动版本，确定CUDA环境，再确定对应的pytorch版本。但是，pytorch-gpu目前不支持较旧的cuda版本，因此下载安装时要提前留意一下。

------



### 1.1 数据结构：

​	pytorch中常用的数据结构为tensor（张量）。相关语法如下所示。（可与nparray、Seriers、DataFrame等做对比）

#### 	1.1.1 创建tensor:

```python
#标准方法，里面的参数自由发挥。
x = torch.tensor（[[1,2,3],[4,5,6]]）

#返回一个张量，包含了从区间[0, 1)的均匀分布中抽取的一组随机数。张量的形状由参数sizes定义
x = torch.rand(5,3)

# 返回一个和输入大小相同的张量，其由均值为0、方差为1的标准正态分布填充。
x = torch.randn_like(x,dtype = torch.float)

#返回一个张量，包含了从标准正态分布（均值为0，方差为1，即高斯白噪声）中抽取的一组随机数。张量的形状由参数sizes定义
a = torch.randn(4,4)  

# 返回一个与size大小相同的用1填充的张量。 默认返回的Tensor具有与此张量相同的torch.dtype和torch.device）
x = x.new_ones(3,3,dtype = torch.double)
```



#### 1.1.2 查看tensor的信息。

与pandas语法相似，假设存在tensor：x，则**x.dim()\size()\type**等语句可以查看tensor的维度、规模、数据格式（与nparray略微不同，np.shape(x)/np.ndim(x)等等）；或类似索引的方式：x[1,1]、x[:,5]查看tensor中的某些数据项。



#### 1.1.3 改变tensor

view操作，可以改变tensor的shape。对比np.arrange(12).reshape(3,4)。

**（*问题：对比numpy和pandas，tensor是否存在合并、拆分等相关操作？*）**

```python
#返回一个张量，包含了从标准正态分布（均值为0，方差为1，即高斯白噪声）中抽取的一组随机数。张量的形状由参数sizes定义
a = torch.randn(4,4)  
print(a)

#返回一个张量，包含了从区间[0, 1)的均匀分布中抽取的一组随机数。张量的形状由参数sizes定义
b = torch.rand(4,4)   
print(b)

c = a.view(16)
print(c)

d = a.view(2,8)  #x.view(-1,11):-1的含义是根据自动修改
print(d)
```

------

**<font color='red'>解决问题：tensor是否存在合并、拆分等相关操作？：</font>**





------



#### 1.1.4 nparray与tensor的互换:

存在tensor：a ，a.numpy()可以将其转化为numpy格式

存在nparray: b，torch.from_numpy(b)可以将其转化为tensor格式

看起来torch牛逼一点啊。

```python
#tensor ——> nparray
a = torch.ones(3,3)
b = a.numpy()

#nparray ——> tensor
c = torch.from_numpy(b)
```



### 1.2 计算梯度

这部分内容，涉及神经网络的基础知识，可查看相关笔记来了解。在pytorch中，可以自动计算梯度（反向传播），PS：体现了人与机器的差别，果然计算机是用来计算的。

相关语法如下所示：假设存在由损失函数得出的：loss，反向传播的方法为：loss.backward（）

```python
loss.backward() #可以联想神经网络model中的forward（）方法
```

其中，又涉及自动求导这一知识点。假设存在以下张量。注意参数：requires_grad=True，含义是可以求导;w.gard()：查看w的导数

```python
x = torch.randn(1,requires_grad=True)
w = torch.randn(1,requires_grad=True)
z = w * x
b = torch.randn(1,requires_grad=True)
y =z + b
y.backward()  #没有提及参数requires_grad=True，自动求导；要注意的是，backward（）方法是会累加导数值的。
w.grad 
```

backward（）方法会自动累加所求结果。在神经网络的计算过程中要使用优化器optimizer对导数清零。

```python
optim.grad_zero()

#补充,存在问题：优化器是如何运行的？
optim.step()
```

------

**<font color='red'>解决问题：优化器是如何运行的？</font>**

https://blog.csdn.net/PanYHHH/article/details/107361827?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1-107361827-blog-99963586.pc_relevant_multi_platform_whitelistv3&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1-107361827-blog-99963586.pc_relevant_multi_platform_whitelistv3&utm_relevant_index=1

------



## 2.应用实例： 

### 2.1 简单的线性回归模型：

#### 2.1.1 网络结构：

首先，关于线性回归模型的基础概念，可查看其他笔记进行复习。在pytorch中，我们可以直接利用pytorch的相关工具包，来创建线性回归模型，也可以自己人为定义一个线性回归模型。本次将先使用人工定义的线性回归模型，代码如下所示。

```python
import torch
import torch.nn as nn

class LinearRegressionModel(nn.Module) :
    def __init__(self,input_dim,output_dim) :
        super(LinearRegressionModel,self).__init__()
        self.linear = nn.Linear(input_dim,output_dim)
    def forward(x) :
        out = self.linear(x)
        return out
```

**<font color='green'>相应知识点：</font>**

- 在本例中，我们定义的线性回归模型，继承于<font color='cornflowerblue'>**nn.Module**</font>（大部分功能都是由nn.Module提供，pytorch牛逼之处，后面可以直接用Module工具包使用线性回归模型，不用自己创建了。再额外说一句，不止是线性回归模型，像Alexnet、VGGnet、Resnet等不需人为定义，可以直接使用。）
- 在本例中，涉及到了python的继承这一特点。在此简单介绍一下：super(LinearRegressionModel,self).__init__(),作用是初始化继承于父类的属性。
-  self.linear = nn.Linear(input_dim,output_dim)这一条语句可以理解为一个“神经元”，专业术语是线性层，就是线性回归模型。它的参数主要有两个，分别是in参数和out参数。顾名思义，in表示上一层的神经元个数（可理解为输入的个数），out表示本层的神经元个数（可理解为输出的个数）。
- 关于 forward(x)这个方法，它的作用是计算前向传播，里面的代码说明linear这个属性具有前向传播的功能（问题：nn.Linear()返回的是一个什么东西？）



------

**<font color='red'>解决问题：nn.Linear()返回了什么？：</font>**

![](D:\Picture\StudyPicture\torch.nn.Linear用法介绍.png)

------

#### 2.2.2 模型训练

在进行训练前，要确定相关的参数。在该线性回归模型中，假设模型的形状为：input_dim和output_dim都分别1.在此基础上，我们还要确定相关参数，包括：学习率、损失函数、优化方法、epoch数。

```python
model = LinearRegressionModel(1,1)
epochs = 100
learning_rate = 0.001
criterion = nn.MSEloss()
optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate )
```

------

**<font color='green'>相应知识点：</font>**

- **nn.MSEloss()用法介绍：**<font color='cornflowerblue'>**创建一个均方误差的损失函数,reduction = ‘mean’ 表示求平均**</font>。在本例中损失函数为：criterion，用法是loss = criterion(x，y)。x为预测值，y为真实值。具体说明如下图所示：

- ![nn.MSEloss()用法](D:\Picture\StudyPicture\nn.MSEloss()用法.png)
- **torch.optim.SGD用法介绍：**用SGD方法对损失函数进行优化，找出最优值。具体说明如下图所示：
- ![](D:\Picture\StudyPicture\torch.optim.SGD()用法介绍.png)
- **parameters()用法介绍：**返回模型的参数，一般用作optimizer的参数。具体说明如下所示：
- ![](D:\Picture\StudyPicture\parameters（）用法介绍.png)

------

确定完相关参数后，在进行模型迭代训练开始之前，要对数据进行处理。在本例中，具体代码如下所示。

```python
x_train = np.randn(11,1)
x_valid = np.randn(11,1)
y_train = np.randn(11,1)

inputs = torch.from_numpy(x_train)
labels = torch.from_numpy(y_train)
```

当然，本例中数据预处理较为简单,要注意的是inputs和labels的shape要与模型相匹配。但在其他例子中，我们往往还需要对数据进行特征筛选、数据归一化标准化等。

在处理完数据后，我们开始进行模型迭代训练，流程一般为：前向传播——》梯度清零——》计算loss值——》反向传播——》进行优化（可以联系上面的一个问题：优化器是怎么样运行的）。具体代码如下所示。

```python
for epoch in range(epochs):
    epoch += 1
    outputs = model(inputs)
    optimizer.zero_grad()
    loss = criterion(outputs,labels)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0 :
         # Python2.6 开始，新增了一种格式化字符串的函数 str.format()
        print('epoch {},loss {}'.format(epoch,loss.items))
```



#### 2.2.4 其他相关操作（模型预测、保存、加载）

```python
#模型预测，输入测试集数据，得出预测值
predicted = model(torch.from_numpy(x_train).requires_grad_()).data.numpy()

#保存模型，保存训练好的权值。
torch.save(model.state_dict(), 'model.pkl')

#加载模型，加载训练好的权值。
model.load_state_dict(torch.load('model.pkl'))
```

#### 2.2.5 autograd机制

使用pytorch的优点之一，是autograd机制。下面通过一个例子简单了解一下autograd机制，代码如下所示。

```python
import torch
a = torch.randn(3,4,requires_grad = True)
b = torch.randn(3,4,requires_grad = True)
c = a + b
t = c.sum()
t.backward()
print(a.grad)
```

![autograd机制](D:\Picture\StudyPicture\autograd机制.png)

这里要注意，autograd求导是累加的，因此我们在正式进行训练时需要进行zero_grad操作。autograd的具体详情请看如下网站

https://zhuanlan.zhihu.com/p/69294347

### 2.2 气温预测实例：

假设存在一系列的气温数据，如下图所示。现要求根据这些数据来预测未来几天的气温情况。

![气温预测数据表](D:\Picture\StudyPicture\气温预测数据表.png)

在上表中，具体含义如下所示：

* year,moth,day,week分别表示的具体的时间
* temp_2：前天的最高温度值
* temp_1：昨天的最高温度值
* average：在历史中，每年这一天的平均最高温度值
* actual：这就是我们的标签值了，当天的真实最高温度
* friend：这一列可能是凑热闹的，你的朋友猜测的可能值，咱们不管它就好了



#### 2.2.1 数据预处理

在本例中，我们先省略对该数据的视图化。我们现在要做的如下所示：首先，对该数据表进行编码，因为在week这一列属性中，数值是Monday等不是数值，我们使用独热码对其进行编码。数据编码完毕后，在数据表中挑选出inputs值和labels值。在确定完inputs值和labels值后，对inputs值进行标准化。

```python
from sklearn import preprocessing
features = pd.get_dummies(features)
labels = np.array(features['actual'])
features = features.drop('actual',axis = 1)
features = np.array(features)
input_features = preprocessing.StandareScale().fit_transfrom()
```

------

**<font color='green'>相关知识点</font>**：

- **pd.get_dummies():**对表格内的数据进行独热编码（one-hot shot），让计算机能识别数据的含义。例如：在性别该项属性中，女性可以表示为 is_female:1，而男性则可以表示为：is_female:0

- pandas和numpy的基础语法：略，请自行看其他笔记。

- **数据标准化：**利用sklearn工具包中的preprocessing模块进行数据的标准化处理，其中sklearn工具包中含有机器学习的常见算法。使用preprocessing.StandareScale().fit_transfrom()该方法，其作用是数据标准化。在对训练和测试数据进行标准化的过程中。训练数据，采用fit_transform()函数，测试数据，用transform()。

- **<font color='red'>问题：为什么要使用标准化？</font>**

  输入数据的不同特征值存在不一样的取值范围，不同的量纲。例如特征值x1：1-2（米），特征值x2：100-200（斤），特征值x2对学习结果会产生较大的影响。

####  2.2.2 构建模型

在处理完数据后，我们可以开始构建相应的网络模型。具体代码如下所示：

```python
import torch.nn as nn

input_dim = 14
hidden_dim = 128
output_dim = 1
my_nn = nn.Sequential(
    nn.Linear(input_dim,hidden_dim),
    nn.Sigmoid(),
    nn.Linear(hidden_dim,output_dim)
)
cost = nn.MSEloss(reduction='mean')
optimizer = torch.optim.Adam(my_nn.parameters(),lr = 0.001)
```

------

**<font color='green'>相关知识点：</font>**

- **torch.nn.Sequential:**一个神经网络的容器，我们可以往里面填入所用模型相关的神经网络模块，在该容器中会按顺序构建相应模块。具体信息如下图所示。
- ![](D:\Picture\StudyPicture\torch.nn.Sequential用法介绍.png)
- 

### 2.3 Mnist分类网络

**<font color='red'>学习目标：</font>**

1. 了解Mnist数据集
2. 学习并nn.Module与nn.functional模块（目前所学其他功能模块：torch.optim、nn.MSEloss）
3. 学习并掌握torch.utils.data中的TensorDataset和DataLoader模块

#### 2.3.1 Mnist 数据集介绍

Mnist数据集中文名是手写数字数据集，里面包含很多张手写数字1-9的图像。在本次项目中，我们需要了解相关的文件操作（pathlib模块、gzip模块、pickle模块）和请求-响应操作（requests模块）。相关代码如下所示（知识点其他笔记中提及，略）。

```python
from pathlib import Path
import requests

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)
```

```python
import pickle
import gzip

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
```



#### 2.3.2 nn.Module和nn.functional介绍

**nn.Module模块：**

- Module的意思是组件、模块，顾名思义，就是用一个一个组件构建出复杂的神经网络，在我们要自定义一个神经网络时，需要继承该nn.Module。
- 代码思路，需要考虑两方面内容。一是构造函数，二是重写forwar方法。一般而言，构造函数涉及layer，forward方法将各个layer联系在一起。
- 作用：自定义一个神经网络。常见的自定义方法如下面代码所示。

```python
from torch import nn
import torch.nn.function as F
class Mnist_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(a,b)
        self.hidden2 = nn.Linear(b,c)
        self.out = nn.Linear(c,d)
        
    def forward(self,x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.out(x)
        return x
    
class my_net1(nn.Module) :
    def __init__(self):
        super(my_net1,self).__init__()
        self.conv1_block = nn.Squential(
            nn.Conv2d(in_channels = 1,out_channels = ,kernel_size = ,stride = , padding = )
            nn.ReLU()
            nn.Maxpool2d(2)
        
        )
        
        '''
        super(my_net1,self).__init__()
        self.conv1_block = nn.Squential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(3, 32, 3, 1, 1)),
                    ("relu1", nn.ReLU()),
                    ("pool", nn.MaxPool2d(2))
                ]
            	)
            )
        
        '''
        '''
        def __init__(self):
        super(MyNet, self).__init__()
        self.conv_block=torch.nn.Sequential()
        self.conv_block.add_module("conv1",torch.nn.Conv2d(3, 32, 3, 1, 1))
        self.conv_block.add_module("relu1",torch.nn.ReLU())
        self.conv_block.add_module("pool1",torch.nn.MaxPool2d(2))
 
        self.dense_block = torch.nn.Sequential()
        self.dense_block.add_module("dense1",torch.nn.Linear(32 * 3 * 3, 128))
        self.dense_block.add_module("relu2",torch.nn.ReLU())
        self.dense_block.add_module("dense2",torch.nn.Linear(128, 10))

        '''
        
        
        self.dense = nn.Linear()
    def forward(self,x):  
        x = self.conv1_block(x)
        x = x.view(x.size(0),-1)
        outputs = self.dense(x)
        return outputs
```

**nn.functional模块：**

pytorch常用模块之一，常用于layer中的函数。与nn.Module相比，nn.fuctional往往没有参数需要进行学习。

**nn.Module和nn.fuctional的区别：**

1. 由nn.Module生成的layer（层）是一个特殊的类，提供需要学习的参数。而nn.fuctional往往不提供参数学习，例如函数。

#### 2.3.3 TensorDataset和DataLoader介绍

##### 2.3.3.1 TensorDataset

TensorDataset用来将tensor打包。

##### 2.3.3.2 DataLoader

DataLoader作用就是对包装好的数据一批次一批次地抛出。

现假设存在一批数据，我们尝试用TensorDataset和DataLoader进行处理，具体代码如下所示。

```python
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = torch.tensor([44, 55, 66, 44, 55, 66, 44, 55, 66, 44, 55, 66])
train_ds = TensorDataset(a,b)
train_dl = DataLoader(train_ds,batch_size = 4,shuffer = True)

print(train_ds[0:2])
print('=' * 80)

for x_train,y_lable in train_ds :
    print(x_train,y_lable)
    
for i, data in enumerate(train_dl, 1):  # 注意enumerate返回值有两个,一个是序号，一个是数据（包含训练数据和标签）
    x_data, label = data
    print(' batch:{0} x_data:{1}  label: {2}'.format(i, x_data, label))    
```

具体效果如下所示。

![TensorDataset和DataLoader介绍](D:\Picture\StudyPicture\TensorDataset和DataLoader介绍.png)

#### 2.3.4 网络结构

###############

#### 2.3.5 模型训练

##################

### 2.4 卷积神经网络

------

**<font color='red'>学习目标：</font>**

1. 学会构建卷积神经网络（网络结构是什么样的？）
2. 掌握使用TensorDataset和DataLoader处理数据集
3. 训练卷积神经网络

------

使用卷积神经网络，我们的思路是：对数据集进行处理、构建网络、确定其他超参数（损失函数、优化器）、确定评价标准、训练网络、测试。

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets,transforms 
import matplotlib.pyplot as plt
import numpy as np
```



#### 2.4.1 数据集处理

本次卷积神经网络实验所使用的数据集同样是Mnist数据集。首先第一步就是获取Mnist数据，然后将其划分成训练集和测试集。划分完数据集后，就将数据打包，批发。具体代码如下所示。

```python
input_sizes = 28
class_nums = 10
num_epochs = 3
batch_sizes = 16

train_dataset = datasets.MNIST(root = './data',train = True,transform = transforms.ToTensor(),download =  True)
test_dataset = datasets.MNIST(root = './data',train = False,transform = transforms.ToTensor)

train_dataloader = torch.utils.data.DataLoader(dataset = train_dataset,batch_size = batch_size,shuffler = True)
test_dataloader = torch.utils.data.DataLoader(dataset = test_dataset,batch_size = batch_size,shuffler = True)
```



#### 2.4.2 构建卷积神经网络

构建卷积神经网络的主要思路是：先利用通过卷积操作提取特征，提取完特征后再将其展平，放入分类网络中进行分类。具体代码如下所示。

```python
class CNN(nn.Module):
    def __init__(self) :
        super(CNN,self).__init__()
        self.conv1 =  nn.Sequential(
            nn.Conv2d(
                in_channels = 1,
                out_channels = 16,
                kernel_size = 5,
                stride = 1,
                padding = 2
            ),
            nn.ReLU()
            nn.MaxPool2d(kernel_size = 2)
        
        )
        
        self.conv2 =  nn.Sequential(
            nn.Conv2d(
                in_channels = 16,
                out_channels = 32,
                kernel_size = 5,
                stride = 1,
                padding = 2
            ),
            nn.ReLU()
            nn.MaxPool2d(kernel_size = 2)
        
        )
        
        self.out = nn.Linear(32 * 7 * 7, 10)
    
    def forward(self,x) :
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)
        output = self.out(x)
        return output
```

在构建卷积神经网络时，我们可以清楚地看到每一次卷积都是以一个“套餐”的形式出现的，即卷积层+激活函数+池化层。这和我们上面的实验Mnist分类网络的网络结构有所不同，Mnist分类网络是在构造函数中创建layer，在forward函数中进行relu激活。

同时，我们还要**<font color='red'>注意参数（一层的输出对应于另一层的输入）要对应</font>**，不然会出现错误。在上述例子中，首先Mnist数据集的图片大小为（1，28，28）即1通道，28*28大小。因此在构建第一层卷积时，我们的in_chanels = 1。与之相对应的out_chanels表示输出feature map的个数。

顺便简单介绍一下nn.Conv2d()的其他参数：kernel_size，卷积核尺寸大小；stride，卷积核移动步长；padding，填充大小。（具体细节请查看卷积神经网络的相关笔记。）输出尺寸计算公式如下图所示（注：池化层计算公式同上）。

![卷积结果计算公式](D:\Picture\StudyPicture\卷积结果计算公式.png)

在上述模型中，可以看见我们通过卷积提取到一系列的特征图（32\*7\*7），在这里简单解释一下这个32\*7\*7是怎么来的。首先32是在最后一个卷积层中输出了32个feature maps（out_channels = 32），然后7*7是输入图片（已知大小为28\*28），通过两次池化操作后（MaxPool2d（kernel_size = 2）），图片大小变为：28/4 * 28 /4。

**简单总结一下在构造函数中的代码思路**：首先，卷积层以“套餐”的形式出现，为：nn.Conv2d(相关参数) --> nn.ReLU()-->nn.MaxPool2d(相关参数)。接着，注意每层输入输出之间的参数关系，最后到了全连接层nn.Linear()。

自定义一个神经网络，我们要注意两个关键点，第一个是**构造函数**（上面已经介绍过了），另一个便是**forward()函数**。接下来我们了解一下forward()中的相关信息。

**在forward函数中**，主要分为两个部分，分别是卷积提取特征和分类。卷积提取特征的过程其实和之前我们所学的一样，都是通过相应的layer得出相关数据。这里要注意的是，当我们提取完特征后，放入分类网络进行分类前，我们要**<font color='red'>注意参数对应</font>**。这时，我们的输入分类网络的数据是32个 7*7的feature map，如果不对其加以处理，一般而言分类网络是无法“识别”这些数据的。因此，此时我们要进行flatten操作（把这些数据“拉直”，改变其形状，使其适配分类网络）。在上面的代码中，**x =  x.view(x.size(0),-1)就对应flatten操作。**

在这里简单介绍一下该行代码的相关知识，首先关于pytorch中的size()函数，它的作用是返回tensor的shape（形状），返回的值是一个tuple的子类。具体效果如下图所示。在进行训练时，我们数据x的shape可以看作：batch_size个的（32 ，7 ，7）。此时，我们使用view(x.size(0),-1)操作对x进行reshape。在这里，x.size(0)表示batch_size，-1表示32\*7\*7（view()函数相关知识点）

![torch.size()作用](D:\Picture\StudyPicture\torch.size()作用.png)



#### 2.4.3 评价标准

在本次实验中，我们将以准确率为评价标准。实现准确率的具体代码如下所示。

```python
def accuracy(predictions,labels):
    pred = torch.max(predictions.data,1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum() 
    return rights, len(labels) 
```

现在来简单介绍一下上述代码。

首先，对于：**pred = torch.max(predictions.data,1)[1]**这行代码，作用是得到分类网络中预测的结果的类别，这里涉及到了机器学习中的分类问题，一般而言，分类问题通过softmax函数，我们可以得到每个类别的概率值，然后从中找到最大可能性的。其中要注意的细节主要是torch.max()这个函数，这个函数的相关信息如下图所示。由下图可知，dim = 1，寻找每行中的最大值，可知predictions.data一行就是一系列的预测结果.同时，因为我们最主要的是得到最大值索引，所以后面是[1]

![torch.max()函数](D:\Picture\StudyPicture\torch.max()函数.png)

接下来，让我们来简单分析一下**rights = pred.eq(labels.data.view_as(pred)).sum()**这一行代码。它的作用是比较预测值和标签值的结果是否相同，然后把所有相同的结果（视为预测正确）求和。

然后：**return rights, len(labels)** 最后就是返回正确的数量和样本数。

简单总结一下该函数的思路：首先是通过torch.max()函数得到预测结果，然后再将预测结果和label值做对比查看是否相同，将所有相同的值求和，这便是正确率，最后返回正确率和样本数。

#### 2.4.4 训练网络模型

我们在上面已经将本次卷积神经网络的实验准备得差不多了（实验数据、网络结构、评价标准）。现在到了最后一步，对网络模型进行训练。训练过程其实和之前的实验相似，先确定损失函数和优化器，然后实例化网络，进行训练。具体代码如下所示。

```python
net = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr = 0.001)

for epoch in range(num_epochs) :
    train_rights = []
    for batch_idx , (data,target) in enumerate(train_loader) :
        net.train()
        output = net(data)
        loss = criterion(output,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        right = accuracy(output,target)
        train_rights.append(right)
        
	if batch_idx % 100 == 0 :
        net.eval()
        test_rights = []
        for data,target in test_loader :
            output = net(data)
            right = accuracy(output,target)
            test_rights.append(right)
    
    train_r = (sum(tup[0]  for tup in train_rights),sum(tup[1]for tup in train_rights))
    test_r = (sum(tup[0] for tup in test_rights),sum(tup[1] for tup in test_rights))
    
    print('当前epoch:{},\tloss值:{:.6f}\t训练集准确率：{:.2f}\t验证集准确率：{:.2f}'.format(
        epoch,
        loss.data,
        100.train_r[0].numpy()/train_r[1],
        100.test_r[0].numpy()/test_r[1]
    ))
    
```

#### 2.4.5 相关知识点

###############

### 2.5 基于经典网络架构训练图像分类模型

------

**<font color='red'>学习目标</font>**：

1. 掌握torchvison包中的dataset模块和transforms模块进行数据集的预处理
2. 掌握torchvision包中的models模块进行经典网络的构建
3. 根据实际需求，对经典网络进行修改，训练。
4. 掌握网络模型的保存与测试

------

```python
import os
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torchvision
#pip install torchvision
from torchvision import transforms, models, datasets
#https://pytorch.org/docs/stable/torchvision/index.html
import imageio
import time
import warnings
import random
import sys
import copy
import json
from PIL import Image
```

此次实验是对一系列花的图像进行分类，一共有102种类别。整个项目的思路是：先处理数据，包括数据集文件目录的划分、数据增强、数据预处理。然后加载models模块中的经典网络，进行迁移学习，接着根据自己的实际要求进行调整，最后对网络模型进行训练、保存、测试。

#### 2.5.1 数据集处理

在本次实验中，一开始我们在对数据集进行处理前，要了解清楚数据集的存放位置。具体代码如下所示。

```python
data_dir = './flower_data'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
```

在确定好文件目录结构后，开始对数据进行预处理操作。具体代码如下所示：

```python
data_transforms = {
    'train' : transforms.Compose([transforms.RandomRotation(45),
                                  transforms.CenterCrop(224),
                                  transforms.RandomHorizontalFlip(p = 0.5),
                                  transforms.RandomVerticalFlip(p = 0.5),
                                  transforms.ColorJitter(brightness = ,contrast = ,saturation = ,hue = )
                                  transforms.RondomGrayscale(p = 0.025),
                                  transforms.ToTensor()
                                  transforms.Normaliza([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
    ]     
    ),
    'valid' : transforms.Compose([
        transforms.CenterCrop(224)
        transforms.ToTensor()
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    )
}
```

在这里主要使用的是transforms模块来进行预处理操作（调整数据尺寸、数据增强、数据标准化），transform模块的具体使用说明：

https://blog.csdn.net/LSJ944830401/article/details/102917336

这里简单介绍一下上面所做的数据预处理操作，主要分成以下几个方面：图像的旋转（随机旋转、水平、垂直翻转）、图像尺寸的调整、图像的色彩调整（亮度、对比度、饱和度、色相，这个属性都在同一个方法中，最后还有一个灰度）、格式的变化（PIL格式转变为tensor格式）、数据标准化、最后还有一个包含其他变化的transforms.Compose。

好了，在上面我们已经准备好了文件的读取路径、文件的预处理操作，接下来就是读取数据，对读取到的数据进行预处理后把数据打包，批量分发，具体代码如下所示。

```python
batch_size = 8

image_datasets = {x : datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x]) for x in ['train','valid']}
dataloaders = {x : torch.utils.data.Dataloader(image_datasets[x],batch_size = batch_size,shuffler = True) for x in ['train','valid']}
dataset_sizes = {x : len(image_datasets[x])  for x in ['train','valid']}
class_nums = image_datasets['train'].classes

```

上面代码已完成了读取数据和批发数据的步骤。要注意的是，在现在的数据集的目录结构中，我们只是按照1，2，3，4这样的顺序存储同一种花，并没有具体表示是什么类型，因此接下来的步骤就是将序号1，2，3，4等和具体的花名一一对应。在本次实验中，已经提供json文件来建立联系，具体实现代码如下所示。

```python
with open('flower_to_name.json','r') as f :
    flower_to_name = json.load(f)

```



#### 2.5.2 网络模型

在本次实验中，我们将采用torchvision中的models模块构建网络，进行迁移学习。所谓的迁移学习，简单来说就是“他山之石可以攻玉”，借鉴他人已经训练好的网络模型（权重参数已经训练好了），在此基础上进行微调或者特征提取。

在开始阶段，我们先设置本次实验所使用的模型名字，采用哪种迁移学习方式（finetue or freeze and train），查看是否能使用gpu进行实验，若可以就创建一个device。具体代码如下所示。

```python
model_name = 'resnet'
feature_extract = True
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu :
    print('cuda wrong')
else:
    print('cuda right')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
```

迁移学习分为两种，一种是微调，feature_extract设置为False时，为微调，即在已有权重参数基础上根据自己的实际需求对所有网络层进行重新训练，所有的权重参数的requires_grad()设置为True。第二种是特征提取，当我们的feature_extract设置为True时，意为特征提取；即保持特征提取的权重参数不变，仅重新训练最后一层的网络权重参数，仅最后一层的权重参数的requires_grad()设置为True。具体代码如下所示。

```python
def set_parameters_requires_grad(model,feature_extracting) :
    if feature_extracting :
        for param in model.parameters() :
            param.requires_grad() = False
```

在实际项目中，我们往往不只要用到一种网络结构，可能是多种，接下来就是定义一个初始化网络模型的函数，提高适用性。该函数的设计思路是：我们要确定初始化模型的名字，是否采用其初始化参数，选取哪种迁移学习，根据实际需求构建相应的全连接层。具体代码如下所示。

```python
def initialize_model(model_name,num_classes,feature_extract,use_pretrained = True) :
    model_ft = None
    input_size = 0
    
    if model_name == 'resnet' :
        model_ft = models.resnet152(pretrained = use_pretrained)
        set_parameter_requires_grad(model_ft,feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Squential(
            nn.Linear(num_ftrs,102)
            nn.LogSoftmax(dim = 1)
        )
        input_size = 224       
	elif model_name == 'vggnet' :
        ###vggnet的迁移学习
        
    elif model_name == 'alexnet' :
        ### alexnet的迁移学习
        
    return model_ft,input_size    
```

在该函数中，根据Pytorch官方文档的规范格式，我们知道该如何进行网络模型的微调：首先是通过torchvision.models相关的网络模型（models.resnet152、models.alexnet等等），然后设定是进行微调还是freeze and train，接着便是设置最后一层fc层的规模，确定输入、输出数据的大小，使用相应的softmax函数进行分类。最后数据集图片的尺寸大小。

在完成上述步骤后，我们已经可以创建一个网络模型,并把它放入gpu中运行。具体代码如下所示。

```python
model_ft,input_size = initialize_model(model_name,102,feature_extract,use_pretrained = True)
model_ft = model_ft.to(device)
filename = 'checkpoint.pth'
#####查看是否训练所有层。
params_to_update = model_ft.parameters()

print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)
```



#### 2.5.3 模型训练

在创建好网络模型后，我们接下来要做的便是对模型进行训练。在训练前，我们还要确定其他的超参数，如优化器、损失函数等等。具体代码如下所示。

```python
optimizer_ft = optim.Adam(params_to_update,lr=1e-2)
scheduler = optim.lr_scheduler.StepLR(optimizer_ft,step_size = 7,gamma = 0.1)
criterion = nn.NLLoss()
```

上面是很常规的超参数设置，有新的地方：学习率衰退。用到的是optim.lr_scheduler模块。

接下来便是创建一个训练函数，代码的实现思路，具体代码如下所示。

```python
def train_model(model,dataloader,criterion,optimizer,num_epochs,is_inception = False,filename) :
    since = time.time()
    best_acc = 0
    
    model.to(device)
    
    train_acc_history = []
    val_acc_history = []
    train_losses = []
    valid_losses = []
    
    LRs = [optimizer.param_groups[0]['lr']]
    
    best_model_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(num_epochs) :
        print('epoch:{}/{}'.format(epoch,num_epochs - 1))
        print('-' *10)
        
        for phase in ['train','valid'] :
            if phase == 'train' :
                model.train()
            elif phase == 'valid' :
                model.eval()
            
            running_losses = 0.0
            running_corrects = 0
            
            for input,labels in dataloaders[phase] :
                inputs = inputs.to(device)
                labels = labels.to(device)
            
            	optimizer.zero_grad()
            
            	with torch.set_grad_enabled(phase == 'train') :
                	outputs = model(inputs)
                	loss = criterion(outputs,labels)
                	_,preds=torch.max(outputs,dim = 1)
                
                	if phase == 'train' :
                    	loss.backward()
                    	optimizer.step()
            
            	running_loss += loss.item() * inputs.size(0)
            	running_corrects = torch.sum(preds == labels.data)
            
        	epoch_loss = running_loss / len(dataloaders(phase).dataset)
        	epoch_acc = running_corrects / len(dataloaders(phase).dataset)
            
            time_elapsed = time.time() - since
            print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            if phase == 'train' and epoch_acc > best_acc :
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                state = {
                    'state_dict' : model.state_dict(),
                    'best_acc' : best_acc,
                    'optimizer' : optimizer.state_dict()
                }
                torch.save(state,filename)
            if phase == 'valid':
                val_acc_history.append(epoch_acc)
                valid_losses.append(epoch_loss)
                scheduler.step(epoch_loss)
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_losses.append(epoch_loss)
        print('Optimizer learning rate : {:.7f}'.format(optimizer.param_groups[0]['lr']))
        LRs.append(optimizer.param_groups[0]['lr'])
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
        
    model.load_state_dict(best_model_wts)
	return model, val_acc_history, train_acc_history, valid_losses, train_losses, LRs
```

太他妈复杂了，里面涉及了很多知识点，很多细节，一时半会解释不清楚。现在先挑几个比较关键的知识点说一下吧。一开始在定义这个train_model的函数时，我们要确定需要用到的参数，经过上面实验的各个步骤我们可以得知，我们需要用到model（模型）、dataloader（数据集）、optimizer（优化算法）、criterion（损失函数）、num_epochs(要进行训练的次数)、filename（保存模型的名称）、还有一个，不重要。

好了，在确定完参数后，开始编写train_model函数的相关内容。主要思路如下所示：把model放入gpu内，然后初始化属性值，包括：best_acc、train_acc_history、valid_acc_history、train_losses、valid_losses、LRs、best_model_wts。

然后开始迭代训练。在迭代训练的过程中，区分训练过程和验证过程。这里是一个我感兴趣的知识点，这里是用以下代码来进行区分的。

```python
for phase in ['train','valid'] :
    if phase == 'train' :
        model.train()
    else phase == 'valid' :
        model.eval()
        
    #####后面就开始训练咯
    running_loss =
    running_acc =
```

训练的第一步，就是要取数据。我感觉取数据也是一个常用的操作，再简单复现一下。

```python
for inputs,labels in dataloaders[phase] :
    inputs = inputs.to(device)
    labels = labels.to(device)
```

取完数据后，就前向传播-->反向传播-->优化这样的过程咯，这个过程我也很感兴趣，再简单复现一下。

```python
optimizer.zero_grad()
with torch.set_grad_enabled(phase == 'train'):
    outputs = model(inputs)
    loss = criterion(outputs,labels)
    _,pred  = torch.max(outputs,1)
    if phase == 'train':
       loss.backward()
       optimizer.step()
    
```

上面的代码看上去和之前做的实验不太相同，首先是使用了torch.set_grad_enabled()这个函数来进行局部梯度的设置。

为什么要进行局部梯度的设置呢，因为在迁移学习中，有两种方式，一种是原封不动把之前的模型数据迁移过来，另一种是在给出模型的基础上进行训练。

再具体展开说明一下，分类任务，分为了特征提取和分类，对分类任务进行迁移学习，就主要考虑上面两个方面，一般而言，就数据集的特征、规模进行考虑，同时根据自己任务的实际需求来调整我们的分类网络。

（未完待续）

#################

两种迁移学习的训练方式、模型的保持与加载、测试集的测试（数据处理、模型加载、进行测试）

#################

```python
model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs  = train_model(model_ft, dataloaders, criterion, optimizer_ft, num_epochs=20, is_inception=(model_name=="inception"))
```

ok，敲完上面的代码，我们的项目基本上完成了，剩下还有一些功能需要实现：进行模型的微调，即对模型的所有参数进行训练；当我们训练完后得到最好的模型，加载这个模型，然后进行测试。具体代码如下所示。

```python
for param in model_ft.named_parameters() :
    param.requires_grad = True

optimizer = optim.Adam(params_to_update, lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
criterion = nn.NLLLoss()
```

上面代码的思路就是：在feature_extract的基础上，要对全面的权重参数进行学习。

在这里又不得不回顾之前的一些步骤代码了，小编写到这里不禁头晕眼花，大脑宕机，太他妈复杂了，学习不能太死板啊，先从简单的入手，万丈高楼平地起，然后在理解基础知识点的基础上，把各个知识点建立联系。


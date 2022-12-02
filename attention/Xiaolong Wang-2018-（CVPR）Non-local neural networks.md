# Xiaolong Wang-2018-（CVPR）Non-local neural networks

**（注：棕色标注表示我的一些疑问；黄色标注表示阅读文献时候的注意点，帮助我重新读的时候更快理解文章内容；绿色标注表示问题（研究目标、研究内容、关键问题），问题背景等；蓝色标注表示某项方法、行为的作用、目的、意义；红色标注表示我认为的重点，需要特别关注；紫色标注表示预备知识；）**

**（注：现阶段（基础），笔记主要围绕三点，1.知识体系结构  2.文章的诉求  3.文章的主要贡献、特点、创新点。）**

![image-20221106105213097](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221106105213097.png)

------

## 背景、研究目标、研究内容

### 背景

**卷积**和**递归运算**都是**一次处理一个局部**邻域的构建块。（Both convolutional and recurrent operations are building blocks that process one local neighborhood at a time. ）

**<font color='red'>在深度神经网络中，捕获长距离依赖性至关重要。（Why？）</font>**

对于顺序数据（例如，在**语音、语言中）（sequential data）**，**重复操作**[38，23]是远程依赖性建模的主要解决方案。（**recurrent operations** [38, 23] are the dominant solution to long-range dependency modeling. ）

对于**图像数据（image data）**，长距离依赖性**由卷积运算的深堆栈形**成的**大感受野**建模[14，30]。（long-distance dependencies are modeled by **the large receptive fields** formed by **deep stacks of convolutional operations**）

**<font color='red'>卷积运算和递归运算都在空间或时间上处理局部邻域；因此，只有当重复应用这些操作，在数据中逐步传播信号时，才能捕获长程相关性。</font>**

重复本地操作（Repeating local operations）有几个限制。

1. 首先，它的计算效率很低。
2. 第二，它导致了需要仔细解决的优化困难[23，21]。
3. 最后，这些挑战使多跳依赖性建模变得困难，例如，当消息需要在远距离位置之间来回传递时。

### 研究目标

1.将非本地操作（non-local operation）作为一个通用的构建块家族（block），用于捕获远程依赖关系。

2.这个构建块可以**插入**到许多计算机视觉架构中

### 研究内容

![image-20221201171703583](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221201171703583.png)

我们提出了非局部操作作为一种高效、简单和通用的组件，用于利用深度神经网络捕获长距离依赖。我们提出的非局部运算是计算机视觉中经典非局部平均运算[4]的推广。

**<font color='red'>直观地说，非本地操作将某个位置的响应计算为输入特征地图中所有位置的特征的加权和</font>**（图1）。（ Intuitively, a non-local operation computes the response at a position as a weighted sum of the features at all positions in the input feature maps (Figure 1)）

这组位置（The set of positions）可以是空间、时间或时空，这意味着我们的操作适用于图像、序列和视频问题。

## 结果与讨论



## 文章好在哪里

### 主要贡献

使用非局部运算有几个优点：

（a）与递归和卷积运算的渐进行为不同，非局部运算通过计算任意两个位置之间的交互直接捕获长距离依赖，而不管它们的位置距离如何；

（b） 正如我们在实验中所显示的，即使只有几层（例如，5层），非局部操作也是有效的，并获得最佳结果；

（c） 最后，我们的非局部操作保持了可变的输入大小，并且可以很容易地与其他操作（例如，我们将使用的卷积）组合。

我们展示了非本地操作在**视频分类应用中的有效性**。在视频中，**长距离交互发生在空间和时间上**的遥远像素之间。**<font color='red'>一个单独的非局部块（它是我们的基本单元）可以以前馈的方式直接捕获这些时空相关性。</font>**

通过一些非局部块，我们称为非局部神经网络的架构对于视频分类比2D和3D卷积网络更准确[48]（包括膨胀的变体[7]）。此外，非局部神经网络比其3D卷积网络在计算上更经济。

在动力学[27]和Charades[44]数据集上进行了全面的消融研究。仅使用RGB而不使用任何铃声（例如，光流、多尺度测试），我们的方法在两个数据集上取得的结果与最新的竞赛获奖者不相上下或更好。

### 相关工作



## 自我思考

### 思路与问题



### 句式积累

![image-20221201165553565](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221201165553565.png)
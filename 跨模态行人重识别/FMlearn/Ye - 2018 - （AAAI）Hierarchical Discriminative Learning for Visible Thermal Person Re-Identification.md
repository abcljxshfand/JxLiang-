# Ye - 2018 - （AAAI）Hierarchical Discriminative Learning for Visible Thermal Person Re-Identification

**（注：**在论文中，黄色标记是读论文时候的main body，绿色标记是背景、名词解释，红色标记是重点，紫色标记是相关工作，蓝色是作用、目的与意义，棕色是疑问、问题。）

## 问题

### the modality-invariance

不同模态间的共性，可以用来减少模态差距。（下图蓝色区域）

与之相对的是modality-specific，每一个模态私有的。（下图绿色紫色黄色区域）

![image-20221121164047285](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221121164047285.png)

多模态模型结构

![image-20221121164351234](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221121164351234.png)



------



## 摘要

**问题：**

现有的跨模态匹配方法主要侧重于对跨模态差异进行建模（Existing cross-modal matching methods mainly **focus on modeling the cross-modality discrepancy**,），而VT-REID也存在由不同相机视图引起的跨视图差异。

本文通过**联合优化模态特定（modality-specific）**和**模态共享（modality-shared）度量（ jointly optimizing the modality-specific and modality-shared metrics）**，提出了提出了一种**<font color='red'>分层的跨模态匹配模型</font>**。大量实验证明了该方法的有效性和鲁棒性。

**模态特定度量（V和T）**将两个异构模态转换为一致的空间，从而可以处理交叉视图变化问题。同时，特定于模态的度量也可以解决人内模态内的变化。

**模态共享度量（M）**旨在最小化跨模态差异，这可以将不同的人与两种不同的模态区分开来。

同时改进了一个**<font color='red'>双流CNN网络</font>**用于共享网络参数。



## 引言

为了解决跨模态问题，提出了一个两阶段模型（表征学习+度量学习）。双流CNN网络提取共享的特征，然后用**分层跨模态度量学习**（HCML）

#### 主要贡献

1. 提出了一种新的VT-REID**分层跨模态匹配模型**，该模型可以同时处理跨模态差异和跨视角变化，以及人内模态变化。这个模型**联合优化模态特定和模态共享度量**
2. 提出了一种改进的**双流CNN网络**来学习深度多模态共享特征表示。该网络用了id loss和contrastive loss，**ld loss**的目的是对特定领域的信息进行建模，以区分每个模态中的不同人员。**contrastive loss**的目的是弥补了两种异质模态之间的差距，并增强了学习表征的模态方差





## 方法

<img src="C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221121100027814.png" alt="image-20221121100027814" style="zoom:200%;" />

两阶段框架（即特征学习和度量学习）来解决VT-REID问题。

1.提出了一个**<font color='red'>TwO流CNN网络（TONE）</font>**来学习两种异质模态的多模态共享特征表示，集成对比损失以弥合两种模态之间的差距，并增强所学习表示的模态不变性。

2.然后，通过联合优化模态特定和模态共享度量，引入了**<font color='red'>分层跨模态度量学习（HCML）方法</font>**。<font color='green'>模态特定度量（V和T）将两个异构模态转换为一致的空间，从而可以处理交叉视图变化问题</font>（The modality-specific metrics (V and T ) transform two heterogenous modalities into a consistent space, thus could handle the cross-view variation problem. ）。同时，特定于模态的度量也可以解决人内模态内的变化。模态共享度量（M）旨在最小化跨模态差异，这可以将不同的人与两种不同的模态区分开来。

### Multi-Modality Sharable Feature Learning

#### 组成

输入端：两类图片（可见、热图像）

baseline：采用AlexNet2（Krizhevsky、Sutskever和Hinton 2012）作为我们的基线网络，它包含五个卷积层（conv1∼ conv5）和三个完全连接的层（fc1∼ fc3）

输出：每个流中fc2（或fc1）的输出作为每个人图像的特征表示

优化：为了学习特征表示，引入了两种优化目标，包括两种身份损失和一种对比损失。

#### Identity loss

**作用：**

身份丢失旨在通过使用模态特定信息来学习辨别特征表示，该信息可以区分每个模态中的不同人

**公式：**

![image-20221202104021861](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221202104021861.png)

在我们的架构中有两个流CNN网络，我们将两种不同模态的学习参数表示为θ1和θ2。fc3的长度由人数（K）定义，**这与许多多类分类问题类似。然后，采用交叉熵损失进行身份预测。**具体而言，可见图像的身份损失由

#### Contrastive Loss

**作用：**

对比损失试图弥合两种异质模态之间的差距，这也可以增强特征学习的模态不变性。

**公式：**

![image-20221202104134919](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221202104134919.png)

将x，z表示为两个流的fc2层的输出，在计算对比损失之前首先引入l2归一化。因此，对比损失的定义如下

### Hierarchical Cross-modality Metric Learning

#### 组成

两个流网络中提取的特征为{Xi}K1i=1；对于热图像，表示特征为{Zj}K2j=1，K1和K2表示每个模式中的人数。每个人在每个模态中可能有多个图像，因此每个Xi和Zj都由Rd×ni（/j）矩阵表示。

![image-20221202104344774](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221202104344774.png)

**HCML的主要思想**是将两个异构模态（模态特定度量）转换为一致的空间，随后可以学习模态共享度量（模态共享度量）。因此，我们制定HCML如下

![image-20221202104521693](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221202104521693.png)

#### Modality-specific Terms

**作用：**

旨在约束每个模态中同一个人的特征向量。

**公式：**

![image-20221202104650769](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221202104650769.png)



#### Modality-shared Term

**作用：**

旨在学习在使用特定于模态的度量进行转换之后能够将不同的人从两种不同模态中区分出来的度量。

**公式：**

![image-20221202104810914](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221202104810914.png)



### Optimization of HCML



## 实验

![image-20221122215845262](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221122215845262.png)



![image-20221122215858571](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221122215858571.png)



## 心得体会

### 思路与问题

**思路：**

解决跨模态行人重识别问题，关键要尽可能消除模态之间的差异带来的影响，同时，也要关注模态内，由于跨摄像头的拍摄角度、行人姿势等因素带来的影响。

因此，本文作者的解决思路是利用深度学习来提取两个模态中共享的特征，当我们得到特征后，用度量学习来优化这些特征，使得在同一模态内，相同行人的距离拉近。不同模态中，把不同的行人推远。

有了解决思路后，作者提出的解决工具是

1.改进后的双流神经网络：TONE

2.一种分层跨模态度量模型：HCML




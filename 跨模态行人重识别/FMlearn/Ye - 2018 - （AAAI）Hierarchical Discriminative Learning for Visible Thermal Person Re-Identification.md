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

![image-20221121100027814](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221121100027814.png)

## 摘要

本文通过联合优化**模态特定（modality-specific）**和**模态共享（modality-shared）度量**，提出了提出了一种**<font color='red'>分层的跨模态匹配模型</font>**。大量实验证明了该方法的有效性和鲁棒性。

**模态特定度量（V和T）**将两个异构模态转换为一致的空间，从而可以处理交叉视图变化问题。同时，特定于模态的度量也可以解决人内模态内的变化。

**模态共享度量（M）**旨在最小化跨模态差异，这可以将不同的人与两种不同的模态区分开来。

同时改进了一个**<font color='red'>双流CNN网络</font>**用于共享网络参数。



## 引言

为了解决跨模态问题，提出了一个两阶段模型（表征学习+度量学习）。双流CNN网络提取共享的特征，然后用**分层跨模态度量学习**（HCML）

#### 主要贡献

1. 提出了一种新的VT-REID**分层跨模态匹配模型**，该模型可以同时处理跨模态差异和跨视角变化，以及人内模态变化。这个模型**联合优化模态特定和模态共享度量**
2. 提出了一种改进的**双流CNN网络**来学习深度多模态共享特征表示。该网络用了id loss和contrastive loss，**ld loss**的目的是对特定领域的信息进行建模，以区分每个模态中的不同人员。**contrastive loss**的目的是弥补了两种异质模态之间的差距，并增强了学习表征的模态方差



## 心得体会

为了解决跨模态问题，作者提出了一种分层的跨模态匹配模型，改进了双层cnn网络。

基本思路是表征学习+度量学习一起使用。整体框架是一个二阶段模型，先表征，后度量。

**表征：**通过双层CNN网络提取两种模态共享的特征。为了达到这个目的，作者使用了两种损失函数，identiy loss和contrastiv loss。id loss，让各自模态提取更好的特征，锦上添花的作用。contrastive loss才是重点，从两种模态中提取共享的特征。**<font color='red'>然后用于度量学习。</font>**

**度量：**有两种度量方式，分别是模态特定和模态共享度量**。特定模态度量**压缩同模态同ID的行人距离，**共享模态度量**学习一个能够将不同模态转到相同特征空间的投影（简单概括：前者压缩类内距离，后者用于跨模态的ID判别）。





## 实验

![image-20221122215845262](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221122215845262.png)



![image-20221122215858571](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221122215858571.png)
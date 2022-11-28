# （同济大学）Wu-2020-（ICIC）Position Attention-Guided Learning for Infrared-Visible Person Re-identification

**（注：棕色标注表示我的一些疑问；黄色标注表示阅读文献时候的注意点，帮助我重新读的时候更快理解文章内容；绿色标注表示问题（研究目标、研究内容、关键问题），问题背景等；蓝色标注表示某项方法、行为的作用、目的、意义；红色标注表示我认为的重点，需要特别关注；紫色标注表示预备知识；）**

**（注：现阶段（基础），笔记主要围绕三点，1.知识体系结构  2.文章的诉求  3.文章的主要贡献、特点、创新点。）**

![image-20221106105213097](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221106105213097.png)

------

## 背景、目的及结论

### 背景

跨模态问题，需要关注的问题之一是要**如何缩小不同模态间的差异**。

**目前的大多数方法侧重于改善全局特征**，以解决跨模态行人重识别问题。大多数深度模型**忽略了一些有区别的局部特征表示**，例如衣服的类型或鞋子的样式。局部特征表示具有显著的可分辨性，并且不受交叉模态模式（cross-modality modes）的影响。



### 目的

1. 构建一个模型来学习特定于共享的特征表示（share specific feature representations）来解决问题。
2. 设计一个模块来增强区分性局部特征表示。

### 结论

![image-20221127202154634](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221127202154634.png)

1. 在这项工作中，我们提出了一个称为双路径局部信息结构（DLIS）的深度模型，该模型学习特定于共享的特征表示以解决问题。双路径网络具有两个单独的分支，其中包含可见流和红外流以提取模态共享特征。每个分支采用ResNet50[46]模型作为主干网络，提取行人的全局特征。

   ![image-20221127203327089](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221127203327089.png)

2. 提出了一个位置注意力引导学习模块（PALM），以增强区分性局部特征表示，并获得优异的识别性能。我们提出的注意机制可以迫使模型仅从跨模态图像中提取局部特征表示而不是全局信息，以形成最终的特征描述符。**我们将两个分支学习的特征图分成几个条带，用于本地信息学习。**

   （**The attention mechanism** we proposed can force the model extracting the local feature representations **rather than** the global information only from the cross-modality images to form the final feature descriptors. We **split the feature maps** learned by the two branches into several stripes **for local information learning**. ）

​	3 .为了监督网络提取鉴别特征以缩小不同模态的裕度，所提出的模型进行了**交叉熵损失函数和异中心损失函数**的联合监督。

## 结果与讨论



## 文章好在哪里

**主要贡献**

（1）我们提出了一种称为双路径本地信息结构（DLIS）的体系结构来学习共享特定特征表示。实验表明，DLIS在红外可见光PReID社区中取得了优异的性能。

（2） 我们提出了一种新的注意力机制，名为位置注意力引导学习模块（PALM），以捕获长程依赖性并增强异质模态的辨别性局部特征表示。该模块可以捕获特定位置信息并提取模态局部特征以形成区分特征描述符。（to **capture long-range dependencies** and enhance the discriminative local feature representations of heterogenous modality. **The module can capture specific position information and extract modality local features to form discriminative feature descriptors**.）

（3） 与最近的研究相比，所提出的模型在包括SYSU-MM01和RegDB数据集在内的两个基准和具有挑战性的数据集上达到了最先进的状态。



**相关工作**

**列举2020年前具有代表性的文献，简单梳理了跨模态行人重识别任务的发展。提出了一个研究思路：关注局部信息（包含更细粒度和区分性的特征）、使用注意力机制**

Wu等人[18]首先提供了名为SYSUMM01的跨模态PReID数据集，并针对跨模态检索问题提出了深度零填充架构。[19]提出的称为cmGAN的GAN模型，包含一个生成器和一个鉴别器。cmGAN中生成器的目的是提取两种模态的特征，然后将提取的特征馈送到鉴别器以区分输入模态。Ye等人[23]提出了一种称为TONE的深度双蒸汽CNN架构。通过联合监督交叉熵损失和对比损失来训练TONE模型，以减少交叉模态变化。基于用分层度量学习训练的TONE模型，Ye等人[24]提出了一种用双约束顶级（DCTR）损失训练的双路径端到端深度模型，以学习最终的区分特征表示。[25]提出的一种双水平差异减少学习（D2RL）方法，分别处理模态和外观差异。该模型首先通过图像级别转换统一图像表示。然后通过特征级双路径网络减少外观差异。

在PReID研究中，零件级信息可能包含更细粒度和区分性的特征（**The part-level information** may contain **more fine-grained and discriminative features** in PReID studies）。仅举几个例子，对于基于部件的卷积基线（PCB）[28]，它将特征图平均分成几个条带，旨在学习PReID任务的部件级局部表示特征。Wang等人[29]提出了一种基于条带的模型，称为多粒度网络（MGN），该模型将输入图像划分为多个条带，用于以多粒度的方式提取有区别的局部特征表示。



## 自我思考

新瓶装旧酒，用了别人的方法来解决一个特定的问题（单模态的行人重识别中提取局部信息）。

比如说：

- 受[28，35]的启发，我们采用了将特征图划分为多条条纹的方法，以确保与身体部位相对应的局部特征表示。

  Inspired by [28, 35], we adopted the method of dividing the feature map into multiple stripes to ensure the local feature representations corresponding to the body parts.

- 受[36，37]的启发，我们设计了一个新的注意力学习模块，名为位置注意力引导学习模块（PALM）。

  Inspired by [36, 37], we design a novel attention learning block which named Position Attention-guided Learning Module (PALM)

- 在PReID研究中，零件级信息可能包含更细粒度和区分性的特征。

  The part-level information may contain more fine-grained and discriminative features in PReID studies.

  仅举几个例子，对于基于部件的卷积基线（PCB）[28]，它将特征图平均分成几个条带，旨在学习PReID任务的部件级局部表示特征。

  To name just a few, for Part-based Convolution Baseline (PCB) [28], it divided the feature map into several stripes equally, aimed to learn the part-level local representations features for PReID task.

  Wang等人[29]提出了一种基于条带的模型，称为多粒度网络（MGN），该模型将输入图像划分为多个条带，用于以多粒度的方式提取有区别的局部特征表示。

  Wang et al [29] proposed a stripe-based model called Multiple Granularities Network (MGN), which divided the input image into multiple stripes for extracting discriminative local feature representations in multiple granularities manner.
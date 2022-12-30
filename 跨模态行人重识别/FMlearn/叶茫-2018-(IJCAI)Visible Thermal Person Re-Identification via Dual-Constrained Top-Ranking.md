# 叶茫-2018-(IJCAI)Visible Thermal Person Re-Identification via Dual-Constrained Top-Ranking

**（注：棕色标注表示我的一些疑问；黄色标注表示阅读文献时候的注意点，帮助我重新读的时候更快理解文章内容；绿色标注表示问题（研究目标、研究内容、关键问题），问题背景等；蓝色标注表示某项方法、行为的作用、目的、意义；红色标注表示我认为的重点，需要特别关注；紫色标注表示预备知识；）**

**（注：现阶段（基础），笔记主要围绕三点，1.知识体系结构  2.文章的诉求  3.文章的主要贡献、特点、创新点。）**

![image-20221106105213097](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221106105213097.png)

------

## 背景、研究目标、研究内容

## 背景



## 研究目标



## 研究内容（方法）

![image-20221203211025171](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221203211025171.png)

它包括**两个主要组成部分**：用于特征提取的双路径网络（一个路径用于可见图像，另一个用于热图像）和用于特征学习的双向双约束顶级损失。

**<font color='red'>注意，浅层（特征提取器）的权重不同以提取模态特定信息，而嵌入FC层（特征嵌入）的权重共享用于多模态共享特征学习。</font>**

在L2规范化之后，我们为网络训练引入了双向双约束顶级损失。同时，身份损失进一步与排名损失相结合，以提高性能。

------

![image-20221203211548709](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221203211548709.png)

------

## Dual-path Network

我们提出了一种双路径网络来提取可见域和热域的特征。

具体而言，双路径特征学习网络包含两个部分：特征提取器和特征嵌入。**<font color='red'>前一特征提取器旨在捕获不同图像模态的模态特定信息。后一种特征嵌入侧重于学习多模态共享空间，以弥合两种异构模态之间的差距。</font>**

浅层：特征提取器

深层：特征嵌入

浅层+深层：投影模型

#### Feature extractor

用相同的网络结构分别对两种模态提取特征，浅层不共享参数，可以提取low-level的特征（纹理、边缘），这些特征一般两种模态共享。但有一个问题：优化，它们是对模态特定信息分别优化。

#### Feature embedding

深层权值共享，目的是消除模态间的差异。

如何消除？投影到统一的空间。而共享结构可以帮助我们实现这个功能

**[Wang et al, 2016a] Liwei Wang, Yin Li, and Svetlana Lazebnik. Learning deep structure-preserving image-text embeddings. In CVPR, pages 5005–5013, 2016**

**[Zhou et al, 2018] Joey Tianyi Zhou, Heng Zhao, Xi Peng, Meng Fang, Zheng Qin, Zheng Qin, and Rick Siow Mong Goh. Transfer hashing: From shallow to deep. IEEE Transaction on Neural Network and Learning Systems, 2018.**

因此，Feature extractor + Feature embedding = 完整的投影模型

![image-20221203213951398](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221203213951398.png)

## 结果与讨论



## 文章好在哪里

### 主要贡献



### 相关工作



## 自我思考

### 思路与问题

终于解答了我一直以来的疑问，双流网络结构，为什么可以解决跨模态匹配问题。

整理一下目前的资料：多模态的表征学习、双流网络结构

双流网络结构可以**<font color='red'>根据功能作用分为两个部分</font>**：domain-specific sub-network 和 shared sub-network （PS：不同论文有不同论文的命名形式，真他妈乱）

**对于domain-spcific sub-network：**

~~（我他妈一直不理解这个子网络，论文有些说提取共享特征，有些又说是specific information，他妈的。）~~

采用相同的网络结构，原因是都是图片

浅层，不共享参数。分别对两种模态数据提取特征，主要是一些low-level的特征信息，包括纹理，边缘，所以可以理解为提取共享信息。**<font color='red'>但是存在一个问题，优化模型，是针对模块特定信息进行优化。</font>**



 **对于shared sub-network：**

共享参数，原因：想把两种模态数据投影到同一空间，消除模态差异

最后，domain-specific sub-network + shared sub-network = 多模态的表征学习模型



**优化模型的方式：**

结构化（对表征学习加以限制） 、 度量学习

### 句式积累




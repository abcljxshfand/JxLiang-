# Ancong Wu-2017-（ICCV）RGB-Infrared Cross-Modality Person Re-Identification

**（注：**在论文中，黄色标记是读论文时候的main body，绿色标记是背景、名词解释，红色标记是重点，紫色标记是相关工作，蓝色是作用、目的与意义，棕色是疑问、问题。）

## 问题

### domain adaption

拉近源域与目标域的“距离”

### **Discriminative feature**

做图像说白了就是希望有足够有判别性的特征，这样在分类或者匹配、检索的时候才能有较好的精度。

如何称之为有**判别性**的**特征**？作者利用编码器的思想，对于同一ID的图形的特征，如果编码后仍可以较好的解码为同一ID的特征，那么我们就说这个特征有判别力。这里有个点值得注意：编码器是针对**图像特征**，非图像本身。**好的特征表示大概有2个衡量标准：可以很好的重构出输入数据、对输入数据一定程度下的扰动具有不变性。**



### 为什么要用单流网络



## 摘要

**介绍传统的行人重识别（单模态、适用场景）：**目前大多数Re-ID都是基于 RGB 图像。但是有时RGB 图像并不适用，例如在黑暗的环境或夜间。

**保证摄像头全天候运行（由于可见光摄像头对夜间的监控安防工作作用有限）**：在许多视觉系统中，红外 (IR) 成像变得必不可少。

**两种模态差异（针对昼夜光照条件不同的问题/可见光模式和红外模式所拍摄的图像分别是）：**为此，需要将 RGB 图像与红外图像进行匹配，这些图像是异构的，具有非常不同的视觉特征。

评估了现有流行的跨域模型，包括三种常用的神经网络结构（单流、双流和非对称 FC 层）并分析它们之间的关系。提出了深度零填充，用于训练单流网络，使其自动进化网络中特定领域的节点，以进行跨模态匹配



## 引言

**RGB图像和IR图像的区别：**由于大多数监控摄像机能够在黑暗中自动从RGB模式切换到IR模式。 分别在白天和夜间在两个室外场景中捕获的RGB图像和红外（IR）图像的示例。每两列中的图像都是同一个人的。由接收不同波长光的设备捕获，同一个人的RGB图像和红外图像看起来非常不同。

![img](https://img-blog.csdnimg.cn/f36041c64a6442c3bbd60687add5028a.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5rex5bqm5a2m5LiN5Lya5Lmg,size_20,color_FFFFFF,t_70,g_se,x_16)

第一行的 **RGB 图像**具有**三个包含可见光颜色信息的通道**，而第三行的 **IR 图像**具有**一个包含不可见光信息的通道**。 因此，它们可以被视为异构数据。

 其次，从成像原理来看，RGB和IR图像的波长范围不同。

### 主要贡献（1）数据集SYSU-MM01

**提出了跨模态行人重识别领域的一个常用数据集-----**-SYSU-MM01
**数据集中共**491个行人id**，**296个id用来训练，99个id用来验证，96个id用来测试模型**，在**训练阶段**使用的**<font color='red'>**296个id既有RGB图像也有IR图像**</font>**，而**测试阶段使用<font color='red'>**IR图像做probe set，RGB图像做gallery set。也就是说待检索图像是红外图像，而检索库为RGB图像**</font>。除此之外，本文还设置了**两种模式—全局搜索和室内搜索**。顾名思义，全局搜索就是在所有场景数据中搜索待检索目标，而室内搜索即是仅在室内拍摄的RGB图像中搜索目标。

### **主要贡献（2）分析三种网络结构**

分析了现存的较流行的三种网络结构-------（单流结构、双流结构、非对称FC层结构）

![img](https://img-blog.csdnimg.cn/20201030141229347.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlbnJvdWRlYmFvemk=,size_16,color_FFFFFF,t_70#pic_center)

上图为论文评估中的四个网络结构。转换块的结构取决于基础网络。转换块和FC层的颜色指示是否共享参数。红色和蓝色表示特定参数，绿色表示共享参数

不同的网络结构，有不同的特点。

在跨模态匹配问题中，我们的网络需要：1.提取模态间共享的特征。2.使用特征进行匹配

根据功能，完善网络结构，分别是the domain-specific network和the shared network。domain-specific network的作用是可以提取不同域的共享特征。the shared network的作用是可以提取用于匹配的区分特征。

#### 单流结构（One-stream Structure）

**组成：**

只有一个输入，所有参数都在整个网络中共享



**作用：**

代表性网络包括AlexNet[16]、VGG[38]、GoogleNet[40]、ResNet[9]等，它们在分类、检测、跟踪和许多其他任务中表现良好。

在Re-ID领域，最先进的网络之一JSTL-DGD[47]也使用一种流结构。通常，在这些任务中，网络的输入是RGB图像，它们具有相同的模态。因此，共享网络中的所有参数适合于这些任务。



#### 双流结构（Two-stream Structure）

**组成：**

**有两个输入**，对应于两个不同域中的数据。在较浅的层中，网络的参数对于每个域都是特定的。在较深的层中，使用共享参数。



<font color='red'>**作用：**</font>

与单流结构相比，双流结构实现了两件事：域自适应（domain adaptation）和区分特征学习（discriminative feature learning）。

**<font color='red'>假设特定于域的网络（the domain-specific network）可以提取不同域的共享特征，然后共享网络（the shared network）可以提取用于匹配的区分特征（Why？）</font>**

------

答：



------



#### 非对称全连接层结构（Asymmetric FC Layer Structure）

**组成：**

如图3中的第三个网络所示，该结构共享除最后一个FC层之外的几乎所有参数。



**作用：**

该设计假设不同域的特征提取可以是相同的，并且在特征级别实现了域自适应。这种特征提取和域自适应的顺序不同于双流结构。



#### 双流变单流

<img src="C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221201105345340.png" alt="image-20221201105345340" style="zoom:200%;" />

<img src="C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221201105415926.png" alt="image-20221201105415926" style="zoom:200%;" />

**原因：**

考虑到双流结构和非对称FC层结构，它们是手动设计的，并在训练期间固定。此外，两个域的域特定结构是解耦的，而共享结构是完全相同的。

相反，如果一个流结构可以隐式学习该结构，则对应于不同域的隐式结构通过共享节点和共享偏置参数（等式（4）和（5））部分耦合，这可以在跨模态匹配任务的训练中提供更大的灵活性。



**思路：**

对于跨模态匹配任务，**<font color='red'>由于域偏移，域特定建模对于提取共享组件进行匹配非常重要</font>**。通常，在神经网络中，例如，双流和不对称FC层结构，这是通过特定领域的结构来建模**（domain-specific structures）**的。因此，我们打算分析单流网络中的特定领域建模。

假设由等式（3）定义的三种类型的节点存在于网络中，则单流网络可以隐式地学习和发展网络中的域特定和共享结构。

将每个层的输出节点分为三种类型：域1特定节点、域2特定节点和共享节点。分类取决于节点的响应是否特定于域。

![image-20221201105722759](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221201105722759.png)



**结果**：

![image-20221201110028304](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221201110028304.png)

### **主要贡献（3）深层零填充算法**

提出能够自动扩展域特定结构的**深层零填充算法（deep zero-padding）**

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020103112165676.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlbnJvdWRlYmFvemk=,size_16,color_FFFFFF,t_70#pic_center)

图中首先将RGB图像转为单通道灰度图，然后将其作为RGB-specific zero-padding的第一通道，使用全零填充第二通道；IR图像则作为IR-specific zero-padding的第二通道，全零填充第一通道。这样就得到两种二通道输入RGB-specific zero-padding和IR-specific zero-padding。
作者根据大量实验证明这种深度零填充的方式能够使网络扩展特定域节点（domain-specific nodes）更加灵活。通俗的说（个人理解）就是网络能够隐式的学习到两种模态的区别。

## 数据集SYSU-MM01

### 数据集介绍

SYSU-MM01包含由6台相机拍摄的图像，包括2个红外相机和4个RGB相机。详见下列表格，每一列：相机的索引号，拍摄地点，室内还是室外，白天还是黑夜，ID数目，每个ID有多少不同的连续RGB帧，每个ID有多少不同的连续IR帧。

![image-20221112101456500](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221112101456500.png)

下表为不同数据集的比较，下图中，相机1-3为室内场景，相机4-6为室外场景，每两列为同一个人。在本数据中，有491个人物ID，296个用于训练，99个用于验证，96个用于测试。

![](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221111204014460.png)

### **评估协议**

在SYSU-MM01数据集中有**491个有效的ID**。我们有一个固定的分割，使用**296个身份用于训练**，**99个用于验证**，**96个用于测试**。在**<font color='red'>训练期间</font>**，**训练集中**在**所有的照相机**中的**296人**的**所有图像**都**可以应用**。

**<font color='red'>在测试阶段</font>**，**IR图像**做**probe set**，**RGB图像**做**gallery set**。也就是说待**检索图像是红外图像**，而**检索库为RGB图像**（用IR搜索RGB。RGB相机的样品用于gallery，红外相机的样品用于probe。）我们设计了全搜索模式和室内搜索模式两种模式。对于所有搜索模式，RGB相机1、2、4和5用于gallery集，红外相机3和6用于probe集。对于室内搜索模式，使用RGB摄像头1和2（不包括室外摄像机4和5）用于gallery集，红外摄像机3和6用于probe。

对于这两种模式，我们都采用单镜头和多镜头设置（single shot / multi shot）。对于RGB相机下的每个身份，我们随机选择该身份的1/10个图像，以形成用于单镜头/多镜头设置的图库集（gallery set）。对于探头组（probe set），使用所有图像。给定探测图像（probe），通过计算探测图像（probe image）和画廊图像（gallery image）之间的相似性来进行匹配。请注意，在不同位置的摄像机之间进行匹配（位置如表2所示）。摄像机2和摄像机3位于同一位置，因此摄像机3的探测图像（probe）跳过摄像机2的画廊图像（gallery）。计算相似度后，我们可以根据相似度的降序获得排名列表。

这个数据集具有挑战性，因为一些人的图像是在室内环境中拍摄的，而有些是在室外环境中。它有491人，每个人至少被逮捕 两个不同的照相机。我们采用了单镜头全搜索模式评估协议，因为它是最具挑战性的情况。



## 心得感悟

本文首次提出跨模态行人重识别任务。作者提供了相关的数据集、网络结构、解决思路。

不过以现在的视角来看待这篇文章，我感觉作者提出的**<font color='red'>深度零填充算法</font>**有些冗余。

主要解决一个问题：如何提出共享特征。**<font color='red'>本文大量出现domain-specific的字眼</font>**，一开始让我晕头转向。

于是我便开始思考，跨模态行人重识别，简单来说就是跨模态检索任务。本质上就是**多模态的检索问题**。

关于多模态，我们需要关注的有：多模态的表示、转化、融合、对齐、协同学习

而在这里，我们会把重心放在多模态的表示这一问题上：我想要提取出红外图像和可见光图像间共有的特征。

一般而言，多模态表征学习，最直接的就是特征拼接。还有就是协同学习。

所谓协同学习，每个模态xi、xn，都有相应的映射f（xi）、g（xn），这些映射能把它们投影到同一个空间中。

问题是，如何学习这些映射？两种思路：**<font color='red'>相似度模型和结构化模型</font>**。度量学习很好理解，量化距离，拉近距离。结构化呢？

![image-20221122172737401](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221122172737401.png)

有了解决思路了，那作者是是使用什么工具来解决问题的呢？有三种选择，单流网络、双流网络、非对称FC网络。**<font color='red'>通过分析思考</font>**（How？），作者使用改进后的单流网络。这种单流网络相比于双流网络、非对称FC网络更加**<font color='red'>灵活</font>**，学习出更鲁棒的domain-specific structure。

在该网络中，我们会存在三种类型的节点，分别是

![image-20221122212222520](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221122212222520.png)

我们可以看到，domain2的数据如果输入到domain1-spcific结点，会输出零值。domain1数据输入domain2节点同理。这两种节点就是我们想要得到的domain-specific structure，能提取出共享的特征。

我们为了得到更多更鲁棒的domain-specific strurcture，提出了deep zero padding算法。具体思路如下。

![image-20221122213920352](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221122213920352.png)

![image-20221122213950411](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221122213950411.png)

**问题：**

1.

经过domain-specific sub-network得到特征后如何放入shared sub-network？

![image-20221203195913437](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221203195913437.png)

**答：**

直接堆叠在一起

20X32X30X21 + 200X32X30X21 =220X32X9X6

2.

与单流结构相比，双流结构实现了两件事：域自适应（domain adaptation）和区分特征学习（discriminative feature learning）。

**<font color='red'>假设特定于域的网络（the domain-specific network）可以提取不同域的共享特征，然后共享网络（the shared network）可以提取用于匹配的区分特征（Why？如何理解？）</font>**

**答：**

对于discriminative features for matching，关键就是match。跨模态行人重识别任务种，因为模态间的误差会极大地影响任务的实现。因此，我们需要消除模态间的差异。

在the domain-specific network提取不同域的共享特征后，我们要清楚的是，这些共享特征，是否work呢？因为存在模态差异，所以不一定有效。所以共享网络（the shared network）通过共享权值的方式去拟合模态间的差异，投影到同一个空间。

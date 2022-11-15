# Ancong Wu-2017-（ICCV）RGB-Infrared Cross-Modality Person Re-Identification

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

#### 单流结构（One-stream Structure）



#### 双流结构（Two-stream Structure）



#### 非对称全连接层结构（Asymmetric FC Layer Structure）



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

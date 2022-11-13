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

### 主要贡献（1）

**提出了跨模态行人重识别领域的一个常用数据集-----**-SYSU-MM01****
**数据集中共**491个行人id**，**296个id用来训练，99个id用来验证，96个id用来测试模型**，在**训练阶段**使用的**<font color='red'>**296个id既有RGB图像也有IR图像**</font>**，而**测试阶段使用<font color='red'>**IR图像做probe set，RGB图像做gallery set。也就是说待检索图像是红外图像，而检索库为RGB图像**</font>。除此之外，本文还设置了**两种模式—全局搜索和室内搜索**。顾名思义，全局搜索就是在所有场景数据中搜索待检索目标，而室内搜索即是仅在室内拍摄的RGB图像中搜索目标。

### **主要贡献（2）**

分析了现存的较流行的三种网络结构-------（单流结构、双流结构、非对称FC层结构）

![img](https://img-blog.csdnimg.cn/20201030141229347.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlbnJvdWRlYmFvemk=,size_16,color_FFFFFF,t_70#pic_center)

上图为论文评估中的四个网络结构。转换块的结构取决于基础网络。转换块和FC层的颜色指示是否共享参数。红色和蓝色表示特定参数，绿色表示共享参数

#### 单流结构（One-stream Structure）



#### 双流结构（Two-stream Structure）



#### 非对称全连接层结构（Asymmetric FC Layer Structure）



### **主要贡献（3）**

提出能够自动扩展域特定结构的**深层零填充算法（deep zero-padding）**

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020103112165676.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlbnJvdWRlYmFvemk=,size_16,color_FFFFFF,t_70#pic_center)

图中首先将RGB图像转为单通道灰度图，然后将其作为RGB-specific zero-padding的第一通道，使用全零填充第二通道；IR图像则作为IR-specific zero-padding的第二通道，全零填充第一通道。这样就得到两种二通道输入RGB-specific zero-padding和IR-specific zero-padding。
作者根据大量实验证明这种深度零填充的方式能够使网络扩展特定域节点（domain-specific nodes）更加灵活。通俗的说（个人理解）就是网络能够隐式的学习到两种模态的区别。
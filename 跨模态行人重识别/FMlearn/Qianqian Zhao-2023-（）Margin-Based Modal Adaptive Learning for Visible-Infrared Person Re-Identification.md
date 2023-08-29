# Qianqian Zhao-2023-（）Margin-Based Modal Adaptive Learning for Visible-Infrared Person Re-Identification

# 摘要



# 引言

存在问题（编故事，写论文看）



解决方案：

思路：



**具体实现：**

为此，我们在本文中提出了一种用于 VIPR 的基于边距的模态自适应学习 (MMAL，marginbased modal adaptive learning) 方法。在我们的 MMAL 方法中，我们应用三元组和标签平滑交叉熵函数来学习外观判别特征并优化**最大均值差异**以鼓励学习的特征具有模态不变性。**<font color='red'>与现有方法 [27,28] 不同，我们的 MMAL 方法并不专注于改进 MMD 以准确测量可见光和红外模态的差异分布，而是注意保持模态差异抑制和外观判别学习的良好平衡</font>**。因此，本文的主要创新之处在于，我们的 MMAL 方法设计了一种简单而有效的边际策略，以避免过度抑制模态差异，从而保护特征提高 VIPR 的判别能力



**主要贡献：**

（1）我们设计了一种基于边缘的模态自适应学习 (MMAL，marginbased modal adaptive learning) 方法来加入 VIPR 的优化模态差异和判别外观，它可以通过边缘最大均值差异 (M3D，a marginal maximum mean discrepancy (M3D) loss function) 损失函数平衡模态不变性和外观判别。 

(2) 在 RegDB [29] 和 RGBNT [30] 数据集上的实验结果表明，我们的方法获得了最先进的性能，例如，在 RegDB 数据集上，对于可见到红外检索模式，排名-1 准确率为 93.24%，平均准确率为 83.77%。



# 方法

![image-20230612162649207](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20230612162649207.png)

我们从两个主要方面描述我们的方法。

 (1) 基于边缘的模态自适应学习 (MMAL)，旨在学习模态不变但具有外观判别性的特征。

 (2) 基于深度网络的 VIPR 模型，解释了如何使用 MMAL 监督深度网络学习特征以及如何采用学习到的特征来实现 VIPR。

## Margin-Based Modal Adaptive Learning

MMAL 由**<font color='red'>两种类型的损失函数组成</font>**，即边际最大均值差异 (M3D) 和外观判别损失函数。前者负责模态不变性，后者负责外观判别函数。




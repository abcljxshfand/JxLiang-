# Haojie Liu-2023-Towards Homogeneous Modality Learning and Multi-Granularity Information Exploration for Visible-Infrared Person Re-Identification

# 摘要





# 引言

存在的问题



解决的思路



具体方案



主要贡献

我们尝试了一条探索不足但意义重大的研究路径来解决跨模态问题。特别是，我们创建了一个统一的中间模态图像空间来嵌入同质模态信息，从而在可见域和红外域之间建立连接。我们的中间模态空间 (AGM) 完全可视化、高保真且易于复制。我们相信 AGM 具有进一步提升跨模态检索性能的巨大潜力。

• 为了进一步阐明跨模态匹配的挑战，我们首次引入了一个称为亮度差距的新概念。这个概念导致了**<font color='red'>灰度归一化（GN）</font>**，这是一种基于样式的归一化方法，能够抑制红外图像的亮度变化并进一步减轻模态差异。

• 我们研究了**多粒度特征学习问题**，并制定了更稳健的头肩描述符来支持行人重识别匹配。头肩部分通过有区别的外观线索有效地增强了人的信息，以构建高维融合特征，从而获得具有竞争力的 Re-ID 性能。

• 开发了一种具有精心设计的**闭环交互式正则化的同步学习策略（SLS）**，以优化全局和头肩信息的潜在互补优势，促使网络获得更多判别特征以进行正确分类。



# 方法

![image-20230613100601601](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20230613100601601.png)
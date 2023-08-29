# (中大)Jiawei Feng-2023-(CVPR)Shape-Erased Feature Learning for Visible-Infrared Person Re-Identification

# 摘要

**问题：**

身体形状是 VI-ReID 的重要模态共享线索之一。**为了挖掘更多不同的模态共享线索**，我们期望在学习特征中擦除与身体形状相关的语义概念可以迫使 ReID 模型提取更多和其他模态共享特征用于识别。



**本文提出的方法：**

我们提出了**形状擦除特征学习范例**，该范例将两个正交子空间中的模态共享特征去相关。

**联合学习**一个子空间中的形状相关特征和正交补充中的形状擦除特征，实现了形状擦除特征和身份丢弃体形信息之间的条件互信息最大化，从而显式增强了学习表示的多样性。



# 引言





# 方法

![image-20230608113603827](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20230608113603827.png)
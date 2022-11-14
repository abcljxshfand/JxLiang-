# 模态互转  //  基于生成对抗网络

## 跨模态行人重识别需要解决的问题（面临哪些挑战）

（1）解决可见光-红外图像间的模态差异

（2）数据集单一且规模较小



## 什么是模态互转？

可见光图象和红外图像这两种图像间的相互转换。

![image-20221114162119188](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221114162119188.png)

## 为什么要模态互转？

把跨模态问题转化为单模态问题。消除模态间的差异。



## 怎么样进行模态互转？

相关文献：

[189] Z. Wang, Z. Wang, Y . Zheng, Y .-Y . Chuang, and S. Satoh, “Learning to reduce dual-level discrepancy for infrared-visible person re-identification,” in CVPR, 2019, pp. 618–626.

[190] G. Wang, T. Zhang, J. Cheng, S. Liu, Y . Yang, and Z. Hou, “Rgbinfrared cross-modality person re-identification via joint pixel and feature alignment,” in ICCV, 2019, pp. 3623–3632.

**训练图像级子网络将红外图像转换为可见图像，将可见图像转换为红外图像。通过图像级子网络，我们可以统一具有不同模态的图像的表示。借助于统一的多光谱图像，训练特征级子网络，以通过特征嵌入来减少剩余的外观差异**

![image-20221114162152867](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221114162152867.png)
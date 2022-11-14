# Wang-2019-（CVPR）Learning to Reduce Dual-Level Discrepancy for Infrared-Visible Person Re-Identification

## 摘要

可见光图像和红外图像之间的模态差异使得IV-Reid任务更难解决。本文提出了一种模型Dual-level Discrepancy Reduction Learning (D2RL) 来解决这两种模态之间的差异。



## 引言

IV-Reid遇到的困难和挑战

<img src="C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221114164146369.png" alt="image-20221114164146369"  />

左图，存在两个图像域，黄色的是可见光图像域，绿色的是红外图像域。通过比较可知，模态间的差异所造成的影响远比外表（姿态、视觉角度、遮挡）差异要大。

右图，在RegDB数据集下，分别对IR-IR、V-V、IV行人重识别任务使用Feature constrain后获得的性能图。红色间隙是可见-可见和红外单模态Reid下的性能差距。黑色间隙表示，跨模态Reid和单模态Reid之间的性能差距。可见，跨模态的Reid要比单模态困难得多。



<img src="C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221114165903226.png" alt="image-20221114165903226"  />

对D2RL的图像解释：上图存在三个空间，分别为：图像空间、统一空间、特征空间。在图像空间中，x和y分别表示可见光图像域和红外图像域，它们两个间存在巨大的差异。以往的方法是通过将x和y直接映射到特征空间并使用特征约束来消除这两种模态间的差异。而我们的方法是先把x和y映射到统一空间内，然后再映射到特征空间。从原始空间映射到统一空间，两者间隙和原始空间相比变小许多了
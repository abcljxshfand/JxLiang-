# 叶茫-2021-（**TPAMI**）Deep Learning for Person Re-identification:A Survey and Outlook

![image-20221203131310222](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221203131310222.png)

![image-20221203131356511](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221203131356511.png)

![image-20221203131420590](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221203131420590.png)

参考：

https://blog.csdn.net/sinat_31253573/article/details/109181606

https://www.cnblogs.com/wangchangshuo/p/16133241.html

https://zhuanlan.zhihu.com/p/342249413

## ![image-20221106105133271](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221106105133271.png)

------

## Closed-world person Re-ID

通常，一个标准的封闭世界Re-ID系统包含三个主要组成部分

特征表示学习（§2.1），其重点是开发特征构建策略；

深度度量学习（§2.2），旨在设计具有不同损失函数或抽样策略的训练目标；

以及排名优化（§2.3），它专注于优化检索到的排名列表。

### 特征表示学习

![image-20221128101208615](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221128101208615.png)

全局特征：学习每个图像的全局表示

局部特征：聚合部分级局部特征，以形成每个人物图像的组合表示。（it aggregates part-level local features to formulate a combined representation for each person image [75], [76], [77];）

------

[75] L. Zhao, X. Li, Y . Zhuang, and J. Wang, “Deeply-learned partaligned representations for person re-identification,” in CVPR, 2017, pp. 3219–3228.

[76] H. Yao, S. Zhang, R. Hong, Y . Zhang, C. Xu, and Q. Tian, “Deep representation learning with part loss for person reidentification,” IEEE T ransactions on Image Processing (TIP), 2019.

[77] Y . Sun, L. Zheng, Y . Yang, Q. Tian, and S. Wang, “Beyond part models: Person retrieval with refined part pooling,” in ECCV, 2018, pp. 480–496.

------

辅助特征：使用辅助信息学习特征表示

视频特征：使用多个图像帧和时间信息学习视频表示

#### 全局特征表示学习

**注意力机制：**

文献中已经广泛研究了注意力方案来增强表征学习[85]

[85] F. Yang, K. Yan, S. Lu, H. Jia, X. Xie, and W. Gao, “Attention driven person re-identification,” Pattern Recognition, vol. 86, pp.

143–155, 2019.

**1） 第一组：关注人物形象。**

典型的策略包括像素级关注[86]和信道方向特征响应重新加权[86]、[87]、[88]、[89]或背景抑制[22]。空间信息集成在[90]中。

**2） 第2组：关注多人图像。**

[91]中提出了一种上下文感知注意特征学习方法，结合了序列内和序列间注意，用于成对特征对齐和细化。

[92]，[93]中增加了注意力一致性属性。

群体相似性[94]，[95]是另一种利用跨图像注意力的流行方法，涉及多个图像用于局部和全局相似性建模。第一组主要增强对未对准/不完美检测的鲁棒性，第二组通过挖掘多个图像之间的关系来改进特征学习。



#### 局部特征表示学习

它学习部分/区域聚集特征，使其对未对准具有鲁棒性[77]，[96]。

[77] Y . Sun, L. Zheng, Y . Yang, Q. Tian, and S. Wang, “Beyond part models: Person retrieval with refined part pooling,” in ECCV, 2018, pp. 480–496.

[96] R. R. Varior, B. Shuai, J. Lu, D. Xu, and G. Wang, “A siamese long short-term memory architecture for human re-identification,” in ECCV, 2016, pp. 135–153.

身体部分要么通过人类解析/姿势估计自动生成（组1），要么大致水平划分（组2）。（The body parts are either automatically generated by **human parsing/pose estimation** (Group 1) or roughly **horizontal division** (Group 2).）





## AGW

<img src="C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221201154200754.png" alt="image-20221201154200754" style="zoom:200%;" />

具体而言，我们的新基线是在BagTricks[122]的基础上设计的，AGW包含以下三个主要改进组件：

[122] H. Luo, W. Jiang, Y . Gu, F. Liu, X. Liao, S. Lai, and J. Gu, “A strong baseline and batch normneuralization neck for deep person reidentification,” arXiv preprint arXiv:1906.08332, 2019.

backbone：ResNet50

<font color='red'>**三个主要改进的部分：**</font>

1）Non-local注意力机制的融合；

2）Generalized-mean (GeM) Pooling的细粒度特征提取；

3）加权正则化的三元组损失（Weighted Regularization Triplet (WRT) loss）



### Non-local Attention (Att) Block

我们采用强大的非局部关注块[246]来获得所有位置的特征的加权和，表示为

![image-20221201154839646](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221201154839646.png)

其中Wz是要学习的权重矩阵，φ（·）表示非局部操作，+xi制定了剩余学习策略。有关详细信息，请参阅[246]。我们采用[246]中的默认设置来插入非本地关注块。

**[246] X. Wang, R. Girshick, A. Gupta, and K. He, “Non-local neural networks,” in CVPR, 2018, pp. 7794–7803.**



### Generalized-mean (GeM) Pooling

为一种**细粒度**的实例检索（a fine-grained instance retrieval），广泛使用的最大池或平均池无法捕获特定于域的区分特征（the domain-specific discriminative features）。我们采用了一个可学习的池化层，名为广义平均（GeM）池化[247]，公式如下：

![image-20221201160004435](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221201160004435.png)

**[247] F. Radenovi´c, G. Tolias, and O. Chum, “Fine-tuning cnn image retrieval with no human annotation,” IEEE TP AMI, vol. 41, no. 7, pp. 1655–1668, 2018.**



### Weighted Regularization T riplet (WRT) loss

除了具有softmax交叉熵的基线身份损失之外，我们与另一个加权正则化三重态损失积分。

![image-20221201155955545](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221201155955545.png)

上述加权正则化继承了正负对之间的相对距离优化的优点，但它避免了引入任何额外的边缘参数。

我们的加权策略类似于[248]，但我们的解决方案没有引入额外的超参数。

[248] X. Wang, X. Han, W. Huang, D. Dong, and M. R. Scott, “Multisimilarity loss with general pair weighting for deep metric learning,” in CVPR, 2019, pp. 5022–5030.
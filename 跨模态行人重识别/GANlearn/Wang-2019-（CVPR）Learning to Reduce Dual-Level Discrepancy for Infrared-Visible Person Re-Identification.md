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



## 方法

![image-20221114210602617](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221114210602617.png)

网络结构如上图所示，其中x是可见光图像，y是红外图像，由上图可知，存在两个子网络，分别是The image-level discrepancy reduction sub-network TI 和 the feature-level discrepancy reduction sub-network TF，TI的作用是将可见光图像（红外图像）投影到统一空间，这样可以减少模态差异。然后，利用TF来消除剩余的外观差异。这两个子网络以端到端的方式进行级联并联合优化。

**对比以前传统的方法**：通过feature embeding 将可见光图像和红外图像投影到特征空间，即

![image-20221114211731757](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221114211731757.png)

然后使用它们之间的距离返回检索列表。

### Image-level discrepancy reduction——Tl

**构成：**使用两个变分自编码器（VAEs）进行<font color='red'>**风格分离（Style disentanglement）**</font>，使用两个GAN进行<font color='red'>**域特定图像生成（domain specific image generation）**</font>。

**（注：**变分自编码器和GAN的目标基本一致，即构建一个从隐变量 *Z* 生成目标数据 *X* 的模型，进行分布之间的变换。）

**作用：**将可见光（红外）图像转换对应的红外（可见光）图像，然后将对应的可见光，红外图像合并形成<font color='red'>**多光谱图像**</font>，即<font color='red'>**统一表示（unified space）**</font>。

#### 风格分离（Style disentanglement）

**disentanglement：解纠缠，又称作解耦**，就是将原始数据空间中纠缠着的数据变化，**变换到一个好的表征空间中**，在这个空间中，不同要素的变化是可以彼此分离的。比如，人脸数据集经过编码器，在潜变量空间Z中，我们就会获得人脸是否微笑、头发颜色、方位角等信息的<font color='red'>**分离表示**</font>，我们把这些分离表示称为Factors。 解纠缠的变量通常包含可解释的语义信息，并且能够反映数据变化中的分离的因子。在生成模型中，我们就可以根据这些分布进行特定的操作，比如改变人脸宽度、添加眼镜等操作。

![image-20221114213433326](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221114213433326.png)



#### 域特定图像生成（domain specific image generation）
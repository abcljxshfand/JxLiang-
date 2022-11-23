# Wang-2019-（CVPR）Learning to Reduce Dual-Level Discrepancy for Infrared-Visible Person Re-Identification

**（注：**在论文中，黄色标记是读论文时候的main body，绿色标记是背景、名词解释，红色标记是重点，紫色标记是相关工作，蓝色是作用、目的与意义，棕色是疑问、问题。）

## 摘要

可见光图像和红外图像之间的模态差异使得IV-Reid任务更难解决。本文提出了一种模型Dual-level Discrepancy Reduction Learning (D2RL) 来解决这两种模态之间的差异。



## 引言

IV-Reid遇到的困难和挑战

<img src="C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221114164146369.png" alt="image-20221114164146369"  />

左图，存在两个图像域，黄色的是可见光图像域，绿色的是红外图像域。通过比较可知，模态间的差异所造成的影响远比外表（姿态、视觉角度、遮挡）差异要大。

右图，在RegDB数据集下，分别对IR-IR、V-V、IV行人重识别任务使用Feature constrain后获得的性能图。红色间隙是可见-可见和红外单模态Reid下的性能差距。黑色间隙表示，跨模态Reid和单模态Reid之间的性能差距。可见，跨模态的Reid要比单模态困难得多。

首先在行人重识别任务中，在同一种类型的多个非重叠区域相机条件下所拍摄的图片，会存在视角、姿态、尺寸等外观差异。而在可见光-红外跨模态行人重识别任务中，因为光源问题，摄像头在可见光模式和红外模式下的成像方式不同，这就导致了可见光图片和红外图像之间存在着明显的模态差异。

经过研究表明，我们把同一个行人在不同类型的图片中存在图像域差异称为δd，在相同类型的图片中存在的外观差异成为和δa，通过量化，δd远比δa要大得多，这说明了图像域间的差异对任务有很大的影响，不同图像域间类内距比类间距还要大，这就导致了VI-Reid和单模态Reid相比，性能差距大，效果不佳。





<img src="C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221114165903226.png" alt="image-20221114165903226"  />

对D2RL的图像解释：上图存在三个空间，分别为：图像空间、统一空间、特征空间。在图像空间中，x和y分别表示同一个行人在可见光图像域和红外图像域下的图像，它们两个间存在巨大的差异。以往的方法是通过将x和y直接映射到特征空间并使用特征约束来消除这两种模态间的差异。而我们的方法是先把x和y映射到统一空间内消除图像域间的差异。在通过统一图像表示后，虽然已经消除图像域间的差异，但仍存在外观差异，这时我们可以将它们映射到特征空间消除外观差异。从原始空间映射到统一空间，两者间隙和原始空间相比变小许多了



## related work

**以往的VI-Reid**

据我们所知，所有先前的方法[21，22，23，2]都将模态差异δm视为外观差异δa的一部分，并尝试使用大多数传统重新识别方法所采用的特征级约束来减少混合差异δm+δa。



## 方法

![image-20221114210602617](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221114210602617.png)

网络结构如上图所示，其中x是可见光图像，y是红外图像，由上图可知，存在两个子网络，分别是The image-level discrepancy reduction sub-network TI 和 the feature-level discrepancy reduction sub-network TF，TI的作用是将可见光图像（红外图像）投影到统一空间，这样可以减少模态差异。然后，利用TF来消除剩余的外观差异。这两个子网络以端到端的方式进行级联并联合优化。

**对比以前传统的方法**：通过feature embeding 将可见光图像和红外图像投影到特征空间，即

![image-20221114211731757](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221114211731757.png)

然后使用它们之间的距离返回检索列表。

### Image-level discrepancy reduction——Tl

**构成：**使用两个变分自编码器（VAEs）进行<font color='red'>**风格分离（Style disentanglement）**</font>，使用两个GAN进行<font color='red'>**域特定图像生成（domain specific image generation）**</font>。

**（注：**变分自编码器和GAN的目标基本一致，即构建一个从隐变量 *Z* 生成目标数据 *X* 的模型，进行分布之间的变换。）

**作用：**消除模态差异。将可见光（红外）图像转换对应的红外（可见光）图像，然后将对应的可见光，红外图像合并形成<font color='red'>**多光谱图像**</font>，即<font color='red'>**统一表示（unified space）**</font>。

#### 风格分离（Style disentanglement）

------

**（注：disentanglement：解纠缠，又称作解耦，**就是将原始数据空间中纠缠着的数据变化，**<font color='red'>变换</font>到一个好的表征空间中**，在这个空间中，不同要素的变化是可以彼此分离的。比如，人脸数据集经过编码器，在潜变量空间Z中，我们就会获得人脸是否微笑、头发颜色、方位角等信息的<font color='red'>**分离表示**</font>，我们把这些分离表示称为Factors。 解纠缠的变量通常包含可解释的语义信息，并且能够反映数据变化中的分离的因子。在生成模型中，我们就可以根据这些分布进行特定的操作，比如改变人脸宽度、添加眼镜等操作。

![image-20221114213433326](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221114213433326.png)

------

**构成：**两对encoder-decoder，分别是：VAEv＝{Ev，Gv}和VAEi＝{Ei，Gi}

**作用：**两种图片输入到VAE后映射至同一空间

**过程：**以VAEv为例，输入可见光图像x到Ev后，Ev会把x映射成一个潜在变量z，然后再把z输入到Gv中，Gv会通过z重构成x对应的红外图像。

**损失函数：**

![image-20221115160232622](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221115160232622.png)



该loss中，λ0和λ1为超参数，将KL散度和l1范数加权，弥补了图像和重建图像之间的不一致性，也使得输出图像可以更加清晰。

#### 域特定图像生成（domain specific image generation）

**构成：**两个GAN，分别是GANv＝｛Gv，Dv｝和GANi＝｛Gi，Di｝

**作用：**生成对应域的图像

**过程：**generation+discrimination

![image-20221115204104294](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221115204104294.png)

**损失函数：**

![image-20221115203347396](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221115203347396.png)

这种损失被用来增强翻译后的图像在可见域内的相似性。

#### Cycle-consistency

利用CycleGAN中的思路，将图像风格迁移前后进行一致性检验，其loss函数为

![image-20221115203537014](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221115203537014.png)

使得输出的图像经过一个生成器可以生成一个与原输入图像尽可能一致的图像。

#### Modality unification

模态统一有三种可能的选择，即将图像与红外模态、可见光模态或多光谱模态统一起来。其中框架中链接TI和TF的部分为多光谱图像组成的unified space。

![image-20221115203840143](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221115203840143.png)

#### Object function

总loss如下所示：

![image-20221115203656897](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221115203656897.png)

### Feature-level discrepancy reduction——TF

**作用：**消除图像内容差异。可看作单模态的行人重识别问题，主要不同之处在于输入的形式发生了变化：在经过Tl子网络之后，文中的做法是将 RGB（红外）图像和生成的红外（RGB）图像在通道上合并成一张图像，这样一张图像变成了 4 个通道。因此，需要将通常所用的网络结构（如：Resnet50）第一层卷积的输入通道数改为 4，



## 实验



![image-20221115205941043](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221115205941043.png)

![image-20221115210114699](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221115210114699.png)

![image-20221115205237528](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221115205237528.png)

![image-20221115210158277](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221115210158277.png)
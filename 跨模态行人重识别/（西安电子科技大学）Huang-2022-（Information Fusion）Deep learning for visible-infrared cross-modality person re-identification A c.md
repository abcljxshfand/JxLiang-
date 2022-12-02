# （西安电子科技大学）Huang-2022-（Information Fusion）Deep learning for visible-infrared cross-modality person re-identification: A comprehensive review

**（注：**在论文中，黄色标记是读论文时候的main body，绿色标记是背景、名词解释，红色标记是重点，紫色标记是相关工作，蓝色是作用、目的与意义，棕色是疑问、问题。）

## 摘要

首先，我们阐明了VI ReID的重要性、定义和挑战。

其次，也是最重要的一点，我们详细分析了现有VI ReID方法的动机和方法。因此，我们将为这些最先进的（SOTA）VI ReID模型**<font color='red'>提供一个全面的分类，包括4个类别和8个子项。</font>**

之后，我们详细介绍了一些广泛使用的数据集和评估指标。

接下来，在基准数据集上对SOTA方法进行了**全面比较**。基于这些结果，我们指出了现有方法的局限性。最后，我们概述了该领域的挑战和未来的研究趋势。

![image-20221123103036791](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221123103036791.png)

## 引言

### 困难与挑战

**VI ReID的挑战主要在于两个方面**：跨模态变化和模态内变化（cross-modality variations and intra-modality variations）。

跨模态变化：可见光和红外图像之间的模态差异导致了跨模态变化。具体而言，如下图所示，可见光和红外图像由于**其不同的成像原理而本质上不同**。可见图像通常有三个通道，包含丰富的视觉信息，如形状、位置、颜色和纹理。而红外图像具有一个通道，并且主要包含轮廓和位置信息。

![image-20221129163448934](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221129163448934.png)

**<font color='red'>跨模态变化对VI ReID产生了不利影响：</font>**

首先，如图3（a）所示，它将导致RGB模态和IR模态之间的**不同特征分布**，这进一步导致同一身份的**不同模态之间的若干未对准问题**。例如，这将使同一身份内的跨模态差异大于不同身份内的模态差异。

其次，如图3（b）所示，**它使更多的人歧视信息受到干扰（it enables much more person-discriminative information to be interfered）**。例如，在VV ReID中，颜色信息是识别不同人物的最重要的外观线索之一。然而，在VIReID中，几乎不能使用颜色信息。这意味着VI ReID任务中的有用信息远远少于VV ReID任务，这使得VI ReID更具挑战性。

与VV ReID类似，VI ReID中的模态内变化也由视角、姿势和曝光中的人物变化引起。这进一步增加了VI ReID的难度。

![image-20221129163720619](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221129163720619.png)

## 方法总结

输入的可见图像或红外图像首先通过使用一些数据增强策略来增强（在训练策略中），然后被馈送到ReID网络以提取相应的人物特征。最后，匹配来自不同图像的人物特征。

应该注意的是，在本文中，我们将从单模态图像中提取的特征称为单模态特征，可以进一步将其分为模态特定特征和模态共享特征。其中，特定于模态的特征是单模态图像所特有的，例如可见图像中的颜色。而模态共享特征是可见图像和红外图像中共同存在的特征。

![image-20221124112714662](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221124112714662.png)

如图4所示，根据它们处理跨模态变化和模态内变化的方式，我们将这些模型分为四类，包括模态共享特征学习、模态特定信息补偿、辅助信息和数据增强。

基于模态共享特征学习的模型旨在提取那些有区别的模态共享特征，并丢弃那些模态特定特征，以同时解决VI ReID的跨模态变化和模态内变化。

然而，如图4所示，基于模态特定信息补偿的模型首先通过使用一些生成模型（如生成对抗网络（GAN））从现有模态生成缺失模态，以减少跨模态变化。然后，他们从生成的原始信息中提取有区别的人物特征，以处理模态内变化。

不同的是，基于辅助信息的模型试图使用一些辅助信息，例如一些人的面具和属性，以便于提取更有区别的人特征。

除了那些广泛使用的数据增强策略，如随机调整大小、裁剪和水平翻转，基于数据增强的模型试图开发一些专用的数据增强战略，以提取VI ReID网络中更多与人相关的特征。

## Modality-shared feature learning

![image-20221124112633945](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221124112633945.png)





### Feature projection

![image-20221124112735500](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221124112735500.png)

![image-20221125110511178](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221125110511178.png)

#### Exploring multi-level features

（**注：**文章-方法名称-目标-具体内容-关键问题）

**<font color='red'>总结</font>：**<font color='red'>**一般来说，低级特征主要包含详细信息，高级特征主要包含语义信息。因此，这两种特征是互补的。考虑到这一点，一些工作试图利用共享特征空间中的多层次互补信息来提取更具鉴别性的模态共享特征**</font>

**X. Xiang, N. Lv, Z. Yu, M. Zhai, A.E. Saddik, Cross-modality person reidentification based on dual-path multi-branch network, IEEE Sens. J. 19 (2019) 11706–11713.**

在[17]中，将**多粒度网络（MGN）**集成到双路径框架中，以组合多级特征并进一步提取区分模态共享特征。



**D. Cheng, X. Li, M. Qi, X. Liu, C. Chen, D. Niu, Exploring cross-modality commonalities via dual-stream multi-branch network for infrared-visible person re-identification, IEEE Access 8 (2020) 12824–12834.**

Cheng等人[15]提出了一种**双流多层对应融合网络（DMCF）**DMCF首先通过参数共享特征提取器分别从RGB和IR图像中提取不同级别的单模态特征。然后，它采用参数共享的多分支结构来融合来自两种模态的相同级别的特征，并使用不同的多粒度划分方法来增强模态共享特征。



**H. Liu, J. Cheng, W. Wang, Y. Su, H. Bai, Enhancing the discriminative feature learning for visible-thermal cross-modality person re-identification, Neurocomputing 398 (2020) 11–19.**

Liu等人[16]提出在他们的共享网络中**使用一些跳过连接**来探索中层特征，以提高学习模态共享特征的可分辨性。



#### Mining global and local information

挖掘全局和局部信息可以使VI-ReID模型对未对准具有鲁棒性。（**Mining** global and local information can make VI-ReID models robust against misalignment.）

因此，许多研究人员致力于如何通过专用参数共享网络来有效地挖掘共享特征空间中的两种类型的信息。

22  21 23 28 <font color='red'>**总结：一些工作侧重于提取有区别的零件特征**</font>：

**S. Liu, J. Zhang, Local alignment deep network for infrared-visible cross-modal person reidentification in 6G-enabled internet of things, IEEE Internet Things J. 8 (20) (2021) 15170–15179.**

Liu等人[22]首先从输入图像中提取模态共享特征，然后**利用统一的分割策略**来获得更具鉴别性的模态共享人部分特征。



**Z. Wei, X. Yang, N. Wang, B. Song, X. Gao, ABP: Adaptive body partition model for visible infrared person re-identification, Proceedings of the IEEE International Conference on Multimedia and Expo (2020) 1–6.**

Wei等人[21]在他们的参数共享网络中**提出了自适应身体分割（ABP）模块**，**以自动检测和区分有效的部分表示**。



**Z. Wei, X. Yang, N. Wang, X. Gao, Flexible body partition-based adversarial learning for visible infrared person re-identification, IEEE Trans. Neural Netw.**

类似地，Wei等人[23]提出了一种基于柔性身体分割（FBP）模型的对抗性学习方法（FBP-al），该方法还可以根据行人图像的特征图自动区分部分表示。



**H. Park, S. Lee, J. Lee, B. Ham, Learning by aligning: Visible-infrared person re-identification using cross-modal correspondences, in: Proceedings of the IEEE Conference on Computer Vision, 2021, pp. 12026–12035.**

Park等人[28]提出在训练期间利用交叉模态图像之间的密集对应，以提取一些有区别的像素级局部特征。



27 20 24<font color='red'>**总结：不同的是，其他工作研究了联合开发VI-ReID全球和本地信息的方法**</font>。

**K. Chen, Z. Pan, J. Wang, S. Jiao, Z. Zeng, Z. Miao, Joint feature learning network for visible-infrared person re-identification, in: Proceedings of the Chinese Conference on Pattern Recognition and Computer Vision, 2020, pp. 652–663.**

Chen等人[27]将局部特征与全局特征相结合，以获得最终的模态共享特征。



**C. Zhang, H. Liu, W. Guo, M. Ye, Multi-scale cascading network with compact feature learning for RGB-infrared person re-identification, in: Proceedings of the International Conference on Pattern Recognition, 2021, pp. 8679–8686.**

Zhang等人[20]提出了一种新的多尺度部分感知级联框架（MSPAC），以级联方式从局部到全局聚合多尺度细粒度特征，这导致了包含丰富和增强的语义特征的统一表示。



**H. Liu, Y. Chai, X. Tan, D. Li, X. Zhou, Strong but simple baseline with dualgranularity triplet loss for visible-thermal person re-identification, IEEE Signal Process. Lett. 28 (2021) 653–657.**

Liu等人[24]在其参数共享网络中提出了一种双粒度三重丢失模块，其中提取的模态共享特征可以从精细（局部）到粗粒度（全局）的方式导出，因此更具鉴别性



#### Employing attention mechanism

37 34 31 30 29 32<font color='red'>**总结：在VI-ReID模型中，注意机制主要用于参数共享网络，以从那些与人相关的区域中提取辨别模态共享特征。**</font>

**[37] Q. Wu, P. Dai, J. Chen, C.-W. Lin, Y. Wu, F. Huang, B. Zhong, R. Ji, Discover cross-modality nuances for visible-infrared person re-identification, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 4330–4339.**

具体而言，Wu等人[37]提出**了一种新的联合模态和模式对齐网络（MPANet）**，该网络使用基于注意力机制的模式对齐模块来发现VI ReID的行人不同模式中的跨模态细微差别，从而提取更具区别性的特征。



**Y. Wu, S. Wan, D. Wu, C. Wang, C. Yuan, X. Qin, H. Wu, X. Zhao, Position attention-guided learning for infrared-visible person re-identification, in: Proceedings of the International Conference on Intelligent Computing, 2020, pp. 387–397.**

受自我注意机制[51]的启发，Wu等人[34]提出了**一种位置注意力引导学习模块（PALM）**，以捕获不同人位置之间的长距离依赖性，以增强局部特征的可分辨性。



**Y. Cheng, X. Li, G. Xiao, W. Ma, X. Gou, Dual-path deep supervision network with self-attention for visible-infrared person re-identification, in: Proceedings of the IEEE International Symposium on Circuits and Systems, 2021, pp. 1–5.**

类似地，Cheng等人[31]提出了一种双路径深度监控网络（DDSN），该网络还使用了自我关注机制[51]来捕获输入图像的人物区域内的潜在上下文信息。



**Y. Li, H. Xu, Deep attention network for RGB-infrared cross-modality person re-identification, J. Phys. Conf. Ser. 1642 (1) (2020) 012015.**

Li等人[30]引入了空间注意力模块和通道注意力模块，以分别探索空间和通道维度中的视觉特征依赖性。



**J. Jiang, K. Jin, M. Qi, Q. Wang, J. Wu, C. Chen, A cross-modal multi-granularity attention network for RGB-IR person re-identification, Neurocomputing 406 (2020) 59–67.**

Jiang等人[29]提出了一个“蝴蝶”注意模块，以便于提取模态之间的共同局部特征，并将其与全局特征进一步融合以进行匹配。



**X. Wei, D. Li, X. Hong, W. Ke, Y. Gong, Co-attentive lifting for infrared-visible person re-identification, in: Proceedings of the ACM International Conference on Multimedia, 2020, pp. 1028–1037.**

此外，注意机制也可用于减少情态差异。例如，Wei等人[32]提出了一种共同关注机制，通过学习和协作模态共享特征来弥合两种模态之间的差距，从而显著减少模态差异。



#### Using graph convolution networks (GCNs)

39-44**<font color='red'>总结：考虑到GCN在建模关系方面的强大能力，在基于特征投影的VIReID模型中，GCN主要用于（1）捕获不同部分特征之间的关系，以增强模态共享特征；（2） 探索不同模态之间的关系，以减少VI ReID中的跨模态变体</font>**



**[38] Y. Feng, F. Chen, Y. mu Ji, F. Wu, J. Sun, Efficient cross-modality graph reasoning for RGB-infrared person re-identification, IEEE Signal Process. Lett.**

Feng等人[38]首先开发了一个模态相似性模块，以减少模态差距并保存身份信息。然后，他们构建了一个完全连通的图来链接全局和局部特征，以探索图上的身份-组成-关系推理。



**[40] H. Zhou, C. Huang, H. Cheng, A relation network design for visible thermal person re-identification, in: Proceedings of the International Conference on Intelligent Computing and Signal Processing, 2021, pp. 511–515.**

Zhou等人[40]提出了一种基于GCN的关系模块，以捕获每个模态内局部身体部位的关系，从而能够区分相应部位中具有相同属性的身份。



**[42] M. Jia, Y. Zhai, S. Lu, S. Ma, J. Zhang, A similarity inference metric for RGB-infrared cross-modality person re-identification, in: Proceedings of the International Joint Conference on Artificial Intelligence, 2020, pp. 1026–1032.**

Jia等人[42]提出了一种相似性推断度量（SIM），该度量通过相似性图推理和相互最近邻推理利用模态内相似性来匹配硬阳性样本。



**[39] M. Ye, J. Shen, D. J. Crandall, L. Shao, J. Luo, Dynamic dual-attentive aggregation learning for visible-infrared person re-identification, in: Proceedings of the European Conference on Computer Vision, 2020, pp. 229–247.**

Ye等人[39]引入了一种跨模态图结构化注意力方案，通过挖掘两种模态的人物图像之间的图形关系来增强特征表示。



**[43] Y. Junhui, M. Zhanyu, X. Jiyang, N. Shibo, L. Kongming, G. Jun, DF2AM: Dual-level feature fusion and affinity modeling for RGB-infrared cross-modality person re-identification, 2021, arXiv preprint arXiv:2104.00226.**

Yin等人[43]提出了一种简单但有效的相似性推断，以获得最佳的内模态和跨模态图像匹配。



**[41] J. Zhang, X. Li, C. Chen, M. Qi, J. Wu, J. Jiang, Global-local graph convolutional network for cross-modality person re-identification, Neurocomputing 452 (2021) 137–146.**

Zhang等人[41]首先使用局部图模块探索不同身体部位的潜在关系，然后使用全局图模块获得两种模态的上下文信息，以减少模态差异。



**[44] Y. Feng, F. Chen, J. Yu, Y. Ji, F. Wu, S. Liu, Homogeneous and heterogeneous relational graph for visible-infrared person re-identification, 2021, arXiv preprint arXiv:2109.08811.**

Feng等人[44]首先为单个模态设计了同构结构图，以学习身份相关特征，然后提出了异构图对齐模块（HGAM），以实现跨模态对齐。



#### Optimizing batch normalization layer

8 45**<font color='red'>总结：优化批次归一化层：通过优化参数共享网络中的批次归一化层，也可以通过对齐特征分布来减少模态差异。</font>**



**[45] W. Li, K. Qi, W. Chen, Y. Zhou, Bridging the distribution gap of visible-infrared person re-identification with modality batch normalization, in: Proceedings of the IEEE International Conference on Artificial Intelligence and Computer Applications (ICAICA), 2021, pp. 23–28.**

Li等人[45]提出了一种模态批量归一化（MBN）层，该层对每个模态子小批量进行归一化，以减少批量归一化带来的分布差距。



**[8] C. Fu, Y. Hu, X. Wu, H. Shi, T. Mei, R. He, CM-NAS: Rethinking crossmodality neural architectures for visible-infrared person re-identification, in: Proceedings of the IEEE International Conference on Computer Vision, 2021, pp. 11823–11832.**

不同的是，Fu等人[8]通过搜索提高了跨模态匹配的性能



#### Optimizing classifier

46 47 49 50<font color='red'>**总结：分类器强制VI ReID模型提取那些ID区分模态共享特征。考虑到这一点，一些工作试图优化分类器层，以更好地提取那些ID辨别模态共享特征**</font>



**[47] M. Ye, X. Lan, Q. Leng, Modality-aware collaborative learning for visible thermal person re-identification, in: Proceedings of the ACM International Conference on Multimedia, 2019, pp. 347–355.**

Ye等人[47]提出，使用单一模态共享分类器学习跨模态特征表示可能会丢失不同模态中的辨别信息。因此，他们提出了一种模态感知协作（MAC）学习方法，通过利用不同分类器之间的关系来正则化模态共享和模态特定身份分类器。



**[50] M. Ye, X. Lan, Q. Leng, J. Shen, Cross-modality person re-identification via modality-aware collaborative ensemble learning, IEEE Trans. Image Process. 29 (2020) 9387–9399.**

后来，在他们的扩展工作[50]中提出了一种协作集成学习策略，通过促进不同分类器之间的知识转移来提高性能。



**[49] A. Wu, Z. Wei-Shi, G. Shaogang, L. Jianhuang, RGB-IR person re-identification by cross-modality similarity preservation, Int. J. Comput. Vis. 128 (2020) 1765–1785.**

类似地，[49]中引入了两个特定于模态的最近邻分类器，通过保持相同模态和跨模态相似性，有助于跨模态匹配共享知识的学习。



**[46] N. Tekeli, A.B. Can, Distance based training for cross-modality person reidentification, in: Proceedings of the IEEE/CVF International Conference on Computer Vision Workshop, 2019, pp. 4540–4549.**

Tekeli等人[46]提出了一种新的基于距离的分数层，该层先于损失函数，并根据距离度量为较近的特征向量提供了较高的分数，而为较远的特征向量给出了较低的分数。



### Feature Disentanglement

<font color='red'>**总结：一些工作建议使用具有一些定制损失函数的网络结构来解开单一模态特征**</font>

**[52] K. Kansal, A.V. Subramanyam, Z. Wang, S. Satoh, SDL: Spectrum-disentangled representation learning for visible-infrared person re-identification, IEEE Trans.**

Kansal等人[52]提出了一种频谱解纠缠表示学习模型，将单一模态特征分解为频谱解纠缠（即模态特定）特征和身份相关（即模态共享）特征。该模型包含一个具有识别损失的频谱消除分支以保持有用的身份相关特征，以及一个具有身份消除损失的频谱提取分支以学习频谱相关信息



<font color='red'>**总结：此外，他们还提出了解纠缠损失，以便于解纠缠的光谱和光谱特征**</font>

**[54] Z. Feng, J. Lai, X. Xie, Learning modality-specific representations for visibleinfrared person re-identification, IEEE Trans. Image Process. 29 (2019) 579–590.**

类似地，[54]还设计了一个基于双分支结构的模型，该模型具有一些专用损失函数，以将单一模态特征分解为模态特定特征和模态共享特征。与[52]不同，他们为每个模态提出了模态特定的判别度量，以增强相应模态特定特征的可分辨性。同时，采用跨模态欧几里德约束和身份丢失来监督模态共享分支，以获得VI ReID的区分性和与人相关的模态共享特征。





<font color='red'>**总结：不同的是，Choi等人[53]以重建和对抗的方式实现了特征分离。**</font>

**[53] S. Choi, S. Lee, Y. Kim, T. Kim, C. Kim, Hi-CMD: Hierarchical cross-modality disentanglement for visible-infrared person re-identification, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2020, pp.**

具体而言，他们首先提出了一种分层交叉模态分解（Hi-CMD）模型，以从输入的可见光和红外图像中自动分解ID辨别因素和ID排除因素。然后，他们只使用了VI ReID的ID判别因子。为此，在Hi-CMD中设计了一个保留身份的个人图像生成（ID-PIG）网络。特别是，它首先在编码器阶段提取ID排除特征和ID辨别特征。然后，通过分别通过重建损失和对抗性损失交换具有相同ID的两个图像的ID排除因子，在解码器阶段实现图像重建和跨模态图像重建。



**[55] W. Hu, B. Liu, H. Zeng, Y. Hou, H. Hu, Adversarial decoupling and modalityinvariant representation learning for visible-infrared person re-identification, IEEE Trans. Circuits Syst. Video Technol. 32 (8) (2022) 5095–5109.**

类似地，Hu等人[55]还提出了一种用于VI ReID的新型对抗性解耦和模态不变表示学习（DMiR）。

该模型通过最小-最大对抗性解缠过程使用身份网和域网，分别将输入特征解缠为身份相关特征和域相关特征。



**[56] N. Pu, W. Chen, Y. Liu, E.M. Bakker, M.S. Lew, Dual Gaussian-based variational subspace disentanglement for visible-infrared person re-identification, in: Proceedings of the ACM International Conference on Multimedia, 2020, pp.2149–2158.**

基于变分自动编码器（VAE），Pu等人[56]提出了一种新的基于双高斯的变分自动编码（DG-VAE），它将跨模态特征分解为身份可分辨信息（IDI）和身份模糊信息（IAI）。具体地说，DG-VAE通过强制IDI码遵循高斯（MoG）分布的混合来确保类间可分离性，其中每个分量对应于特定身份。同时，IAI代码需要遵循标准高斯分布。此外，提出了一种三重交换重构（TSR）策略，通过将IDI和IAI压缩到单独的分支中来促进解纠缠过程。



**[9] X. Tian, Z. Zhang, S. Lin, Y. Qu, Y. Xie, L. Ma, Farewell to mutual information variational distiilation for cross-modal person re-identification, in: Computer Vision and Pattern Recognition, 2021, pp. 1522–1531.**

<font color='red'>**总结：最近，Tian等人[9]提出了一种用于VI ReID的互信息拟合策略，称为变分自蒸馏（VSD）**</font>，该策略保留了足够的任务相关信息，同时通过使用变分推理方法重建信息瓶颈的目标，去除了那些与任务无关的细节（即视点）（见表4）。



### Metric learning

#### Identity loss

<font color='red'>**总结：身份丢失可以促进VI ReID网络中与人相关的信息提取，这通常通过使用原始交叉熵损失进行训练来实现。不同的是，一些工作试图优化身份丢失，以提取更具歧视性的与人相关的特征**</font>

[59]Y. Hao, N. Wang, L. Jie, X. Gao, HSME: Hypersphere manifold embedding for visible thermal person re-identification, in: Proceedings of the AAAI Conference on Artificial Intelligence, 2019, pp. 8385–8392.

Hao等人[59]使用Sphere Softmax函数来学习超球面流形嵌入，并约束该超球面上的模态内变化和跨模态变化，从而可以用清晰的边界对不同身份的图像进行分类。



#### Contrastive loss and its variations

<font color='red'>**总结：To reduce cross-modality variants, some works introduce contrastive loss into the VI-ReID**</font>



**[57]M. Ye, X. Lan, J. Li, P.C. Yuen, Hierarchical discriminative learning for visible thermal person re-identification, in: Proceedings of the AAAI Conference on Artificial Intelligence, 2018, pp. 7501–7508.**

Ye等人[57]提出了一种分层跨模态度量学习策略，以更好地利用VI ReID中的对比损失，这首先将两种不同模态转换为一致的空间。之后，随后学习模态共享特征，同时在一致空间中压缩来自同一个人的特征



**[79]Z. Sun, Y. Zhu, S. Song, J. Hou, S. Du, Y. Song, The multi-layer constrained loss for cross-modality person re-identification, in: Proceedings of the International Conference on Artificial Intelligence and Signal Processing, 2020, pp. 1–6.**

[79]提出了一种多层约束（MLC）损失，其本质上应用了跨模态对比损失来约束多层特征。



#### Triplet loss and its variations

<font color='red'>**总结：Ye等人[58]首先将VV-ReID任务中的三重态损失转移到VI ReID任务中**</font>



**[58] M. Ye, Z. Wang, X. Lan, P.C. Yuen, Visible thermal person re-identification via dual-constrained top-ranking, in: Proceedings of the International Joint Conference on Artificial Intelligence, 2018, pp. 1092–1099. [59] Y. Hao, N. Wang, L. Jie, X. Gao, HSME: Hypersphere manifold embedding for visible thermal person re-identification, in: Proceedings of the AAAI Conference on Artificial Intelligence, 2019, pp. 8385–8392.**

具体而言，他们提出了双向双约束顶级（BDTR）损失，其中包含两个三元组约束，即跨模态顶级约束和模态内顶级约束，以分别解决跨模态变化和模态内变化。



**[82] H. Liu, S. Ma, D. Xia, S. Li, SFANet: A spectrum-aware feature augmentation network for visible-infrared person reidentification, IEEE Trans. Neural Netw.**

基于BDTR损失，Liu等人[82]提出了一种双向三约束顶推排名损失（BTTR），它引入了额外的三元组损失，即模态间顶推排名丢失，以进一步促进学习区分性特征嵌入。



**[65] J. Wang, S. Jiao, Y. Li, Z. Miao, Two-stage metric learning for cross-modality person re-identification, in: Proceedings of the International Conference on Multimedia and Image Processing, 2020, pp. 28–32.**

根据BTTR损失的相同想法，Wang等人[65]还提出了混合模态三重态损失。



**[66] Y.-B. Zhao, J.-W. Lin, Q. Xuan, X. Xi, HPILN: a feature learning framework for cross-modality person re-identification, IET Image Process. 13 (14) (2019) 2897–2904.**

Zhao等人[66]提出了一种硬五倍体丢失，其目的是从最硬的五倍体对中选择最硬的全局三倍体和最硬的跨模态三倍体，以同时处理跨模态和跨模态变化。



**[80] L. Zhang, H. Guo, K. Zhu, H. Qiao, G. Huang, S. Zhang, H. Zhang, J.Sun, J. Wang, Hybrid modality metric learning for visible-infrared person re-identification, ACM Trans. Multimed. Comput. Commun. Appl. 18 (2022) 1–15.**

Zhang等人[80]提出了一种综合的混合模态度量学习框架，该框架还涉及四种基于配对的相似性约束，以解决所有模态内和跨模态变化。



<font color='red'>**总结：不同的是，一些工作试图优化这些三重态损耗变体中的距离函数**</font>

**[67] G. Gao, H. Shao, Y. Yu, F. Wu, M. Yang, Leaning compact and representative features for cross-modality person re-identification, World Wide Web 25 (2022) 1649–1666.**

**[64] H. Ye, H. Liu, F. Meng, X. Li, Bi-directional exponential angular triplet loss for RGB-infrared person re-identification, IEEE Trans. Image Process. 30 (2021) 1583–1595.**

例如，等人[67]和Ye等人[64]提出用余弦距离代替BDTR损失的欧几里德距离进行训练。



**[78] X. Hu, Y. Zhou, Cross-modality person ReID with maximum intra-class triplet loss, in: Proceedings of the Chinese Conference on Pattern Recognition and Computer Vision, 2020, pp. 557–568.**

Hu等人[78]将三重态损失与软边缘损失相结合，从而获得了VI ReID的新的最大类内三重态（MICT）损失。



**[69] P. Wang, F. Su, Z. Zhao, Y. Zhao, L. Yang, Y. Li, Deep hard modality alignment for visible thermal person re-identification, Pattern Recognit. Lett. 133 (2020) 195–201.**

此外，Wang等人[69]提出了一种硬模态对齐（HMA）损失，该损失首先挖掘具有大模态差异的硬特征子空间，然后在该硬特征子中执行一些三元组约束，以使模态分布更易于区分。



**[85] J. Liu, Y. Sun, F. Zhu, H. Pei, Y. Yang, W. Li, Learning memory-augmented unidirectional metrics for cross-modality person re-identification, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2022, pp.19366–19375.**

**[86] Y. Sun, C. Cheng, Y. Zhang, C. Zhang, L. Zheng, Z. Wang, Y. Wei, Circle loss: A unified perspective of pair similarity optimization, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp.6397–6406.**

基于圆损失[86]，Liu[85]提出了一种用于VI ReID的新的记忆增强单向度量学习方法。它学习了两个单向的显式跨模态度量，并通过基于记忆的增强进一步增强了它们。



#### Center loss and its variations

**<font color='red'>总结：中心损失旨在探索不同特征中心之间的距离关系，而不是直接约束特征距离</font>**

**[75] Y. Zhu, Z. Yang, L.-C. Wang, S. Zhao, X. Hu, D. Tao, Hetero-center loss for cross-modality person re-identification, Neurocomputing 386 (2020) 97–109.**

具体而言，Zhu等人[75]首先提出了异质中心损失（HC损失），通过约束两种异质模态之间的类内中心距离来学习模态共享特征。



**[77] H. Liu, X. Tan, X. Zhou, Parameter sharing exploration and hetero-center triplet loss for visible-thermal person re-identification, IEEE Trans. Multimed.**

Liu等人[77]进一步提出，通过提出新的PSE损失，将异质中心损失与三重态损失结合起来。通过用中心距离替换相应的特征距离，这种损失放松了传统三元组损失的严格限制。



**[83] W. Li, K. Qi, W. Chen, Y. Zhou, Unified batch all triplet loss for visibleinfrared person re-identification, in: Proceedings of the International Joint Conference on Neural Networks (IJCNN), 2021, pp. 1–8, http://dx.doi.org/10.**

Li等人[83]将PSE损失改写为批处理所有形式，即使用批处理中的所有样本，以提高模型性能。



**[71] Y. Ling, Z. Zhong, Z. Luo, P. Rota, S. Li, N. Sebe, Class-aware modality mix and center-guided metric learning for visible-thermal person re-identification, in: Proceedings of the ACM International Conference on Multimedia, 2020, pp.889–897.**

Ling等人[71]提出了一种中心引导的度量学习（CML）约束，该约束在全局上减少了同一类的模态间中心之间的距离，并鼓励不同类的中心远离以处理大的跨模态差异。同样，它将每个样本拉到其对应模态的类中心附近，以克服模态内的变化。



**[61] M. Ye, X. Lan, Z. Wang, P.C. Yuen, Bi-directional center-constrained top-ranking for visible thermal person re-identification, IEEE Trans. Inf. Forensics Secur. 15 (2020) 407–419.**

Ye等人[61]提出了一种双向中心约束顶级（eBDTR）损失，该损失将前两个约束（即，跨模态顶级约束和跨模态顶级限制）合并到单个公式中，以处理跨模态和内部模态变化。



**[76] J. Sun, Y. Li, H. Chen, Y. Peng, X. Zhu, J. Zhu, Visible-infrared cross-modality person re-identification based on whole-individual training, Neurocomputing 440 (2021) 1–11.**

Sun等人[76]提出了一种新的整体-个体训练（WIT）模型，该模型包含整体部分和个体部分。在该模型中，为整个部分开发了两个损失函数，即中心最大均值差异（CMMD）损失，以拉入两种模态的中心，以及类内异质中心（ICHC）损失，将具有相同身份的图像拉入其跨模态中心。同时，对于单个部分，使用交叉模态三元组（CMT）损失来区分具有不同身份的行人图像。



#### Adversarial loss

<font color='red'>**总结：一些作品[60，63，68，81]也使用对抗性学习来学习模态不变特征**</font>

**[60] P. Dai, R. Ji, H. Wang, Q. Wu, Y. Huang, Cross-modality person re-identification with generative adversarial training, in: Proceedings of the International Joint Conference on Artificial Intelligence, 2018, pp. 677–683.**

**[63] Y. Hao, J. Li, N. Wang, X. Gao, Modality adversarial neural network for visible-thermal person re-identification, Pattern Recognit. 107 (2020) 107533.**

例如，[60，63]设计了一个特征提取器作为生成器，一个模态分类器作为鉴别器，以通过生成器和鉴别器的最小最大博弈来学习有区别的公共表示。



**[81] X. Hao, S. Zhao, M. Ye, J. Shen, Cross-modality person re-identification via modality confusion and center aggregation, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 16403–16412.**

Hao等人[81]提出了一种模态混淆学习网络（MCLNet），通过混淆特征学习过程中的模态辨别来学习模态不变特征。它通过最小-最大博弈最小化了联运差异，并最大化了不同实例之间的跨模态相似性。通过这样做，他们在模态混淆和一般的跨模态特征学习之间取得了平衡。



**[68] P. Zhang, Q. Wu, X. Yao, J. Xu, Beyond modality alignment: Learning part-level representation for visible-infrared person re-identification, Image Vis. Comput.108 (2021) 104118.**

Zhang等人[68]提出了一种双对齐部分感知表示（DAPR）框架，以同时缓解模态偏差并挖掘不同级别的判别表示。它通过反向传播来自具有对抗策略的模态分类器的反向梯度来分层地减少高级特征的模态差异，从而学习模态不变的特征空间。



## Modality-specific information compensation

**<font color='red'>总结：已经提出了许多基于模态特定信息补偿的模型，其遵循这样的思想，即首先从现有模态特定信息中生成缺失的模态特定信息以解决跨模态变化，然后从生成的原始信息中提取有区别的人特征以处理模态内变化。根据它们生成缺失模态特定信息的方式，我们将这些模型进一步划分为两个子类，即单模态信息补偿和跨模态信息补偿。</font>**

Single-modality information compensation：<font color='red'>**总结：基于单模态信息补偿的模型通常生成一个缺失的模态特定信息，而不是所有信息。**</font>

Cross-modality information compensation：<font color='red'>**总结：基于跨模态信息补偿的模型同时生成VI ReID的所有缺失模态特定信息。**</font>

![image-20221125111617879](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221125111617879.png)

### Single-modality information compensation

**<font color='red'>总结：一些作品从真实的可见光图像中生成假红外图像进行补偿</font>**

**[88] G. Wang, T. Zhang, J. Cheng, S. Liu, Y. Yang, Z. Hou, RGB-Infrared crossmodality person re-identification via joint pixel and feature alignment, in: Proceedings of the IEEE International Conference on Computer Vision, 2019, pp. 3622–3631.**

例如，Wang等人[88]提出了第一个基于单模态信息补偿的工作，即对齐生成对抗网络（AlignGAN），该网络采用了两种对齐策略，包括像素级对齐和特征级对齐，用于VI ReID。具体而言，AlignGAN首先通过从真实可见图像生成假红外图像来实现像素级对准，然后通过特征对准模块匹配生成的假红外图像和真实红外图像。



**[89] Z. Zhang, S. Jiang, C. Huang, Y. Li, R.Y. Da Xu, RGB-IR cross-modality person ReID based on teacher-student GAN model, Pattern Recognit. Lett. (2021) 155–161.**

之后，Zhang等人[89]提出了一种师生GAN模型（TS-GAN），该模型从现有可见图像中生成假IR图像，以减少跨模态变化，并指导有区别的人特征的提取。



<font color='red'>**总结：一些作品采用了从真实红外图像生成假可见图像的方式进行补偿**</font>

**[92] H. Dai, Q. Xie, Y. Ma, Y. Liu, S. Xiong, RGB-infrared person re-identification via image modality conversion, in: Proceedings of the International Conference on Pattern Recognition, 2021, pp. 592–598.**

例如，Dai等人[92]设计了一个名为CE2L的新模型，该模型首先通过图像模态转换模块将红外图像转换为可见图像，然后通过使用特征提取模块和VI ReID的特征学习模块学习其辨别特征。



**[87] X. Zhong, T. Lu, W. Huang, J. Yuan, W. Liu, C.-W. Lin, Visible-infrared person re-identification via colorization-based siamese generative adversarial network, in: Proceedings of the International Conference on Multimedia Retrieval, 2020, pp. 421–427.**

**[90] X. Zhong, T. Lu, W. Huang, M. Ye, X. Jia, C.-W. Lin, Grayscale enhancement colorization network for visible-infrared person re-identification, IEEE Trans.**

Zhong等人[87，90]建议使用彩色化方法为灰度图像着色，而不是使用跨模态转换模型。因此，他们提出了一种灰度增强彩色化网络（GECNet），该网络首先将从原始红外图像中提取的特征与其彩色图像中的特征进行融合，然后将融合的特征与从可视图像提取的特征进行匹配，以用于VI ReID。



### Cross-modality information compensation

<font color='red'>**总结：大多数现有的基于跨模态信息补偿的模型通过从现有模态生成缺失模态的图像来补偿缺失的模态特定信息，即图像级补偿**。</font>

**[91] Z. Wang, Z. Wang, Y. Zheng, Y.-Y. Chuang, S. Satoh, Learning to reduce duallevel discrepancy for infrared-visible person re-identification, in: Proceedings of TheIEEE Conference on Computer Vision and Pattern Recognition, 2019, pp.**

例如，Wang等人[91]提出了一种双水平差异减少学习（D2RL）策略。该策略首先通过设计图像级子网络将红外图像转换为可见图像，将可见图像转换为红外图像，从而减少了模态差异。然后，提出了一个特征级子网络，通过引入一些特征级约束来减少剩余的外观差异。



**[94] X. Fan, W. Jiang, H. Luo, W. Mao, Modality-transfer generative adversarial network and dual-level unified latent representation for visible thermal person re-identification, Vis. Comput. (2020) 1–16.**

类似地，Fan等人[94]设计了一种模态转移生成对抗网络（mtGAN），以从目标模态中的源图像生成跨模态对应物，从而获得同一个人的成对图像。



**[93] G.-A. Wang, T. Zhang, Y. Yang, J. Cheng, J. Chang, X. Liang, Z. Hou, Crossmodality paired-images generation for RGB-infrared person re-identification, in: Proceedings of the AAAI Conference on Artificial Intelligence, 2020, pp.**

Wang等人[93]提出生成跨模态配对图像，并执行全局集合级和细粒度实例级对齐。



**[95] Y. Yang, T. Zhang, J. Cheng, Z. Hou, P. Tiwari, H.M. Pandey, et al, Crossmodality paired-images generation and augmentation for RGB-infrared person re-identification, Neural Netw. 128 (2020) 294–304.**

之后，Yang等人[95]提出了[93]的扩展版本。具体来说，在双层策略的基础上，他们进一步引入了一个潜在的流形空间，以随机抽样和生成一些不可见类的图像，从而在测试时实现更好的泛化。



**[96] B. Hu, J. Liu, Z.-j. Zha, Adversarial disentanglement and correlation network for rgb-infrared person re-identification, in: Proceedings of the IEEE International Conference on Multimedia and Expo, 2021, pp. 1–6.**

Hu等人[96]提出了一种新的对抗性解纠缠和相关网络（ADCNet），该网络进一步投资于在跨模态图像翻译处理中学习人的模态不变和辨别表示，从而获得更好的结果。



**[97] D. Xia, H. Liu, L. Xu, L. Wang, Visible-infrared person re-identification with data augmentation via cycle-consistent adversarial network, Neurocomputing 443 (2021) 35–46.**

Xia等人[97]提出了一种图像模态转换（IMT）网络，该网络学习从源模态的图像生成目标模态的图像。具体来说，他们通过CycleGAN进行了跨模态图像转换。这些生成的图像用作数据增强工具，以扩大训练数据集的大小并增加其多样性。



**[101] J. Liu, J. Wang, N. Huang, Q. Zhang, J. Han, Revisiting modality-specific feature compensation for visible-infrared person re-identification, IEEE Trans. Circuits Syst. Video Technol. 32 (10) (2022) 7226–7240.**

Liu等人[101]首先通过重新访问现有的基于模态特定特征补偿的模型，揭示了性能不足的原因，然后相应地提出了一种新的基于两阶段GAN的模型，实现了新的最先进性能。代替图像级补偿，



**[100] Y. Lu, Y. Wu, B. Liu, T. Zhang, B. Li, Q. Chu, N. Yu, Cross-modality person re-identification with shared-specific feature transfer, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2020, pp.**

Lu等人[100]根据不同模态样本的共享特征对其相似性进行建模，然后在模态之间/跨模态传输共享和特定特征，从而实现特征级补偿。



**<font color='red'>总结：不同的是，Zhang[102]提出了一种新的特征级模态补偿网络（FMCNet）</font>**

**[102] Q. Zhang, C. Lai, J. Liu, N. Huang, J. Han, FMCNet: feature-level modality compensation for visible-infrared person re-identification, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2022, pp.**

它不是生成缺失模态的图像，而是直接从另一模态的现有模态共享特征中生成一个模态的那些缺失模态特定特征。为此，首先设计了单模态特征分解模块，将单模态特征分为模态特定特征和模态共享特征。然后，提出了一个特征级模态补偿模块，以从现有模态共享特征中生成那些缺失的模态特定特征。



## Auxiliary information

<font color='red'>**总结：大多数现有的基于辅助信息的模型，这些模型主要探讨了使用人面具、人姿势或辅助模态来提取那些有区别的人特征的方法**</font>

### Person mask

<font color='red'>**总结：为了增强输入图像的轮廓信息并减少背景杂波的影响**</font>

**[103] M. Qi, S. Wang, G. Huang, J. Jiang, J. Wu, C. Chen, Mask-guided dual attentionaware network for visible-infrared person re-identification, Multimedia Tools Appl. 80 (12) (2021) 17645–17666. [104] Y. Huang, Q. Wu, J. Xu, Y. Zhong, P. Zhang, Z. Zhang, Alleviating modality bias training for infrared-visible person re-identification, IEEE Trans. Multimed.**

Qi等人[103]通过引入前景人物作为强调信息来丰富这些表示，提出了一种新的掩码引导双网络（MDAN）。



**[108] Z. Zhao, B. Liu, Q. Chu, Y. Lu, N. Yu, Joint color-irrelevant consistency learning and identity-aware modality adaptation for visible-infrared cross modality person re-identification, in: Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 35, 2021, pp. 3520–3528.**

基于人类解析掩码，Zhao等人[108]提出了一种新的VI ReID颜色无关一致性学习方法，该方法通过对随机服装颜色变换施加颜色无关一致约束来学习具有高辨别性的颜色无关特征。





### Person posture

**[111] Y. Miao, N. Huang, X. Ma, Q. Zhang, J. Han, On exploring pose estimation as an auxiliary learning task for visible-infrared person re-identification, 2021, arXiv preprint arXiv:2201.03859.**

**<font color='red'>总结：Miao等人[111]利用姿势估计作为辅助学习任务，在端到端框架中辅助VI ReID任务。通过以互惠的方式联合训练这两个任务，他们的模型学习了更高质量的模态共享和ID相关特征。</font>**



**[112] Z. Cao, G. Hidalgo, T. Simon, S.-E. Wei, Y. Sheikh, OpenPose: Realtime multiperson 2D pose estimation using part affinity fields, IEEE Trans. Pattern Anal.Mach. Intell. 43 (2021) 172–186.**

Ye等人[36]提出了一种结构感知位置变换器，以捕捉那些具有VI ReID行人结构信息的模态共享特征，该变换器采用了预训练的OpenPose[112]来获得关键点热图，用于建模行人的结构信息。



### Auxiliary modality

<font color='red'>**总结：与直接提供人物相关信息的上述两种辅助信息不同，基于辅助模态的模型试图引入一种额外的模态来弥合RGB和IR图像之间的差距，以减少模态差异。**</font>

**[107] M. Ye, J. Shen, L. Shao, Visible-infrared person re-identification via homogeneous augmented tri-modal learning, IEEE Trans. Inf. Forensics Secur. 16 (2020) 728–739.**

Ye等人[107]提出了一种用于VI ReID的均匀增强三模态（HAT）学习方法，其中从其均匀可见图像生成辅助灰度模态。它保留了可见光图像的结构信息，并近似了红外图像的图像样式。为了减少可见光和红外模态之间的差距，Li等人[105]使用了辅助X模态作为辅助，并将红外-可见光双模交叉模态学习重新表述为X红外-可见三模态学习问题。



**[106] H. Liu, Z. Miao, B. Yang, R. Ding, A base-derivative framework for cross-modality RGB-infrared person re-identification, in: Proceedings of the International Conference on Pattern Recognition, 2021, pp. 7640–7646.**

为了进一步减少模态差异，Liu等人[106]引入了两种辅助模态，并进一步将双模态跨模态学习问题重新表述为四模态学习问题。通过加倍的输入图像，学习到的人物特征变得更有辨别力。



**[104] Y. Huang, Q. Wu, J. Xu, Y. Zhong, P. Zhang, Z. Zhang, Alleviating modality bias training for infrared-visible person re-identification, IEEE Trans. Multimed.24 (2021) 1570–1582.**

为了解决模态偏差训练（MBT）问题，Huang等人[104]引入了包含可见光和红外信息的第三模态数据，以进一步防止红外模态在训练过程中被淹没。他们制定了一个新的基于多任务学习的VI ReID模型，以利用人掩模预测和VI ReID之间的内在关系，而不是简单地使用人掩模进行特征选择，从而实现了两个任务中人身信息的交互。在这样做的过程中，该模型通过人面具预测和VI ReID同时捕获discriminative modality-invariant person body information。



## Data augmentation

<font color='red'>**总结：大多数现有的基于数据扩充的模型。这些模型专注于开发一些新的数据增强策略，以优化VI ReID模型的训练过程，从而获得有区别的人物特征**</font>

**[115] X. Fan, H. Luo, C. Zhang, W. Jiang, Cross-spectrum dual-subspace pairing for RGB-infrared cross-modality person re-identification, 2020, arXiv preprint arXiv:2003.00213.**

例如，[115]中提出了一种新的多光谱图像生成方法，通过随机混洗可见图像的不同通道（即，蓝色、绿色、红色和灰色光谱）来生成更多训练样本，以跨模态重新识别同一个人。



**[116] J. Liu, W. Song, C. Chen, F. Liu, Cross-modality person re-identification via channel-based partition network, Appl. Intell. (2021) 1–13.**

为了解决样本不足的问题，Liu等人[116]构建了一个生成器，该生成器将一些可见图像作为输入，并通过随机组合这些图像的通道来输出一些新图像，从而获得更多的训练样本。



**[120] M. Ye, W. Ruan, B. Du, M.Z. Shou, Channel augmented joint learning for visibleinfrared recognition, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 13567–13576.**

Ye等人[120]提出了一种新的方法，通过随机交换可见图像的通道来均匀地生成彩色相关图像。



**[113] J.K. Kang, T.M. Hoang, K.R. Park, Person re-identification between visible and thermal camera images based on deep residual CNN using single input, IEEE Access 7 (2019) 57972–57984.**

Kang等人[113]提出了一种新的策略，从输入的可见光和红外图像中生成三种类型的辅助图像用于训练。



**[114] J.K. Kang, M.B. Lee, H.S. Yoon, K.R. Park, AS-RIG: Adaptive selection of reconstructed input by generator or interpolation for person re-identification in cross-modality visible and thermal images, IEEE Access 9 (2021) 12055–12066.**

Kang等人[114]进一步提出了一种通过使用生成器或插值算法来自适应选择重建输入的新策略，以提高人ReID的准确性。具体而言，他们的模型可以自适应地选择生成器或插值算法来重建输入数据以进行数据扩充。



**[119] M. Yang, Z. Huang, P. Hu, T. Li, J. Lv, X. Peng, Learning with twin noisy labels for visible-infrared person re-identification, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2022, pp.14308–14317.**

不同的是，Yang等人[119]没有处理这些输入图像，而是研究了VI ReID标签级别中的一个新问题，称为双噪声标签，这可以被视为噪声标签的新范式。此外，他们还设计了一种新的方法，即双重鲁棒训练，用于学习带有噪声的注释和对应关系。
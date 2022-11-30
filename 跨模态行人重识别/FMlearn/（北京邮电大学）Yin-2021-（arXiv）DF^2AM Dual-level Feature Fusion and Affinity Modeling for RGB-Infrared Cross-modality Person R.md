# （北京邮电大学）Yin-2021-（arXiv）DF^2AM: Dual-level Feature Fusion and Affinity Modeling for RGB-Infrared Cross-modality Person Re-identification

![image-20221106105213097](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221106105213097.png)

------

## 问题、研究目标、研究内容

### 问题

由于类内差异和跨模态差异，RGB红外人物重新识别是一项具有挑战性的任务。现有的工作主要集中于通过在模态之间对齐图像样式或特征分布来学习模态共享的全局表示，而来自身体部分的局部特征和人物图像之间的关系在很大程度上被忽略。



### 研究目标

1.从局部到全局的方式学习区分特征的注意力（ learning attention for discriminative feature from local to global manner ）

2.为了进一步从人物图像中挖掘全局特征之间的关系（to further mining the relationships between global features from person images）

### 研究内容

![image-20221129170102641](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20221129170102641.png)

<font color='red'>**1.局部到全局学习对辨别特征的注意（ learning attention for discriminative feature from local to global manner ）：**</font>

它背后的关键思想在于一个人的不同部分包含不同的歧视性信息。网络模型仍然可以使用注意力机制从上半身捕捉有用的信息，而不管行人的下半身被什么东西遮挡（例如自行车）。具体而言，我们提出了局部关注：局部特征的关注是局部确定的，即，对其自身应用学习的变换函数，其中细化的部分聚集特征考虑不同身体部位之间的重要性。然而，这样的局部策略不能从全局角度充分利用特征信息。

我们的解决方案是使用全局平均池（GAP）从特征地图中获取全局特征信息，这被称为全局关注。通过这种方式，我们考虑全局特征及其局部信息，以从全局和局部角度确定人的不同身体部位之间的重要性。这也与人类在寻找辨别线索方面的认知一致：进行比较，然后确定重要性。

<font color='red'>**2.上述方法独立处理每个样本，忽略人物图像之间的关系。因此，提出了一种新颖而有效的相似性推断，以获得最佳的模态内和模态间图像匹配：**</font>

它利用样本相似性中的类内紧性和类间可分离性作为监督信息，来建模类内和互质样本之间的相关性。特别是，每个样本都包含一些结构信息，并使用成对关系将信息传播给其邻居，如图1所示。这种邻居推理方案可以弥补同一个人的不同图像中存在的特定信息的不足，并进一步增强从对象级别学习的特征的鲁棒性。

## 结果与讨论



## 文章好在哪里

### 主要贡献

1. 我们建议通过对特征的局部和全局观点来学习对歧视性表征的关注（learn the attention for discriminative representation by taking both local and global views of features.）。
2. 我们设计了一种有效的相互邻居推理，通过建模模态内和模态间图像之间的相关性来捕获对象的长期依赖性（We design an efficient mutual neighbor reasoning to capture long-range dependency of objects, by modeling affinity between intra- and inter-modality images.）。
3. 我们提出的方法在两个最受欢迎的基准数据集上实现了与现有技术相比的显著性能改进。



### 相关工作



## 自我思考

看不懂
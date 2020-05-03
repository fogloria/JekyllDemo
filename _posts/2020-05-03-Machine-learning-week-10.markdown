---
layout: post
title:  "ML week #10 SVM"
date:   2020-05-03 21:26:24 +0800
categories: 
---
## SVM

Support Vector Machine 支持向量机

* SVM算法原理及数学推导

- SVM算法中的核函数
- SVM算法解决分类问题及回归问题

决策边界的选择，不仅要考虑已经存在的数据上的是否分类正确，还要考虑是否能够更好地划分未出现的测试数据。

支撑向量机如何解决“不适定问题呢”？SVM要找到一条泛化性比较好的决策边界，就是这条直线要离两个分类都尽可能的远，我们认为这样的决策边界就是好的。

离决策边界最近的点到决策边界的距离尽可能地远。那也就是说，我们可以忽略其他大部分的数据点，只关注这几个特殊的点即可。

SVM算法寻找的是使得Margin区间最大的中间的决策边界（超平面），而衡量Margin使用的是数据点之间的距离。**涉及距离的**，就应当对量纲问题进行关注，即进行**标准化处理**。

***C越小，Margin越大***

Soft Margin可以看作是给Hard Margin加上一个正则化项，提高其容错性。



## 核函数

将数据从原始空间映射到Feature Space中去，就可以解决原本的线性不可分问题。



## 优缺点：

SVM算法的主要优点有：

1. 解决高维特征的分类问题和回归问题很有效,在特征维度大于样本数时依然有很好的效果。
2. 仅仅使用一部分支持向量来做超平面的决策，无需依赖全部数据。
3. 有大量的核函数可以使用，从而可以很灵活的来解决各种非线性的分类回归问题。
4. 样本量不是海量数据的时候，分类准确率高，泛化能力强。

SVM算法的主要缺点有：

1. 如果特征维度远远大于样本数，则SVM表现一般。
2. SVM在样本量非常大，核函数映射维度非常高时，计算量过大，不太适合使用。
3. 非线性问题的核函数的选择没有通用标准，难以选择一个合适的核函数。
4. SVM对缺失数据敏感。





Ref:

[入门支持向量机1：图文详解SVM原理与模型数学推导](http://mp.weixin.qq.com/s?__biz=MzI4MjkzNTUxMw==&mid=2247484477&idx=1&sn=226e099c1951b6c11b1e7fb6b7a092a3&chksm=eb932d8bdce4a49d0595b6c642fc2e5969fdc05a185f97a39cc1a896e24d56d8703541a28f9c&scene=21#wechat_redirect)

[入门支持向量机2:软间隔与sklearn中的SVM](http://mp.weixin.qq.com/s?__biz=MzI4MjkzNTUxMw==&mid=2247484512&idx=1&sn=7a6b75f312e92bbdecafdedf979ed929&chksm=eb932dd6dce4a4c0ae4ea087878ec7a5f5ccc0724a85aa93daff3d08c33ecf86a3d809e51a82&scene=21#wechat_redirect)

[入门支持向量机3：巧妙的Kernel Trick](http://mp.weixin.qq.com/s?__biz=MzI4MjkzNTUxMw==&mid=2247484546&idx=1&sn=33c6c5cb698b8835b2ee57dd8ea7c221&chksm=eb932d34dce4a4221f40f3daa26863a5fd05dcbcf74738d5423316c643e3ff930904d1a33fca&scene=21#wechat_redirect)

[入门支持向量机4：多项式核函数与RBF核函数代码实现及调参](http://mp.weixin.qq.com/s?__biz=MzI4MjkzNTUxMw==&mid=2247484572&idx=1&sn=fd6e86ce45167286fb6ba4089b7b29dd&chksm=eb932d2adce4a43c44d26e79d4968f395d7cc22a31d84aef7944b227e1843b3f0722a5e894ed&scene=21#wechat_redirect)

[入门支持向量机5：回归问题及系列回顾总结](http://mp.weixin.qq.com/s?__biz=MzI4MjkzNTUxMw==&mid=2247484596&idx=1&sn=7e93eb135d66c86238ccf516f0ae65ec&chksm=eb932d02dce4a41447a9cb34d627f435c760a5deb125a40d4c2a77f99e2187194d6bfbda4cbc&scene=21#wechat_redirect)



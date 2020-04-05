---
layout: post
title:  "ML week #6 Pipeline, bias&variance, L1&L2"
date:   2020-04-05 22:43:00 +0800
categories: 
---
## sklearn中的Pipeline

研究一个因变量与多个自变量间多项式的回归分析方法，称为多项式回归*Polynomial Regression*,自变量x和因变量y之间的关系被建模为n次多项式。

创建一个一元二次方程，并增加一些噪音：

```python
import numpy as np
import matplotlib.pyplot as plt

x=np.random.uniform(-3,3,size=100)
X=x.reshape(-1,1)
y=0.5+x**2+x+2+np.random.normal(0,1,size=100)
plt.scatter(x,y)
plt.show()
```

线性回归拟合：

```python
from sklearn.linear_model import LinearRegression

lin_reg=LinearRegression()
lin_reg.fit(X,y)
y_predict=lin_reg.predict(X)
plt.scatter(x,y)
plt.plot(x,y_predict,color='r')
plt.show()
```

通过多项式回归优化（添加一个特征，对X中每个数据进行平方）：

```python
#创建一个新特征
(X**2).shape

#凭借一个新数据数据集
X2=np.hstack([X,X**2])

#用新的数据集进行线性回归训练
lin_reg2=LinearRegression()
lin_reg2.fit(X2,y)
y_predict2=lin_reg2.predict(X2)

plt.scatter(x,y)
plt.plot(np.sort(x),y_predict2[np.argsort(x)],color='r')
plt.show()
```

多项式回归的过程：

- 将原始数据通过`PolynomialFeatures`生成相应的多项式特征
- 多项式数据可能还要进行特征归一化处理
- 将数据送给线性回归

Pipeline就是将这些步骤都放在一起。参数传入一个列表，列表中的每个元素是管道中的一个步骤。

多项式回归在算法并没有什么新的地方，完全是使用线性回归的思路，关键在于为数据添加新的特征，而这些**新的特征是原有的特征的多项式组合**，采用这样的方式就能解决非线性问题。

这样的思路跟PCA这种降维思想刚好相反，而多项式回归则是升维，添加了新的特征之后，使得更好地拟合高维数据。



## 偏差与方差

**偏差bias：**偏差衡量了模型的预测值与实际值之间的偏离关系。

> *问题本身的假设不正确/欠拟合。如：针对非线性问题使用线性回归/采用的特征与问题完全没有关系。*   高偏差如参数学习算法、***线性回归***

**方差variance：**训练数据在不同迭代阶段的训练模型中，预测值的变化波动情况/离散情况。

> *模型没有学习到问题的本质，学习到很多噪音。通常可能是使用模型太复杂/过拟合。*   高方差如非参数学习算法、 **kNN**

![GDrcoF.png](https://s1.ax1x.com/2020/04/05/GDrcoF.png)

↗️ 低偏差、高方差：过拟合，就是模型太贴合训练数据了，导致其泛化/通用能力差，若遇到测试集，则准确度下降的厉害。

**模型误差=偏差+方差+不可避免的误差（噪音）**

![GDso1s.png](https://s1.ax1x.com/2020/04/05/GDso1s.png)

训练数据太小一定是不好的，会过拟合，模型复杂度太高，方差很大，不同数据集训练出来的模型变化非常大。选择合适的模型复杂度，复杂度高的模型通常对训练数据有很好的拟合能力。



在机器学习领域，主要的挑战来自方差。处理高方差的手段有：

- 降低模型复杂度
- 减少数据维度；降噪
- 增加样本数
- 使用验证集



## 模型正则化 L1正则 L2正则

### L1正则

我们说，LASSO回归的全称是：Least Absolute Shrinkage and **Selection Operator** Regression.

这里面有一个特征选择的部分，或者说L1正则化可以使得参数稀疏化，即得到的参数是一个稀疏矩阵。

所谓**稀疏性**，说白了就是**模型的很多参数是0**。通常机器学习中特征数量很多，例如文本处理时，如果将一个词组（term）作为一个特征，那么特征数量会达到上万个（bigram）。在预测或分类时，那么多特征显然难以选择，但是如果代入这些特征得到的模型是一个**稀疏模型，很多参数是0**，表示**只有少数特征对这个模型有贡献，绝大部分特征是没有贡献的，即使去掉对模型也没有什么影响**，此时我们就可以**只关注系数是非零值的特征**。

这相当于**对模型进行了一次特征选择，只留下一些比较重要的特征**，**提高模型的泛化能力，降低过拟合的可能**。



### L2正则

**“岭回归”**



L1正则化就是在损失函数后边所加正则项为L1范数，加上L1范数容易得到稀疏解（0比较多），一般来说L1正则化较常使用。

L2正则化就是损失后边所加正则项为L2范数，加上L2正则相比于L1正则来说，得到的解比较平滑（不是稀疏），但是同样能够保证解中接近于0（但不是等于0，所以相对平滑）的维度比较多，降低模型的复杂度。



*Ref:

[机器学习-泛化能力](https://segmentfault.com/a/1190000016425702 "机器学习-泛化能力")

[PCA主成分分析](https://www.zhihu.com/question/41120789?sort=created)

[《浅析多项式回归与sklearn中的Pipeline](http://mp.weixin.qq.com/s?__biz=MzI4MjkzNTUxMw==&mid=2247484400&idx=1&sn=3ca55d15e7ccd2d6234a5cf5c7abff73&chksm=eb932a46dce4a3509cfab261d80748b2a6eab43d09142a9838c7a4b0d9fb87772673d5494f0d&scene=21#wechat_redirect)》

[《ML/DL重要基础概念：偏差和方差](http://mp.weixin.qq.com/s?__biz=MzI4MjkzNTUxMw==&mid=2247484409&idx=1&sn=740b2a7b4201d7d2e0186e590e8e4a30&chksm=eb932a4fdce4a3593542dc91dda56ca5c92a673b56013d18fc502963bf8ab3e4626f90ec83fa&scene=21#wechat_redirect)》

[《（理论+代码）模型正则化：L1正则、L2正则](http://mp.weixin.qq.com/s?__biz=MzI4MjkzNTUxMw==&mid=2247484437&idx=1&sn=40f4b448ed6b26b5e67690764a3f0cbb&chksm=eb932da3dce4a4b5820f1f8a6616edc08bd6700fc03055ed14a6eb4148d97c9fc299af94f3b1&scene=21#wechat_redirect)》



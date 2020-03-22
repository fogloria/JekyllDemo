---
layout: post
title:  "ML week #4 LinearCongression"
date:   2020-03-22 11:34:00 +0800
categories: 
---
## 机器学习week 4: 线性回归算法是模型之母

#### 损失函数

最小二乘法（最小化误差的平方）
$$
a=\frac{\sum_{i=1}^m (x_i-\overline{x})\times(y_i-\overline{y})}{\sum_{i=1}^m({x_i-\overline{x}})^2}
$$

$$
b=\overline{y}-a\overline{x}
$$

线性：方程是线性（一次函数）的；

回归：用方程来模拟变量之间是如何关联的。

结果有很好的可解释性

需要一条直线，最大程度的拟合样本特征和样本数据标记之间的关系。

建模过程，就是找到一个模型，最大程度的拟合数据。

要想最大程度的拟合数据，本质上就是找到没有拟合的部分，即损失的部分尽量小，就是**损失函数**（loss function）（也有算法衡量拟合程度，称函数为**效用函数**（utility function））。

所有算法模型依赖于最小化或最大化某一个函数，称之为“目标函数”。

“损失函数”是最小化一组函数。描述了单个样本预测值和真实值之间误差的程度，用来度量模型一次预测的好坏。

#### 期望风险

是损失函数的期望，表达理论上模型f(X)关于联合分布P(X,Y)的平均意义下的损失，也叫期望损失/风险。

#### 经验风险

模型f(X)关于训练数据集的平均损失，称为经验风险/损失。

#### 结构风险最小化

当样本容量不大时，经验风险最小化容易“过拟合”，为减缓该问题，提出结构风险最小理论。结构风险最小化为经验风险与复杂度同时较小。公式上表现为，比经验风险多一个**正则化项(regularizer)**，也叫罚项(penalty)，是函数的复杂度J(f)*权重系数λ。

>1、损失函数：单个样本预测值和真实值之间误差的程度。
>
>2、期望风险：是损失函数的期望，理论上模型f(X)关于联合分布P(X,Y)的平均意义下的损失。
>
>3、经验风险：模型关于训练集的平均损失（每个样本的损失加起来，然后平均一下）。
>
>4、结构风险：在经验风险上加上一个正则化项，防止过拟合的策略。



### 最小二乘法

*二乘* 指**平方**。

#### 总结

一类机器学习算法的基本思路：

1. 通过分析问题，确定问题的损失函数/效用函数；
2. 通过最优化损失/效用函数，获得机器学习的模型。

### 代码实现

不同的数据可选择不同的函数，通过最小二乘法得到不一样的拟合曲线。

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.array([1.,2.,3.,4.,5.])
y = np.array([1.,3.,2.,3.,5,])

plt.scatter(x,y)
plt.axis([0,6,0,6])
plt.show()

# 首先要计算x,y的均值
x_mean=np.mean(x)
y_mean=np.mean(y)

# a的分子num、分母d
num=0.0
d=0.0
for x_i,y_i in zip(x,y): # zip函数打包成[(x_i,y_i)...] 的形式
    num=num+(x_i-x_mean)*(y_i-y_mean)
    d=d+(x_i-x_mean)**2
a=num/d
b=y_mean-a*x_mean

# 在求出a,b后，可计算出y的预测值，首先绘制模型直线：
y_hat=a*x+b
plt.scatter(x,y) # 绘制散点图
plt.plot(x,y_hat,color='r') # 绘制直线
plt.axis([0,6,0,6])
plt.show()

# 进行预测
x_predict=6
y_predict=a*x_predict+b
print(y_predict)
```



其中

> num=num+(x_i-x_mean)*(y_i-y_mean)
>
> d=d+(x_i-x_mean)**2

可看成两个向量的对应项相乘再相加，即两个向量**“点乘”**。可用numpy中的dot运算。

**向量化**是十分常用的加速计算方式，特适合深度学习等需要大数据的领域。(相对的，for循环是很慢的……)

#### 创建一个SimpleLinearRegression.py

```python
import numpy as np
class SimpleLinearRegression:
  def __init__(self):
    """模型初始化函数"""
    self.a_=None
    self.b_=None
    
   def fit(self,x_train,y_train):
    """根据训练数据集训练模型"""
    assert x_train.ndim==1 # 简单线性回归模型仅能处理一维向量特征
    assert len(x_train)==len(y_train) # 特征向量的长度与标签的长度相同
    x_mean=np.mean(x_train)
    y_mean=np.mean(y_train)
    num=(x_train-x_mean).dot(y_train-y_mean)
    d=(x_train-x_mean).dot(x_train-x_mean)
    self.a=num/d
    self.b=y_mean-self.a_*x_mean
    
    return self
  
  def predict(self,x_predict):
    """给定待预测数据集x_predict，返回表示其结果向量"""
    assert x_predict.ndim==1
    assert self.a_ is not None and self.b_ is not None
    return np.array([self.predict(x) for x in x_predict])
  
  def _predict(self,x_single):
    """给定单个待预测数据，返回预测结果值"""
    return self.a_*x_single+self.b_
  
  def __repr__(self):
    """返回一个可以用来表示对象的可打印字符串"""
    return "SimpleLinearRegression()"
```



#### 评价指标——R方➡️ 1-预测值与真实值之差的平方/均值与真实值之差的平方

预测值=样本均值 是baseline模型，该模型错误较多，自己的模型错误较少，因此R方衡量了拟合住数据的地方，即*没有产生错误的相应指标*。因此：

> R方越大，错误率越低，最大值为1；
>
> 当R方<0，则自己的模型不如基准模型，很可能数据不存在线性关系。



### 多元线性回归

多元线性回归的正规方程解

缺点：时间复杂度较高O(n^3)

优点：不需对数据归一化处理，原始数据计算参数，不存在量纲问题

#### 代码实现

```python
import numpy as np
from .metrics import r2_score

class LinearRegression:
  def __init__(self):
    """初始化模型"""
    self.coef_=None # 系数（theta0~1向量）
    self.interception_=None # 截距（theta0数）
    self._theta=None # 整体计算出的向量theta
    
   def fit_normal(self,X_train,y_train):
    """根据训练数据训练模型"""
    assert X_train.shape[0]==y_train.shape[0] # size必须相等
    # 正规化方程求解
    X_b=np.hstack([np.ones((len(X_train),1)),X_train])
    self._theta=np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
      
    self.interception_=self._theta[0]
    self.coef_=self._theta[1:]
    return self
  
  def predict(self,X_predict):
    """给定预测的数据集X_predict,返回表示结果向量"""
    assert self.interception_ is not None and self.coef_ is not None
    assert X_predict.shape[1]==len(self.coef_)
    X_b=np.hstack([np.ones((len(X_predict),1)),X_predict])
    y_predict=X_b.dot(self._theta)
    return y_predict
  
  def score(self,X_test,y_test):
    """确定当前模型准确率"""
    y_predict=self.predict(self,X_test)
    return r2_score(y_test,y_predict)
  
  def __repr__(self):
    return "LinearRegression()"
      
```

⚠️ 错误待解决🤔

![84bNMn.png](https://s1.ax1x.com/2020/03/22/84bNMn.png)






---
layout: post
title:  "ML week #7 Logistic Regression"
date:   2020-04-12 23:50:00 +0800
categories: 
---
## 逻辑回归的本质及数学推导

对问题划分层次，利用非线性变换和线性模型的组合，将未知复杂问题分解为已知简单问题。

预测样本发生的概率是多少，由于概率是一个数，因此被叫做“逻辑回归”。

比如某银行使用逻辑回归做风控模型，先设置一个阈值0.5，如果得到它逾期的概率大于0.5，就不放款；否则就放款。对于“放款” or “不放款”来说，实际上是一个标准的分类问题。**逻辑回归只能解决二分类问题，如果是多分类问题，LR本身是不支持的。**

数学家们发现：正态分布在线性变换下保持稳定，而逻辑分布可以很好地近似正态分布。因此可以使用标准逻辑分布的累积分布函数σ(t) 来替换正态分布的累积分布函数Fε(t)。

标准逻辑分布的概率密度函数为
$$
f(x)=\frac{e^-x}{(1+e^-x)^2}
$$
，对应的积累分布函数为：
$$
\sigma(t)=\frac{1}{1+e^-t}
$$
在学术界被称为**sigmoid函数**，是在数据科学领域，特别是神经网络和深度学习领域中非常重要的函数！值域为（0，1）



> 逻辑回归假设数据服从伯努利分布，通过极大似然函数的方法，运用梯度下降来求解参数，来达到将数据二分类的目的。



**为什么要使用sigmoid函数作为假设？**

因为线性回归模型的预测值为实数，而样本的类标记为（0,1），我们需要将分类任务的真实标记y与线性回归模型的预测值联系起来，也就是**找到广义线性模型中的联系函数**。**如果选择单位阶跃函数的话，它是不连续的不可微。而如果选择sigmoid函数，它是连续的**，而且能够将z转化为一个接近0或1的值。



### 损失函数

逻辑回归的损失函数不是定义出来的，而是根据**最大似然估计**推导出来的。并使用**梯度下降**法得到参数。



## 逻辑回归代码实现与调用

我们在线性回归的基础上，修改得到逻辑回归。主要内容为：

- 定义sigmoid方法，使用sigmoid方法生成逻辑回归模型
- 定义损失函数，并使用梯度下降法得到参数
- 将参数代入到逻辑回归模型中，得到概率
- 将概率转化为分类

```python
import numpy as np
# 因为逻辑回归是分类问题，因此需要对评价指标进行更改
from .metrics import accuracy_score

class LogisticRegression:

    def __init__(self):
        """初始化Logistic Regression模型"""
        self.coef_ = None
        self.intercept_ = None
        self._theta = None

    """
    定义sigmoid方法
    参数：线性模型t
    输出：sigmoid表达式
    """
    def _sigmoid(self, t):
        return 1. / (1. + np.exp(-t))
    
    """
    fit方法，内部使用梯度下降法训练Logistic Regression模型
    参数：训练数据集X_train, y_train, 学习率, 迭代次数
    输出：训练好的模型
    """
    def fit(self, X_train, y_train, eta=0.01, n_iters=1e4):
        
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        """
        定义逻辑回归的损失函数
        参数：参数theta、构造好的矩阵X_b、标签y
        输出：损失函数表达式
        """
        def J(theta, X_b, y):
            # 定义逻辑回归的模型：y_hat
            y_hat = self._sigmoid(X_b.dot(theta))
            try:
                # 返回损失函数的表达式
                return - np.sum(y*np.log(y_hat) + (1-y)*np.log(1-y_hat)) / len(y)
            except:
                return float('inf')
        """
        损失函数的导数计算
        参数：参数theta、构造好的矩阵X_b、标签y
        输出：计算的表达式
        """
        def dJ(theta, X_b, y):
            return X_b.T.dot(self._sigmoid(X_b.dot(theta)) - y) / len(y)

        """
        梯度下降的过程
        """
        def gradient_descent(X_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):
            theta = initial_theta
            cur_iter = 0
            while cur_iter < n_iters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient
                if (abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
                    break
                cur_iter += 1
            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        # 梯度下降的结果求出参数heta
        self._theta = gradient_descent(X_b, y_train, initial_theta, eta, n_iters)
        # 第一个参数为截距
        self.intercept_ = self._theta[0]
        # 其他参数为各特征的系数
        self.coef_ = self._theta[1:]
        return self

    """
    逻辑回归是根据概率进行分类的，因此先预测概率
    参数：输入空间X_predict
    输出：结果概率向量
    """
    def predict_proba(self, X_predict):
        """给定待预测数据集X_predict，返回表示X_predict的结果概率向量"""
        assert self.intercept_ is not None and self.coef_ is not None, \
            "must fit before predict!"
        assert X_predict.shape[1] == len(self.coef_), \
            "the feature number of X_predict must be equal to X_train"

        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        # 将梯度下降得到的参数theta带入逻辑回归的表达式中
        return self._sigmoid(X_b.dot(self._theta))

    """
    使用X_predict的结果概率向量，将其转换为分类
    参数：输入空间X_predict
    输出：分类结果
    """
    def predict(self, X_predict):
        """给定待预测数据集X_predict，返回表示X_predict的结果向量"""
        assert self.intercept_ is not None and self.coef_ is not None, \
            "must fit before predict!"
        assert X_predict.shape[1] == len(self.coef_), \
            "the feature number of X_predict must be equal to X_train"
        # 得到概率
        proba = self.predict_proba(X_predict)
        # 判断概率是否大于0.5，然后将布尔表达式得到的向量，强转为int类型，即为0-1向量
        return np.array(proba >= 0.5, dtype='int')

    def score(self, X_test, y_test):
        """根据测试数据集 X_test 和 y_test 确定当前模型的准确度"""

        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)

    def __repr__(self):
        return "LogisticRegression()"
```



## 决策边界、多项式、正则化

### 决策边界

所谓决策边界就是能够把样本正确分类的一条边界，主要有线性决策边界(linear decision boundaries)和非线性决策边界(non-linear decision boundaries)。

注意：决策边界是假设函数的属性，由参数决定，而不是由数据集的特征决定。

### 多项式

为逻辑回归算法添加多项式项。设置pipeline。列表中每个元素是管道中的一步，每一步是一个元组，元组第一个元素是字符串表示做什么，第二个元素是类的对象。管道的第一步是添加多项式项，第二部是归一化，第三部进行逻辑回归过程，返回实例对象。

### 正则化

添加多项式项之后，模型会变变得很复杂，非常容易出现过拟合。因此就需要使用正则化，且sklearn中的逻辑回归，都是使用的正则化。





Ref:

[《出场率No.1的逻辑回归算法，是怎样“炼成”的？](http://mp.weixin.qq.com/s?__biz=MzI4MjkzNTUxMw==&mid=2247484074&idx=1&sn=25a66eedf3a9e7cb439e157309614f88&chksm=eb932b1cdce4a20a3ba127426fd1a406feb9cca75f1ae575ad4bbd9dd087b1d7a035aca570fa&scene=21#wechat_redirect)》

[《逻辑回归的本质及其损失函数的推导、求解](http://mp.weixin.qq.com/s?__biz=MzI4MjkzNTUxMw==&mid=2247484100&idx=1&sn=50c9caf07c84135b467305685472f2cc&chksm=eb932b72dce4a264b29f18d427547b516c2d2e91f825a658da5d385144952fb6c26d54129c7a&scene=21#wechat_redirect)》

[《逻辑回归代码实现与调用](http://mp.weixin.qq.com/s?__biz=MzI4MjkzNTUxMw==&mid=2247484105&idx=1&sn=7ad5725fc9a2bba86c96ff352924f19e&chksm=eb932b7fdce4a269b1d964081481632b52cd795bb04b0baacb23c99bf8f5789d7ef2e601c6f9&scene=21#wechat_redirect)》
[《逻辑回归的决策边界及多项式](http://mp.weixin.qq.com/s?__biz=MzI4MjkzNTUxMw==&mid=2247484138&idx=1&sn=8bbc9f2a4c17a95ea0f11bb2714c38eb&chksm=eb932b5cdce4a24a056e1876903cf2eaab4a8ca0af72780a5e8dc3f4c9a2562f669366523217&scene=21#wechat_redirect)》

[《sklearn中的逻辑回归中及正则化](http://mp.weixin.qq.com/s?__biz=MzI4MjkzNTUxMw==&mid=2247484159&idx=1&sn=abc9a968a839383677d68de50d320581&chksm=eb932b49dce4a25fc86e5c6a924decb85852c009a006b0a70d0976d6658c5a8dcf3220ba1149&scene=21#wechat_redirect)》
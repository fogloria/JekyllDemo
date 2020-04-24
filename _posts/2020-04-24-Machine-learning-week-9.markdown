---
layout: post
title:  "ML week #9 PCA"
date:   2020-04-24 21:56:06 +0800
categories: 
---
## 主成分分析法的思想及其原理

PCA principle component analysis：析取主成分显出的最大的个别差异，发现便于人类理解的特征 / 削减回归分析聚类分析中变量的数目。

如何找到让样本间距最大的轴？

“映射”：将n维特征映射到k维上，这k维是全新的正交特征也被称为主成分，是在原有n维特征的基础上重新构造出来的k维特征。我们要选择的就是让映射后样本间距最大的轴。

> - 样本归0
> - 找到样本点映射后方差最大的单位向量



## PCA算法的实现及调用

求前n个主成分：现将数据集在第一个主成分上的分量去掉，在没有第一个主成分的基础上再寻找第二个主成分。

在二维数据中求得第一主成分分量就够了，对于高维数据需要先将数据集在第一个主成分上的分量去掉，然后在没有第一个主成分的基础上再寻找第二个主成分，依次类推，求出前n个主成分。

```python
import numpy as np


class PCA:
    def __init__(self, n_components):
        # 主成分的个数n
        self.n_components = n_components
        # 具体主成分
        self.components_ = None

    def fit(self, X, eta=0.001, n_iters=1e4):
        '''均值归零'''
        def demean(X):
            return X - np.mean(X, axis=0)

        '''方差函数'''
        def f(w, X):
            return np.sum(X.dot(w) ** 2) / len(X)
        
        '''方差函数导数'''
        def df(w, X):
            return X.T.dot(X.dot(w)) * 2 / len(X)

        '''将向量化简为单位向量'''
        def direction(w):
            return w / np.linalg.norm(w)

        '''寻找第一主成分'''
        def first_component(X, initial_w, eta, n_iters, epsilon=1e-8):
            w = direction(initial_w)
            cur_iter = 0
            while cur_iter < n_iters:
                gradient = df(w, X)
                last_w = w
                w = w + eta * gradient
                w = direction(w) 
                if(abs(f(w, X) - f(last_w, X)) < epsilon):
                    break      
                cur_iter += 1     
            return w
        
        # 过程如下：
        # 归0操作
        X_pca = demean(X)
        # 初始化空矩阵，行为n个主成分，列为样本列数
        self.components_ = np.empty(shape=(self.n_components, X.shape[1]))
        # 循环执行每一个主成分
        for i in range(self.n_components):
            # 每一次初始化一个方向向量w
            initial_w = np.random.random(X_pca.shape[1])
            # 使用梯度上升法，得到此时的X_PCA所对应的第一主成分w
            w = first_component(X_pca, initial_w, eta, n_iters)
            # 存储起来
            self.components_[i:] = w
            # X_pca减去样本在w上的所有分量，形成一个新的X_pca，以便进行下一次循环
            X_pca = X_pca - X_pca.dot(w).reshape(-1, 1) * w
        
        return self

    # 将X数据集映射到各个主成分分量中
    def transform(self, X):
        assert X.shape[1] == self.components_.shape[1]
        return X.dot(self.components_.T)

    def inverse_transform(self, X):
        return X.dot(self.components_)
```





## 数据降维应用：降噪&人脸识别

PCA通过选取主成分将原有数据映射到低维数据再映射回高维数据的方式进行一定程度的降噪。







Ref:

[数据降维1：主成分分析法思想及原理](http://mp.weixin.qq.com/s?__biz=MzI4MjkzNTUxMw==&mid=2247484343&idx=1&sn=6a7dd3b9979b306265da0747f15064e2&chksm=eb932a01dce4a317c6c344dde4b4e30c99e46fd06416508997043d17d2b4899a649b7cc570c5&scene=21#wechat_redirect)

[数据降维2：PCA算法的实现及使用](http://mp.weixin.qq.com/s?__biz=MzI4MjkzNTUxMw==&mid=2247484331&idx=1&sn=8e7b882d2e14e3c32d2a27669962b44b&chksm=eb932a1ddce4a30b65d82dcaf9b4f2967f14cd9f2bc532f9c8e186d5dd4e9ad3a5dbfa4027c6&scene=21#wechat_redirect)

[数据降维3：降维映射及PCA的实现与使用](http://mp.weixin.qq.com/s?__biz=MzI4MjkzNTUxMw==&mid=2247484370&idx=1&sn=fe01e5057f94c248ce69ef8766bffcb8&chksm=eb932a64dce4a3729c046346aa71a5ba2285e2f5237fe710bac805312db36379609fbd21430a&scene=21#wechat_redirect)

[数据降维之应用：降噪&人脸识别](http://mp.weixin.qq.com/s?__biz=MzI4MjkzNTUxMw==&mid=2247484382&idx=1&sn=d8d488b01935ca5e7dc05a9ee302cf03&chksm=eb932a68dce4a37e5ee4b576b56daba6bc2deee243a9a7c3e87ca56f5f602e00c6eb676a5f69&scene=21#wechat_redirect)




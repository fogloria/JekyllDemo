---
layout: post
title:  "Machine learning week #1 sklearn KNN"
date:   2020-03-01 15:59:25 +0800
categories: 
---
# 机器学习Week 1：kNN算法

## 知识点总结

* 什么是kNN？

  判定一个新样本的属性/取值，观察离它最近的K个样本。

  近：常用欧式距离，决定*权重*。

  分类（属性）：K的取值不同，结果不同，引申出不同结果的不同概率。//检测垃圾邮件

  回归（取值）：*平均值*的计算。//股价预测

* kNN的特性

  懒惰学习，在“训练”数据集时不耗时，新样本出现时才开始计算距离；

  维度高/新样本多，运算复杂。

  👉 引申：KD树（取中值分叉，比较距离大小。若len(points)//2=3, 取points[3]为中值 ）

* sklearn

  为模型拟合，数据处理，模型选择与评估等提供工具

* 监督学习与无监督学习

  机器学习的流程：训练数据集👉机器学习算法 -fit（拟合）👉模型 输入样例👉模型 -predict👉输出结果

  > It is called **supervised** because of the presence of the outcome variable to guide the learning process. In the **unsupervised** learning problem,we observe only the features and have no measurements of the outcome.					
  >
  > *——The elements of statistical learning*

  

## 实践代码/案例分析

1. kNN算法：

```python
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from collections import Counter

raw_data_x=[[3.393533211, 2.331273381],
            [3.110073483, 1.781539638],
            [1.343853454, 3.368312451],
            [3.582294121, 4.679917921],
            [2.280362211, 2.866990212],
            [7.423436752, 4.685324231],
            [5.745231231, 3.532131321],
            [9.172112222, 2.511113104],
            [7.927841231, 3.421455345],
            [7.939831414, 0.791631213]
           ]
raw_data_y=[0,0,0,0,0,1,1,1,1,1]
X_train=np.array(raw_data_x)
Y_train=np.array(raw_data_y)

plt.scatter(X_train[Y_train==0,0],X_train[Y_train==0,1],color="g",label="negitive")
plt.scatter(X_train[Y_train==1,0],X_train[Y_train==1,1],color="r",label="positive")
plt.xlabel('tumor size')
plt.ylabel('time')
plt.axis([0,10,0,5])
plt.show()

x=[8.90933607318, 3.365731514]
distances[sqrt(np.sum((x_train-x)**2)) for x_train in X_train]
distances

nearest=np.argsort(distances)
nearest

k=6
topK_y=[y_train[i] for i in nearest[:k]]
topK_y

votes=Counter(topK_y)
votes

votes.most_common(1)
predict_y=votes.most_common(1)[0][0]
predict_y
```

2. 代码封装：

```python
import numpy as np
from math import sqrt
from collections import Counter

class kNNClassifier:
  	def __init__(self,k):
      """初始化分类器"""
      assert k>=1,"k must be valid"
      self.k=k
      self._X_train=None
      self._y_train=None
      
    def fit(self,X_train,y_train):
  """根据训练数据集X_train和y_train训练kNN分类器"""
  			assert X_train.shape[0]==y_train.shape[0],
    		assert self.k<=X_train.shape[0],
   			self._X_train=X_train
        self._y_train=y_train
       	return self
      
    def predict(self,X_predict):
      """根据目标样本X_predict预测标签y_predict"""
      	assert self._X_train is not None and self._y_train is not None,
        assert X_predict.shape[1]==self._X_train.shape[1],
        y_predict=[self._predict(x) for x in X_predict] 
        return np.array(y_predict)
      
    def _predict(self,x):
      """定义预测结果的模型"""
      	distances=[sqrt(np.sum((x_train-x)**2)) for x_train in self._X_train]
        nearest=np.argsort(distances)
        topK_y=[self._y_train[i] for i in nearest]
        votes=Counter(topK_y)
        return votes.most_common(1)[0][0]
      
    def __repr__(self):
      	return "kNN(k=%d)" % self.k
```

**疑问**：在jupyter中魔法命令后会报错 TypeError: 'module' object is not callable

3. 在sklearn中调用kNN：

```python
from sklearn.neighbors import KNeighborsClassifier
raw_data_x=[[3.393533211, 2.331273381],
            [3.110073483, 1.781539638],
            [1.343853454, 3.368312451],
            [3.582294121, 4.679917921],
            [2.280362211, 2.866990212],
            [7.423436752, 4.685324231],
            [5.745231231, 3.532131321],
            [9.172112222, 2.511113104],
            [7.927841231, 3.421455345],
            [7.939831414, 0.791631213]
           ]
raw_data_y=[0,0,0,0,0,1,1,1,1,1]
X_train=np.array(raw_data_x)
y_train=np.array(raw_data_y)
x=np.array([8.90933607318, 3.365731514])

# 创建kNN_classifier实例
kNN_classifier = KNeighborsClassifier(n_neighbors=6)

# kNN_classifier做一遍fit(拟合)的过程，没有返回值，模型就存储在kNN_classifier实例中
kNN_classifier.fit(X_train, y_train)

# kNN进行预测predict，需要传入一个矩阵，而不能是一个数组。reshape()成一个二维数组，第一个参数是1表示只有一个数据，第二个参数-1，numpy自动决定第二维度有多少
y_predict = kNN_classifier.predict(x.reshape(1,-1))
y_predict
```

待完成：使用sklearn中提供的机器学习库完成一些小demo。



## 参考tips：

@饼干：X_train[y_train==0,0]是个二维数组，数组的第一个参数表示行，这里面是一个布尔表达式即选取y_train值为0的那一行；第二个参数为0，表示第0列。

scikit-learn 之 kNN 分类 https://www.joinquant.com/view/community/detail/bb850ee76d1cae16cc587f29c4439ebd






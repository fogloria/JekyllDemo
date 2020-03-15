---
layout: post
title:  "ML week #3 Preprocessing"
date:   2020-03-15 23:08:00 +0800
categories: 
---
# 机器学习Week 3：数据预处理和特征工程

## 数据归一化

把所有的数据映射到同一尺度上

### 最值归一化：把所有数据映射到 0-1 之间。特征的分布具有明显边界，受outlier影响较大。

### 均值方差归一化：均值为0，方差为1。数据没有明显的边界，可能存在极端数据值。



最值归一化的实现：

```python
import numpy as np
# 创建100个随机数
x = np.random.randint(0,100,size=100)
# 最值归一化（向量）
# 最值归一化公式，映射到0，1之间
(x - np.min(x)) / (np.max(x) -  np.min(x))
# 最值归一化（矩阵）
# 0～100范围内的50*2的矩阵
X = np.random.randint(0,100,(50,2))
# 将矩阵改为浮点型
X = np.array(X, dtype=float)
# 最值归一化公式，对于每一个维度（列方向）进行归一化。
# X[:,0]第一列，第一个特征
X[:,0] = (X[:,0] - np.min(X[:,0])) / (np.max(X[:,0]) - np.min(X[:,0]))
# X[:,1]第二列，第二个特征
X[:,1] = (X[:,1] - np.min(X[:,1])) / (np.max(X[:,1]) - np.min(X[:,1]))
# 如果有n个特征，可以写个循环：
for i in range(0,2):    
  X[:,i] = (X[:,i]-np.min(X[:,i])) / (np.max(X[:,i] - np.min(X[:,i])))
 
import matplotlib.pyplot as plt
#简单绘制样本，看横纵坐标
plt.scatter(X[:,0],X[:,1])
plt.show()
```



方差归一化的实现：

```python
import numpy as np
import matplotlib.pyplot as plt
X2 = np.array(np.random.randint(0,100,(50,2)),dtype=float)
#套用公式，对每一列做均值方差归一化
for i in range(0,2):
  X2[:,i]=(X2[:,i]-np.mean(X2[:,i]))/np.std(X2[:,i])
plt.scatter(X2[:,0],X2[:,1])
plt.show()

np.mean(X2[:,0])
np.std(X2[:,1])
```



建模时要将数据集划分为**训练数据集 & 测试数据集**

训练数据集的均值mean_train,方差std_train，对测试数据集，用训练的均值和方差。(x_test - mean_train)/std_train

```python
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
#加载鸢尾花数据集
iris=datasets.load_iris()
X = iris.data
y = iris.target
X_train,X_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.2,random_state=666)
#使用数据归一化
from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()
#归一化的过程根训练模型一样
standardScaler.fit(X_train)
standardScaler.mean_
standardScaler.scale_ #表述数据分布范围的变量，替代std_
#使用transform
X_train_standard = standardScaler.transform(X_train)
X_test_standard = standardScaler.transform(X_test)
```



## KNN算法优缺点

* 和朴素贝叶斯等算法比，对数据没有假设，准确度高，对异常点不敏感
* 主要靠周围有限的邻近样本，不靠判别类域的方法确定所属类别，对类域的交叉或重叠较多的待分样本集来说，KNN较其他方法更适合
* 样本不平衡时，对稀有类别预测准确率低
* 可解释性低
* 维度灾难：随着维度增加，看似相近的亮点距离越来越大，而KNN非常依赖距离



## KD树

* 选择切分维度：方差越大，分布越分散，从方差大的维度开始切分，有较好的切分效果和平衡性。

* 确定中值点：若len(points)//2=3, 取points[3]

* 维度的变化：二维是：x->y->x…；三维则是：x->y->z->x…

  

**检索**：从顶点开始，以距离为半径画圆，根据交叉情况更新最近邻点。若遍历第二层左右叶子节点，与当前最优距离相等，则不更新（以当前为最近邻）。



sklearn中的KDtree

```python
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from sklearn.neighbors import KDTree
np.random.seed(0)
points = np.random.random((100,2))
tree = KDTree(points)
point = points[0]
#kNN
dists,indices = tree.query([point],k=3)
print(dists,indices)
#query radius
indices = tree.query_radius([point],r=0.2)
print(indices)
fig = plt.figure()
ax = fig.add_subplot(111,aspect='equal')
ax.add_patch(Circle(point,0.2,color='r',fill=False))
X,Y=[p[0] for p in points], [p[1] for p in points]
plt.scatter(X,Y)
plt.scatter([point[0]],[point[1]],c='r')
plt.show()
```



# 特征工程

？？？数据标准化的原因：

数量级的差异将导致迭代收敛速度减慢；

当使用梯度下降法寻求最优解时，很有可能走“之字型”路线（垂直等高线走），从而导致需要迭代很多次才能收敛；



特征值服从正态分布，标准化后，转换成标准正态分布。

MinMax归一化，MinMaxScaler对异常值的存在非常敏感。

MaxAbs归一化，属性缩放到[-1,1]，不会破坏任何稀疏性。



## 归一化与标准化：

同：都能取消由于量纲不同引起的误差，是线性变换，对向量X按照比例压缩再进行平移。

异：

* 归一化消除量纲压缩到 [0,1] 区间，标准化调整特征整体的分布；
* 归一化与最大值最小值相关，标准化与均值标准差相关；
* 归一化输出在 [0,1] 之间，标准化无限制。

若数据存在异常值和较多噪音，用标准化，可间接通过中心化避免异常值和极端值的影响

Normalization归一化，Standardization标准化，regularization正则化

## 归一化与标准化的应用场景：

- 在分类、聚类算法中，需要使用距离来度量相似性的时候（如SVM、KNN）、或者使用PCA技术进行降维的时候，标准化(Z-score standardization)表现更好；

- 在不涉及距离度量、协方差计算、数据不符合正太分布的时候，可以使用第一种方法或其他归一化方法。

  比如图像处理中，将RGB图像转换为灰度图像后将其值限定在[0 255]的范围；

- 基于树的方法不需要进行特征的归一化。

  例如随机森林，bagging与boosting等方法。

  如果是基于参数的模型或者基于距离的模型，因为需要对参数或者距离进行计算，都需要进行归一化。

**一般来说，建议优先使用标准化。对于输出有要求时再尝试别的方法，如归一化或者更加复杂的方法。很多方法都可以将输出范围调整到[0, 1]，如果我们对于数据的分布有假设的话，更加有效的方法是使用相对应的概率密度函数来转换。**

**除了上面介绍的方法外，还有一些相对没这么常用的处理方法：RobustScaler、PowerTransformer、QuantileTransformer和QuantileTransformer等。**

![88yhrQ.png](https://s1.ax1x.com/2020/03/15/88yhrQ.png)

## 无监督分箱：

等距、等频、二值化Binarization、

聚类（有序性）——

```python
from sklearn.cluster import KMeans
kmodel=KMeans(n_clusters=k) #k为聚成几类
kmodel.fit(data.reshape(len(data),1)) #训练模型
c=pd.DataFrame(kmodel.cluster_centers_) #求聚类中心
c=c.sort_vlues(by='列索引') #排序
w=pd.rolling_mean(c,2).iloc[1:] #用滑动窗口求均值的方法求相邻两项求中点，作为边界点
w=[0]+list(w[0]+[data.max()]) #把首末边界点加上
d3=pd.cut(data,w,labels=range(k)) #cut函数
```



## 有监督分箱：

待进一步学习：卡方分箱法，最小熵法





参考资料：https://www.cnblogs.com/juanjiang/archive/2019/05/30/10948849.html






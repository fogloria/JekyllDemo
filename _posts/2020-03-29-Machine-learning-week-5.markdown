---
layout: post
title:  "ML week #5 Gradient descent"
date:   2020-03-29 23:23:00 +0800
categories: 
---
## 最优化方法：梯度下降

梯度下降优化算法，对原始模型的损失函数进行优化，以寻找最优的参数，使损失函数值最小。

梯度下降算法，抓住参数与损失值之间的导数，即能够计算梯度gradient，通过导数得知此刻某参数应该朝什么方向，以怎样的速度运动，能安全高效降低损失值，朝最小损失值靠拢。

梯度下降，是一种基于搜索的最优化方法，对原始模型的损失函数进行优化，找到使损失函数（局部）最小的参数。

#### 梯度

梯度是多元函数的导数，对每个变量微分，是**向量**

例，二元函数
$$
f(x_1,x_2)=x_1^2+x_1x_2-3x_2
$$
则f的梯度是
$$
\bigtriangledown f=(\frac{\partial f}{\partial x_1},\frac{\partial f}{\partial x_2})=2x_1+x_2,x_1-3
$$
对应的，在点(1,2)处，梯度取值为(4,-2)。当梯度为0时，就是优化问题的解。为了找到这个解，我们沿着梯度的反方向进行线性搜索，从而减少误差值。

**随机选择一个方向，每次迈步都选择最陡的方向，直到这个方向上能达到的最低点。**

超参数：学习率/eta learning rate, 初始点集（多组a0,b0)

##### 学习率

当学习率过小，收敛学习速度变慢，使得算法的效率降低；学习率过大又会导致不收敛，在“错误的道路上”越走越远。我们要对异常进行进行处理。

梯度下降之前要使用归一化

#### 随机梯度下降与批量梯度下降的比较：

如果是批量搜索，那么每次都是沿着一个方向前进，但是随机梯度下降法由于不能保证随机选择的方向是损失函数减小的方向，更不能保证一定是减小速度最快的方向，所以搜索路径就会呈现下图的态势。即随机梯度下降有着不可预知性。但实验结论告诉我们，通过随机梯度下降法，依然能够达到最小值的附近（用精度换速度）。

**随机梯度下降法的过程中，学习率的取值很重要**，这是因为如果学习率一直取一个固定值，所以可能会导致点已经取到最小值附近了，但是固定的步长导致点的取值又跳去了这个点的范围。因此我们希望在随机梯度下降法中，**学习率是逐渐递减的**。

批量梯度下降法BGD(Batch Gradient Descent)。

- 优点：全局最优解；易于并行实现；
- 缺点：当样本数据很多时，计算量开销大，计算速度慢。

针对于上述缺点，其实有一种更好的方法：随机梯度下降法SGD（stochastic gradient descent），随机梯度下降是每次迭代使用一个样本来对参数进行更新。

- 优点：计算速度快；
- 缺点：收敛性可能不好。



待续：https://www.bilibili.com/video/BV1Lx411j7iv

https://www.bilibili.com/video/BV1Ux411j7ri





### 学习内容

[《还不了解梯度下降法？看完这篇就懂了！](http://mp.weixin.qq.com/s?__biz=MzI4MjkzNTUxMw==&mid=2247483976&idx=1&sn=aedbed8f21deeb02d0bcb9341a99435b&chksm=eb932bfedce4a2e808a24c726a305e1b92e14f05f4cf7625cdb4d3ae24deee476b0231ff51ce&scene=21#wechat_redirect)》

《[手动实现梯度下降（可视化）](http://mp.weixin.qq.com/s?__biz=MzI4MjkzNTUxMw==&mid=2247483985&idx=1&sn=759dc972a7dc1bd01af53b68619c01c8&chksm=eb932be7dce4a2f161a08ff529050f8a105c54d58452aa05a2569320f1626e8f5e204016abc4&scene=21#wechat_redirect)》

《[线性回归中的梯度下降](http://mp.weixin.qq.com/s?__biz=MzI4MjkzNTUxMw==&mid=2247484001&idx=1&sn=9e7a22277acf5049fd1d945bfe4229db&chksm=eb932bd7dce4a2c11b91f4bcaa7de8cc35f041651b9481f4a72f83f5c2897abac992b949cf02&scene=21#wechat_redirect)》

《[速度更快的随机梯度下降法》](http://mp.weixin.qq.com/s?__biz=MzI4MjkzNTUxMw==&mid=2247484045&idx=1&sn=621cbf1b174b0c6d30cc8747b667a909&chksm=eb932b3bdce4a22d7a998e715f9db9400437de2d39dafd3d6885e7be0a9a40f4cf6248caf016&scene=21#wechat_redirect)

《[梯度下降番外：非常有用的调试方式及总结](http://mp.weixin.qq.com/s?__biz=MzI4MjkzNTUxMw==&mid=2247484074&idx=2&sn=6ec6cc66c9b865b7f304604172e11b2b&chksm=eb932b1cdce4a20a0b4dbd471d586501b998c4e69237baebf2ebecac4a54c4d379d09583c5c8&scene=21#wechat_redirect)》






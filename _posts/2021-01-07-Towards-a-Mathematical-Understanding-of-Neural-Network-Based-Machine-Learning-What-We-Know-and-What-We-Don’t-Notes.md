---
layout: post
title: "Towards a Mathematical Understanding of Neural Network-Based Machine Learning: What We Know and What We Don’t Notes"
date: 2021-01-07
image: /images/cover/C_Scenery10.jpg         
tags: [Paper-Notes]
toc: false
published: false
---
{: class="table-of-content"}
* TOC
{:toc}

# "Towards a Mathematical Understanding of Neural Network-Based Machine Learning: What We Know and What We Don’t" ——Notes



“工欲善其事，必先利其器”，有一套好的底层语言对于构建一个理论体系是必要的，那么现在有这么一篇综述性质的文章，总结了关于对于神经网络的机器学习模型的数学方面研究的进展，那肯定是要好好读一下的。

## Introduction

神经网络模型，他可以非常强，对高维空间中的函数也可以从未有过的拟合精确度和效率，但是它也被人戏称为黑盒模型，从数学研究的角度，需要解决的问题有两个：

1. 做出一套好的理论模型来解释它成功的原因
2. 提出效果更加好，更加鲁棒的模型

实际上，现在大部分数学上的工作还是停留在第一个问题上的，而有一种很好的设计方法是先提出一个好的连续的模型然后离散化处理（ResNet？）

### The setup of supervised learning

监督学习的问题范式是如此提出的：

给出数据集$$ S=\left\{\left(\boldsymbol{x}_{i}, y_{i}=f^{*}\left(\boldsymbol{x}_{i}\right)\right), i \in[n]\right\} $$，去给出 $$ f $$ 的估计$$ f^{*} $$ ，为了简单化处理，不妨假设数据被限制 $$ x_{i} \in X:=[0,1]^{d} $$，同时被映射到一个限制区域 $$\sup _{x \in X}  \mid f^{*} (x)\mid \leq 1 $$ 中，然后就是标准的处理流程：

1. 选取一个假设空间，以及一个备择函数集，被记为 $$ \mathcal{H}_{m} $$，经典方法往往选取多项式或者分段多项式

2. 选一个损失函数，通过最小化“经验损失” ,有时候会加上一个正则项

   $$
   \hat{\mathcal{R}}_{n}(f)=\frac{1}{n} \sum_{i}\left(f\left(x_{i}\right)-y_{i}\right)^{2}=\frac{1}{n} \sum_{i}\left(f\left(x_{i}\right)-f^{*}\left(x_{i}\right)\right)^{2}
   $$

3. 选择优化算法，比如说GD，SGD等等进行训练

训练的最终目标，是模型的泛化性，或者说最小化“总体损失”，实际操作中，利用合理采样的测试集来代替总体的分布进行计算

$$
\mathcal{R}(f)=\mathbb{E}_{x \sim P}\left(f(x)-f^{*}(x)\right)^{2}
$$

### The main issues of interest

那么在问题的范式给出了以后，关于神经网络的数学研究主要是从以下三个方面给出的

1. 假设空间角度，给出一个假设空间，它里面的备择函数用来模拟什么样的函数比较好，以及用训练集损失和测试集损失与泛化性之间的差异
2. 损失函数的角度，损失函数往往是非凸的，那么往往会出现多个鞍点和不太好的局部最优
3. 训练算法的角度，有两个问题，算法能优化当前的问题嘛？优化出的结果它的泛化性如何？

当问题背景如下设定时：

- $$m$$ : 自由变量的个数
- $$n$$ : 训练集的大小
- $$t$$ :训练的迭代数
- $$d$$ : 输入的维度

研究比较多的是$$ m, n, t \rightarrow \infty \text ， d \gg 1$$

### Approximation and estimation errors

如果以最优解必然存在作为前提的话，不妨假设问题的最优解为 $$f^{*}$$，当前模型的最优解为 $$\hat f$$，而当前假设空间下的模型解为 $$ f_{m} $$

$$
f_{m}=\operatorname{argmin}_{f \in \mathcal{H}_{m}} \mathcal{R}(f)
$$

误差 $$f^{*}-\hat{f}$$ 可以被拆解为两部分：

$$
f^{*}-\hat{f}=f^{*}-f_{m}+f_{m}-\hat{f}
$$

第一部分误差 $$f^{*}-f_{m}$$ 是，由于假设空间的有限的选择，导致解的理想最优解和模型最优解的最优解的差距，被称为逼近误差（approximation error）

第二部分误差 $$f_{m}-\hat{f}$$ ，是在导致假设空间的模型固定的情况下，由于数据集有限导致的现实最优解和模型最优解的差距，被称为估计误差（estimation error）

#### Approximation Error

在传统模型，比如多项式，截断傅立叶级数的假设空间下，可以给出一个误差上界：

$$
\left\|f-f_{m}\right\|_{L^{2}(X)} \leq C_{0} m^{-\alpha / d}\|f\|_{H^{\alpha}(X)}
$$

为了达到合适的精度，在问题维度变大一些的情况下，把误差控制到同一个数量级，由 $$m^{-\alpha / d}$$ 一项可以看出，需要指数级增长的模型大小来抵消线性增长的问题维度的影响，这是一个重要的问题，被称为“维度诅咒”（Curse of Dimenson）

对于CoD也不是没有办法,拿一个具体的实例来说，在高维空间中有随机变量 $$X$$ 与函数 $$g(X)$$ ，要估算 $$E(g)$$ 

$$
E(g)=\int_{X} g(x) d x=I (g)
$$

这个问题当然可以用传统的网格法去计算，但是这样由于问题维度上升，计算量就增长的很快。但是如果用蒙特卡洛法，从 $$X$$ 中抽样一个 $$i.i.d$$ 的样本 $$\left\{\boldsymbol{x}_{i}\right\}_{i=1}^{n}$$，

$$
I_{n}(g)=\frac{1}{n} \sum_{i=1}^{n} g\left(\boldsymbol{x}_{i}\right)
$$

这个的误差可以由切比雪夫不等式控制：

$$
\mathbb{E}\left(I(g)-I_{n}(g)\right)^{2}=\frac{\operatorname{Var}(g)}{n}
\\
\operatorname{Var}(g)=\int_{X} g^{2}(\boldsymbol{x}) d \boldsymbol{x}-\left(\int_{X} g(\boldsymbol{x}) d \boldsymbol{x}\right)^{2}
$$

于是蒙特卡洛算法的收敛速度是 $$O(1 / \sqrt{n})$$，这和维度 $$d$$ 无关，证明了蒙特卡洛法可以解决维度诅咒问题

于是可以看出关于逼近误差的控制，可以通过选取合适的算法（假设空间）来限制为了达到一定逼近误差的计算量。

#### Estimation Error

关于估计误差，这部分是由数据集所导致的问题，以龙格现象为例，尽管对于数据集拟合的很好，但是逼近误差就很大了：

<img src="/Users/karlwu/Documents/GitHub/changhaowu.github.io/images/2021-01-07-Towards-a-Mathematical-Understanding-of-Neural-Network-Based-Machine-Learning-What-We-Know-and-What-We-Dont-Note/Runge-phenomenon.png" alt="Runge-phenomenon" style="zoom:40%;" />

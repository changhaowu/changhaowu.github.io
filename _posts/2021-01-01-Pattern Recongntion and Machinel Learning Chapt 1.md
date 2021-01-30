---
layout: post
title: "PRML Chapt 1 Introduction Notes"
date: 2021-01-01
image: /images/cover/C_Design3.jpeg   
tags: [PRML-Notes]
toc: true
published: true
---

# Chapt-1-Introduction-Notes

## 1.1 Concept & Definition

模式识别（Pattern Recongntion）在历史上就是长期被研究的问题，在天文学中有J.Kepler通过研究行星观测的数据总结出了Kepler定理这一经验公式，量子力学中寻找原子谱（Atomic Spectra）也同样是一个模式识别问题。模式识别可以被总结为在给出的数据中去寻找该数据的总体的一般的规律，而回到机器学习中，以一个经典的模式识别问题：手写体识别为例，给出一些关于模式识别问题的定义

{:refdef: style="text-align: center;"}
<img src="/images/2021-01-01-Pattern-Recongntion-and-Machinel-Learning-Chapt-1-Introduction/Zip Codes handwriting.png" alt="Zip Codes handwriting" style="zoom:35%;" />
{:refdef}

手写体识别的，输入是 $$28 \times 28$$ 的像素图 $$\mathbf{x}$$ ，输出是 $$0-9$$ 的识别数字结果的一个0-1整数向量 $$\mathbf{t}$$ ,模式识别问题的任务就是要找出一个映射：

$$
\mathbf{y}:\mathbf{x} \in \mathbf{X} \rightarrow \mathbf{t} \in \mathbf{T}
$$


在手写体识别问题中，会给出训练集 $$ \left\{\mathbf{x}_{1}, \ldots, \mathbf{x}_{N}\right\} $$ 和对应的标签  $$ \left\{\mathbf{t}_{1}, \ldots, \mathbf{t}_{N}\right\} $$ ,通过训练集去训练可调节模型 $$\mathbf{y}$$,在通过训练之后，通过一个从与 $$\mathbf{x}$$ 同分布中采样获得的测试集去检验模型，具体来说，被检验的模型性能被称为泛化性（Generalization），模型的泛化性是模式识别的中心问题。

在实际过程中，在获取 $$\mathbf{x}$$ 的几何训练集前，还有一个过程被称为预处理（Pre-poccessing）,预处理目标是提取出对问题关键的信息而过滤对问题无用的信息，这样的好处不少，比如可以减少计算时间。

像手写体识别这样的，训练集中有对应的 $$<\mathbf{x},\mathbf{t}>$$ 给出的模式识别被定义为监督学习 （Supervised Learning ），$$\mathbf{t}$$ 可以通过有限的离散的形式表达的被称为分类（classification），而后面给出的曲线拟合的例子，其结果 $$\mathbf{t}$$ 需要给出连续的形式被称为回归（Regression）

当然有些问题，训练集中仅仅给出了 $$\mathbf{x}$$ ，则被称为无监督学习（Unsupervised Learning），其根据学习目标，有几个代表性问题，有比如要找出相似的分组比如聚类（Clustering）,又比如找出给出的数据的分布（density estimation），也有从高维空间投影到二维空间这样来做可视化（VIsualization）

还有一类问题，训练集是以一个状态（State）给出的，目标是要把给出的状态标定一个合理的值被称为（Reward），比如下棋时给出一个棋局，再给出相应的奖励/状态值，来驱动问题，这样就可以做到自动下棋，不深入赘述。

## 1.2 Polynomial Curve Fitting

手写体问题的现行主流解决方案不太数学，于是换一个比较数学的问题，曲线多项式拟合来详细给出一个模式识别问题从给出到解决的过程。

该问题为一个监督回归问题，问题给了训练集，以形式：$ \mathbf{x} \equiv\left(x_{1}, \ldots, x_{N}\right)^{\mathrm{T}},
 \mathbf{t} \equiv\left(t_{1}, \ldots, t_{N}\right)^{\mathrm{T}} $，而该数据集是通过在 $  y = \sin (2 \pi x)  $ 的基础上加一个服从高斯分布的随机误差产生的，这模拟了一般场景的情况，训练集可视化如下：

{:refdef: style="text-align: center;"}
<img src="/images/2021-01-01-Pattern-Recongntion-and-Machinel-Learning-Chapt-1-Introduction/data set visualization.png" alt="data set visualization" style="zoom:35%;" />
{:refdef}

曲线拟合问题的目标是给出一个新的 $$ \widehat{x} $$ 时通过模型 $$ y(x,\mathbf{w}) $$ 给出 $$ \widehat{t} $$, $$\mathbf{w}$$ 为可调节参数，具体来说，多项式拟合采用多项式函数:

$$
y(x, \mathbf{w})=w_{0}+w_{1} x+w_{2} x^{2}+\ldots+w_{M} x^{M}=\sum_{j=0}^{M} w_{j} x^{j}
$$

该函数对于系数 $$ \mathbf{w} $$ 来说是线性的，并且有可调节参数 $$ \mathbf{M} $$ ，任何对模型的优化都应当给出一个标准（citeria），这里采用平方误差：

$$
\mathbf{w^{*}=\arg \min_{\mathbf{w}}}E(\mathbf{w})=\arg \min_{\mathbf{w}}\frac{1}{2} \sum_{n=1}^{N}\left\{y\left(x_{n}, \mathbf{w}\right)-t_{n}\right\}^{2}
$$

值得注意的是，由于 $$ y(x,\mathbf{w}) $$ 对于 $$ \mathbf{w} $$ 来说是线性的，于是可知 $$\frac{d E(\mathbf{w})}{d w}$$ 是线性的，因此 $$\mathbf{w^{}}$$ 有唯一解，在给出了使 $$ E(\mathbf{w})$$ 最小的 $$\mathbf{w^{}}$$ 后，就完成了多项式曲线拟合的问题，但具体实践上，还有两个问题有待解决，可调节参数 $$ \mathbf{M} $$ 的选择以及给出具体的数据集后 $$\mathbf{w^{*}}$$ 的计算。

### 1.2.1 The choice of Hyperparameter

对于问题中的待拟合的函数，$$ y = \sin (2 \pi x) $$ 其泰勒展开为一个 $$x^{k}$$ 的无穷级数，因此直觉上来说，当给出的拟合多项式的可调节参数 $$ \mathbf{M} $$ 越大，拟合的效果越好。但是实际上，反直觉的是，如同下图所示，在当可调节参数 $$ \mathbf{M} $$ 为 $$9$$ ，即给出的样本数时，反而发生了过拟合现象，也就是说，给出的Citeria其实并没有做到和模型泛化性的等价，反而变成了一个求根问题，当给出 $$9$$ 个样本时，参数变成了求过 $$9$$ 点的方程。

{:refdef: style="text-align: center;"}
<img src="/images/2021-01-01-Pattern-Recongntion-and-Machinel-Learning-Chapt-1-Introduction/overfit_curfit.png" alt="overfit_curfit" style="zoom:35%;" />
{:refdef}

为了修正标准，需要做两件事情，考虑到更加合理的对于泛化性的表达（主要矛盾）和考虑样本量的影响（次要矛盾）

先解决简单的样本量，当固定当可调节参数 $$ \mathbf{M} $$ 为 $$9$$ ，换一个数据量大的训练集，同时为了减少数据量对于标准的影响，采用 RMS (root-mean-squared) 误差：

$$
E_{\mathrm{RMS}}=\sqrt{2 E\left(\mathbf{w}^{\star}\right) / N}
$$

换了 $$N=15$$ 以及  $$N=100$$ 的训练集以后，发现过拟合现象减少了

{:refdef: style="text-align: center;"}
<img src="/images/2021-01-01-Pattern-Recongntion-and-Machinel-Learning-Chapt-1-Introduction/More-data-better-generalization.png" alt="More-data-better-generalization" style="zoom:35%;" />
{:refdef}

但是往往实际问题中，这种通过增大样本量来解决问题的方案是不可行的

于是回到主要矛盾，对于要求泛化性，需要的是泛化性，但是数值上的标准给出的是尽可能的去拟合样本数据，这其实是不等价的。采用观察试验结果的方法，发现过拟合的 $$\mathbf{w}^{*}$$ 在会震荡的非常的厉害，即数值上非常大来满足完全过给出的样本点的要求：

{:refdef: style="text-align: center;"}
<img src="/images/2021-01-01-Pattern-Recongntion-and-Machinel-Learning-Chapt-1-Introduction/Parameter oscillation .png" alt="Parameter oscillation " style="zoom:35%;" />
{:refdef}

于是考虑修正误差函数，即限制参数在数值上的大小来换取泛化性，于是考虑 $$w$$ 的范数，$$\|\mathbf{w}\|^{2} \equiv \mathbf{w}^{\mathrm{T}} \mathbf{w}=w_{0}^{2}+w_{1}^{2}+\ldots+w_{M}^{2} $$ ，同时有 $$\lambda$$ 来控制对于样本点拟合精度和参数大小的比例，同是值得一提的是，有时候 $$w_{0}$$ 项会被省略因为他不会根据 $$x$$ 波动，省略它反而提高了常数项的拟合精度

$$
\widetilde{E}(\mathbf{w})=\frac{1}{2} \sum_{n=1}^{N}\left\{y\left(x_{n}, \mathbf{w}\right)-t_{n}\right\}^{2}+\frac{\lambda}{2}\|\mathbf{w}\|^{2}
$$

在调节不同的参数 $$\lambda$$ 后，一个好的 $$\lambda$$ 可以使泛化性和拟合精度达到一个平衡：

{:refdef: style="text-align: center;"}
<img src="/images/2021-01-01-Pattern-Recongntion-and-Machinel-Learning-Chapt-1-Introduction/modified_error.png" alt="modified_error" style="zoom:40%;" />
{:refdef}

若没有 Bayesian 的相关基础，请先移步至后面的 [Bayesian Probability](#1.4bayesian-probability) 部分

## 1.3 Curve Fitting Re-visited

那么在具体的给出了关于多项式拟合问题的损失函数后，当然已经可以对其进行数值优化了，之前的推导是就是利用预设的，规定的，常用的 $$L^2$$ 均方损失函数得出的，直接利用已有的结论当然是可以的，但是忽略 $$L^2$$ 损失函数推导本身是一种损失，PRML提供了一种角度，可以推导出均方损失函数，在贝叶斯主义的角度下，$$L^2$$ 正则项也得到了理论框架下的解释 

### 1.3.1 Frequentist Curve Fitting

在概率角度下，利用已知的数据去预测新的输入相应的输出可以用一个正态分布描述，在观察到 $ \mathbf{x} \equiv\left(x_{1}, \ldots, x_{N}\right)^{\mathrm{T}},
 \mathbf{t} \equiv\left(t_{1}, \ldots, t_{N}\right)^{\mathrm{T}} $ 下，有新的输入 $$x$$ ，需要利用已有的观测去估计一些中间变量（在多项式拟合问题中，是利用观测去估计 $$\mathbf{w},\beta$$ ），描述关于新的输入 x 的输出 $$y$$ 的不确定性：


$$
p(t \mid x, \mathbf{w}, \beta)=\mathcal{N}\left(t \mid y(x, \mathbf{w}), \beta^{-1}\right)
$$

不管贝叶斯还是频率，关于利用正态分布描述，这一点是达成一致的，而 $$\beta=(\sigma^2)^{-1}$$ 为准度（precision parameter）

{:refdef: style="text-align: center;"}
<img src="/images/2021-01-01-Pattern-Recongntion-and-Machinel-Learning-Chapt-1-Introduction/Schematic_illustration_posterior_polynomial.png" alt="Schematic_illustration_posterior_polynomial" style="zoom:30%;" />
{:refdef}

现在有训练集 $$\{\mathbf{x}, \mathbf{t}\}$$，利用数据集去确定参数 $$\mathbf{w},\beta$$ ，如果有假设数据集 $$\mathbf{x}$$ ，那么可以给出似然函数：

$$
\begin{aligned}
p(\mathbf{t} \mid \mathbf{x}, \mathbf{w}, \beta)&=\prod_{n=1}^{N} \mathcal{N}\left(t_{n} \mid y\left(x_{n}, \mathbf{w}\right), \beta^{-1}\right)
\\
\ln p(\mathbf{t} \mid \mathbf{x}, \mathbf{w}, \beta)&=-\frac{\beta}{2} \sum_{n=1}^{N}\left\{y\left(x_{n}, \mathbf{w}\right)-t_{n}\right\}^{2}+\frac{N}{2} \ln \beta-\frac{N}{2} \ln (2 \pi)
\end{aligned}
$$

而上式的第一部分正是之前得出的不带正则项的损失函数，利用最大似然优化得到：

$$
\mathbf{w}_{ML}=\arg\min_{\mathbf{w}}\frac{\beta}{2} \sum_{n=1}^{N}\left\{y\left(x_{n}, \mathbf{w}\right)-t_{n}\right\}^{2}
$$

而准度 $$\beta$$ 也可以进一步得出：

$$
\frac{1}{\beta_{\mathrm{ML}}}=\frac{1}{N} \sum_{n=1}^{N}\left\{y\left(x_{n}, \mathbf{w}_{\mathrm{ML}}\right)-t_{n}\right\}^{2}
$$

有了 $$\mathbf{w}_{ML}$$ 和 $$\beta_{ML}$$ 后，就可以得出 $$t$$ 的似然函数，并进一步优化就可以得出 $$t$$

$$
p\left(t \mid x, \mathbf{w}_{\mathrm{ML}}, \beta_{\mathrm{ML}}\right)=\mathcal{N}\left(t \mid y\left(x, \mathbf{w}_{\mathrm{ML}}\right), \beta_{\mathrm{ML}}^{-1}\right)
$$

### 1.3.2 Prior distribution

上述过程是频率主义的观点，很不贝叶斯。频率主义嘛，数据量不大的时候，就会严重的过拟合，之前利用了 $$L^2$$ 正则项进行优化，这源于一个idea：“过拟合往往是由 $$\mathbf{w}_{ML}$$ 在数值上非常大导致的”，那么贝叶斯的先验分布，也是基于这样的想法，因此规定 $$\mathbf{w}$$ 先验分布为：

$$
p(\mathbf{w} \mid \alpha)=\mathcal{N}\left(\mathbf{w} \mid \mathbf{0}, \alpha^{-1} \mathbf{I}\right)=\left(\frac{\alpha}{2 \pi}\right)^{(M+1) / 2} \exp \left\{-\frac{\alpha}{2} \mathbf{w}^{\mathrm{T}} \mathbf{w}\right\}
$$

$$\alpha$$ 是先验分布的准度，而 $$M+1$$ 是多项式的次数，也是正态分布的维度，这个问题里面， $$M$$ 是超参数。由贝叶斯公式，在参数 $$\mathbf{w}$$ 的角度下：

$$
p(\mathbf{w} \mid \mathbf{x}, \mathbf{t}, \alpha, \beta) \propto p(\mathbf{t} \mid \mathbf{x}, \mathbf{w}, \beta) p(\mathbf{w} \mid \alpha)
$$

代入先验分布和似然函数，利用贝叶斯优化中最大后验（ $$maximum\;posterior$$ ），取 $$\lambda=\alpha / \beta$$，上述优化等价于：

$$
\mathbf{w}_{ML}=\arg_{\mathbf{w}}\min\sum_{n=1}^{N}\left\{y\left(x_{n}, \mathbf{w}\right)-t_{n}\right\}^{2}+\lambda \; \mathbf{w}^{\mathrm{T}} \mathbf{w}
$$

这正是上述的带正则的损失函数

### 1.3.2 Full Bayesian

完全贝叶斯是不停的利用概率的加，乘和边际来进行，在多项式曲线拟合中，给了训练集 $$\{\bold{x}，\bold{t}\}$$，有新的输入 $$x$$ ，要去预测 $$t$$，因此要计算 $$p(t \mid x, \mathbf{x}, \mathbf{t})$$ ，在这里假设参数 $$\alpha,\beta$$ 已知，则有：

$$
p(t \mid x, \mathbf{x}, \mathbf{t})=\int p(t \mid x, \mathbf{w}) p(\mathbf{w} \mid \mathbf{x}, \mathbf{t}) \mathrm{d} \mathbf{w}
$$

由之前给出的后验分布的替代和预测值的分布：

$$
p(t \mid x, \mathbf{w}, \beta)=\mathcal{N}\left(t \mid y(x, \mathbf{w}), \beta^{-1}\right)
\\
p(\mathbf{w} \mid \mathbf{x}, \mathbf{t}, \alpha, \beta) \propto p(\mathbf{t} \mid \mathbf{x}, \mathbf{w}, \beta) p(\mathbf{w} \mid \alpha)
$$

在进行一次对参数 $$\mathbf{w}$$ 的积分后，就可以解析的算出：

$$
\begin{aligned}
p(t \mid x, \mathbf{x}, \mathbf{t})&=\mathcal{N}\left(t \mid m(x), s^{2}(x)\right)
\\
m(x) &=\beta \phi(x)^{\mathrm{T}} \mathbf{S} \sum_{n=1}^{N} \phi\left(x_{n}\right) t_{n} \\
s^{2}(x) &=\beta^{-1}+\phi(x)^{\mathrm{T}} \mathbf{S} \phi(x)
\\
\mathbf{S}^{-1}&=\alpha \mathbf{I}+\beta \sum_{n=1}^{N} \boldsymbol{\phi}\left(x_{n}\right) \boldsymbol{\phi}(x)^{\mathrm{T}}
\end{aligned}
$$

关于如何的解析的算出将在后面第三章具体给出，下面是贝叶斯方法去拟合带噪声的正弦函数，参数 $$\alpha,\beta$$ 在左边给出：

{:refdef: style="text-align: center;"}
<img src="/images/2021-01-01-Pattern-Recongntion-and-Machinel-Learning-Chapt-1-Introduction/bayesian_curve_fitting.png" alt="bayesian_curve_fitting" style="zoom:35%;" />
{:refdef}

## 1.4 Bayesian Probability

“贝叶斯是个好东西”，发自内心的，我这么想，尤其是在现代算力大爆发的情况下，在某些领域，样本远比算力来的值钱，这个时候，强依赖大量样本的频率主义思维就会弱于贝叶斯主义方法。贝叶斯方法，是在对于一件事情有一个先验认识的时候，结合试验得到的数据修正，那么就有两个具体的问题，知识怎么用概率来表示，以及如何用实验数据结果修正。

首先解释如何将知识变成概率，用概率表示不确定性，是大家都认可的方法，但是用概率去表示知识（common sense），就有一些难以接受了，但是 通过表示该知识的置信程度，Cox(1946)建立了[Cox Theorem](https://en.wikipedia.org/wiki/Cox%27s_theorem) 来将知识转化成先验分布，这个过程是符合概率的加，乘的

或者说，拿估计投硬币的正面概率这个最为简单的例子来说，依赖先验知识，对概率空间的划分有一个先验分布，因为我对于投硬币的结果是正面这件事，我的知识告诉我，若它是均匀硬币，则投掷出正面的结果是 $$1/2$$ ，反之亦然，即：

$$
P(\text { Head })=1 / 2 
\\
P(\text { Tail })=1 / 2
$$

然后通过实验的样本可能逐渐发现硬币并非均匀的，会有一些偏差，这是通过实验数据进行修正的，而如果是频率主义的话，这枚硬币的试验结果不好，可能就是 $$100\%$$ 正面硬币了

{:refdef: style="text-align: center;"}
<img src="/images/2021-01-01-Pattern-Recongntion-and-Machinel-Learning-Chapt-1-Introduction/Coin-toss-problem.png" alt="Coin-toss-problem" style="zoom:20%;" />
{:refdef}

第二个问题是具体的如何利用样本和先验修正概率，在给出观测样本 $$ \mathcal{D}=\left\{t_{1}, \ldots, t_{N}\right\} $$ 以及 先验分布$$ p(\mathbf{w}) $$ 后，就可以利用贝叶斯公式来得到修正过的概率/后验概率 $$ p(\mathcal{D} \mid \mathbf{w}) $$：

$$
p(\mathbf{w} \mid \mathcal{D})=\frac{p(\mathcal{D} \mid \mathbf{w}) p(\mathbf{w})}{p(\mathcal{D})}
$$

而值得一提的是，$$ p(\mathcal{D} \mid \mathbf{w})$$ 尽管形式上似乎是 $$\mathbf{w}$$ 的概率，实际上并非如此，其被称为似然函数（likelihood function），于是在 $$\mathbf{w}$$ 的意义下，贝叶斯公式也可以被总结为：

$$
\text { posterior } \propto \text { likelihood } \times \text { prior }
$$

话说再进一步的话，成为常数部分的 $$p(\mathcal{D})$$ 其实是在参数空间上 $$p(\mathcal{D} \mid \mathbf{w}) p(\mathbf{w})$$ 进行了积分（marginlization）

$$
p(\mathcal{D})=\int p(\mathcal{D} \mid \mathbf{w}) p(\mathbf{w}) \mathrm{d} \mathbf{w}
$$

因此变成了在 $$\mathbf{w}$$ 意义下的常数

无论是频率主义还是贝叶斯主义， $$p(\mathcal{D} \mid \mathbf{w})$$ 似然函数都很重要，但是在如何看待 $$\mathbf{w}$$ 上，两者是有分歧的，在频率主义中， $$\mathbf{w}$$  认为是一个固定的参数，利用数据去进行估计，而贝叶斯主义则认为， $$\mathbf{w}$$ 是不确定的，有且仅有 $$\mathcal{D}$$ 是观察到的，确定性的，其余的参数都要利用一个关于参数  $$\mathbf{w}$$ 的分布进行描述

一个常用的频率主义的参数估计法就是最大似然估计 $$maximum \; likelihood$$ ， 使得观察到的数据出现的可能性最大的参数就是 $$\mathbf{w}^{*}$$
$$
\mathbf{w}^{*}=arg\max_{\mathbf{w}} p(\mathcal{D} \mid \mathbf{w})
$$

## 1.5 The Curse of Dimensionality

维度诅咒，指模型的复杂度以模型输入维度的多项式级增加的现象

往往发生在输入的向量维度较大的时候，大部分算法都难以避免这个问题。比如下面的分类问题，有数据集 $$\mathcal{D} $$ ，要预测新的点 $$x$$ 的类，一种很朴素的方法就是在根据输入 $$x$$ 附近的格子内，频率最高的类来分类（最近邻算法），如下图所示：

<img src="/images/2021-01-01-Pattern-Recongntion-and-Machinel-Learning-Chapt-1-Introduction/naive_clustering.png" alt="naive_clustering" style="zoom:40%;" />

但是尽管简单，该算法本身只需要根据统计最近邻的格子，避免了计算复杂度，但是却仍然对于数据集提出了要求：同样数量的数据集，假设较为均匀的分布在空间内，可能二维内一个体积为1的格子内有较多的数据可以用来分类，同样数量的数据集，在十维的空间的格子，甚至都没有一个数据，直观的情况如下图所示：

<img src="/images/2021-01-01-Pattern-Recongntion-and-Machinel-Learning-Chapt-1-Introduction/scarce_dataset.png" alt="scarce_dataset" style="zoom:40%;" />

而如果换一个模型，利用多项式去做拟合边界，在一开始不剪枝的情况下，多项式为：
$$
y(\mathbf{x}, \mathbf{w})=w_{0}+\sum_{i=1}^{D} w_{i} x_{i}+\sum_{i=1}^{D} \sum_{j=1}^{D} w_{i j} x_{i} x_{j}+\sum_{i=1}^{D} \sum_{j=1}^{D} \sum_{k=1}^{D} w_{i j k} x_{i} x_{j} x_{k}
$$
可以看出三次项贡献了绝大部分的系数，也就是说，此时模型本身都出现了维度诅咒的问题，同理维度 $$D$$ 的模型，系数的数量为 $$D^M$$ ，复杂度增长的非常快

但是在实践过程中，由于数据有流形分布律，集中在高位空间中的一个低维流形上，因此模型的大部分系数并没有需要训练，但是维度诅咒仍然是需要不可忽视的问题

## 1.6 Decision Theory

有了概率的方法去描述一件事件的不确定性，进一步的，需要利用决策论来做出最优的决策，下面将一步步的利用一个医学诊疗问题来讨论决策论的一些key idea

现在有一组像素图代表病人的 X-ray 图以及对应的标签  $$\{\mathbf{x},\mathbf{t}\}$$ ，然后要建模去自动诊疗，输入为 $$x$$ ，输出为 $$t=\{0,1\}$$，$$\mathcal{C}_{1}$$ 对应 $$t=0$$ 代表该图为存在癌症，反之 $$\mathcal{C}_{2}$$ 对应 $$t=1$$ 代表该图为没有癌症，于是利用概率建模的话，考虑去建模后验概率 $$p\left( \mathcal{C}_{k} \mid \mathbf{x}\right )$$，利用贝叶斯概率有：
$$
p\left(\mathcal{C}_{k} \mid \mathbf{x}\right)=\frac{p\left(\mathbf{x} \mid \mathcal{C}_{k}\right) p\left(\mathcal{C}_{k}\right)}{p(\mathbf{x})}
$$
其中 $$p\left(\mathcal{C}_{k}\right)$$ 为先验概率，那么一种决策方案就是：最小化分类错误的概率，那么就是选后验概率比较大的类即可

下面来研究错误分类的情况，误检率越小越好，在此之前，当分类器训练完成的时候，就可以把概率空间切割成两部分，$$\mathcal{R}_{k}$$ 中的点被分到 $$ \mathcal{C}_{k}$$ 类中，那么误检率就可以写作：
$$
\begin{aligned}
p(\text { mistake }) &=p\left(\mathbf{x} \in \mathcal{R}_{1}, \mathcal{C}_{2}\right)+p\left(\mathbf{x} \in \mathcal{R}_{2}, \mathcal{C}_{1}\right) \\
&=\int_{\mathcal{R}_{1}} p\left(\mathbf{x}, \mathcal{C}_{2}\right) \mathrm{d} \mathbf{x}+\int_{\mathcal{R}_{2}} p\left(\mathbf{x}, \mathcal{C}_{1}\right) \mathrm{d} \mathbf{x}
\end{aligned}
$$
后验概率 $$p\left(\mathbf{x} \mid  \mathcal{C}_{1}\right) $$ 和联合概率概率 $$p\left(\mathbf{x}, \mathcal{C}_{1}\right)$$中间差一个系数：
$$
p\left(\mathbf{x}, \mathcal{C}_{k}\right)=p\left(\mathcal{C}_{k} \mid \mathbf{x}\right) p(\mathbf{x})
$$
 对于任意的 $$x$$，都是一样的，因此后验概率最大等价于联合概率最大。利用估计出的联合分布，在下图中，利用联合概率最大的决策原则将概率空间划分

<img src="/Users/karlwu/Documents/GitHub/changhaowu.github.io/images/2021-01-01-Pattern-Recongntion-and-Machinel-Learning-Chapt-1-Introduction/decision_region.png" alt="decision_region" style="zoom:40%;" />

刚刚的决策原则在很多场景下都是好的，但是在这个问题中会有一些问题：不妨假设一种比较罕见的病症，但是由于宣传，健康人群会自己误判成疑似病例的情况

那么在大部分的输入 $$x$$ 的实际标签都是 $$t=1$$，于是就有一个很朴素，很傻瓜的分类器，把所有的输入都预测为  $$t=1$$ ，也可以在实际生产场景中得到很低的误检率，都是问题是检测器根本就没有做有意义的预测，而少部分的病人就会承担非常大的恶果，这样之前的决策原则就会出问题，考虑引入数学期望，在训练期间，错误标记成 $$t=1$$ 会产生很大的损失

更加普遍的问题的提法应当是当误分类 $$\mathcal{C}_{k}$$ 为 $$\mathcal{C}_{j}$$ 时产生损失 $$L_{k j}$$ ，于是训练过程变成最小化训练集的损失期望：
$$
\begin{aligned}
\mathbb{E}[L]&=\sum_{k} \sum_{j} \int_{\mathcal{R}_{j}} L_{k j} p\left(\mathbf{x}, \mathcal{C}_{k}\right) \mathrm{d} \mathbf{x}
\\
& \propto \sum_{k} \sum_{j} \int_{\mathcal{R}_{j}} L_{k j} p\left(\mathbf{x}\mid \mathcal{C}_{k}\right) \mathrm{d} \mathbf{x}
\end{aligned}
$$

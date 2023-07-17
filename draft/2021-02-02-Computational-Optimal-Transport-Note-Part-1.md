---
layout: post
title: "Computational Optimal Transport Note Part 1: Theoretical Foundations"
date: 2021-02-02
image: /images/cover/C_Scenery9.jpg   
tags: [OTNotes]
toc: false
# published: true
published: false
---

{: class="table-of-content"}
* TOC
{:toc}

# Computational Optimal Transport Note Part 1: Theoretical Foundations

这一部分主要叙述关于计算最优传输的理论部分，和一般的数学问题一样，也是从最简单的情况开始一步步泛化 ，第一步是从两个 $$\sum$$ 中的概率向量 $$(\mathbf{a}, \mathbf{b})$$ 出发的，而后拓展到两个离散测度 $$(\alpha, \beta)$$，进一步到普通的两个任意测度的问题框架下

## Histograms and Measures

这一部分中，分别在离散情况和连续情况下，叙述了最优传输中，如何数学地定义空间中运输的起始地和目的地，及其上相应的需要搬运的质量

在问题设定下，直方图（Histogram）和概率向量（Probability vector）是两个等价的概念，对于 $$\mathbf{a} \in \Sigma_{n}$$，都表示：

$$
\Sigma_{n} \stackrel{\text { def. }}{=}\left\{\mathbf{a} \in \mathbb{R}_{+}^{n}: \sum_{i=1}^{n} \mathbf{a}_{i}=1\right\} .
$$

这是离散的概率向量，其与离散测度的区别在于，离散测度进一步约束 $$\mathbf{a} \in \Sigma_{n}$$ 中要求非负性的条件，改成通过 DIrac 脉冲函数定义：

$$
\alpha=\sum_{i=1}^{n} \mathbf{a}_{i} \delta_{x_{i}}
$$

其中 $$x_{1}, \ldots, x_{n} \in \mathcal{X}$$ 代表位置，而 $$\text { a } = （\mathbf{a}_{i}）$$ 代表权重要求 $$\mathbf{a}_{i} > 0$$，这避免了没有意义的位置 $$x_{i}$$ 存在

那么进一步的，就想把离散测度拓展到连续测度的情况，而最优传输的框架，一个很大的优点是，离散的情况和连续的情况，是可以在一个框架内进行研究的：

定义连续情况下的积分：对于在空间 $$\mathcal{X}$$ 任意的拉东测度 $$\mathcal{M}(\mathcal{X})$$，$$f \in \mathcal{C}(\mathcal{X})$$，在这个框架下，对于 $$f \in \mathcal{C}(\mathcal{X})$$ 的积分对应一个离散测度 $$\alpha$$ 的求和

$$
\int_{\mathcal{X}} f(x) \mathrm{d} \alpha(x)=\sum_{i=1}^{n} \mathbf{a}_{i} f\left(x_{i}\right)
$$

定义连续情况下的测度变换：若令 $$\mathcal{X}=\mathbb{R}^{d}$$，有测度 $$\alpha$$，其相对应勒贝格测度有 $$\mathrm{d} \alpha(x)=\rho_{\alpha}(x) \mathrm{d} x$$，$$\rho_{\alpha}=\frac{\mathrm{d} \alpha}{\mathrm{d} x}$$：

$$
\forall h \in \mathcal{C}\left(\mathbb{R}^{d}\right), \quad \int_{\mathbb{R}^{d}} h(x) \mathrm{d} \alpha(x)=\int_{\mathbb{R}^{d}} h(x) \rho_{\alpha}(x) \mathrm{d} x
$$

对 $$\alpha$$ 做进一步的说明：

- $$\alpha \in \mathcal{M}(\mathcal{X})$$ 这个约束对于最优传输是很弱的，这不足以不要求测度是以 Dirac 函数的和或者一个密度函数给出，有时候相应的需要在上面的函数进行一些补充，当 $$\mathcal{X}$$ 不是紧空间，补充要求 $$f$$ 有一个紧的支撑集
- 对应于之前离散的概率测度，$$\alpha \in \mathcal{M}_{+}^{1}(\mathcal{X})$$ 为连续情况下的概率测度，$$\alpha(\mathcal{X})=\int_{\mathcal{X}} \mathrm{d} \alpha=1$$

一些直观的可视化：

{:refdef: style="text-align: center;"}
<img src="/images/2021-02-02-Computational-Optimal-Transport-Note-Part-1/Discrete_histogram_density.png" alt="Discrete_histogram_density" style="zoom:40%;" />
{:refdef}

## Assignment and Monge Problem

这一部分开始定义蒙日最优传输问题，离散的情况，利用代价矩阵定义，有一个离散的特殊情况，当目的地与起始地数目一致，其上质量平分质量，对应一个置换；连续的情况需要定义前向算子（Push-forward），而利用前向算子有结论：空间中的一个测度对应一个随机变量

### Discrete Setting

首先是离散的情况，给出代价矩阵 $$\left(\mathbf{C}_{i, j}\right)_{i \in  [n] , j \in [ m ]}$$，和两个离散测度 $$\alpha=\sum_{i=1}^{n} \mathbf{a}_{i} \delta_{x_{i}} , \; \beta=\sum_{j=1}^{m} \mathbf{b}_{j} \delta_{y_{j}}$$

要去求一个映射 $$T:\left\{x_{1}, \ldots, x_{n}\right\} \rightarrow\left\{y_{1}, \ldots, y_{m}\right\}$$，其满足条件：

$$
\forall j \in [ m ], \quad \mathbf{b}_{j}=\sum_{i: T\left(x_{i}\right)=y_{j}} \mathbf{a}_{i}
$$

可以记作 $$T_{\#} \alpha=\beta$$ ，当然这个映射是满射的，可视化传输映射如下图：

{:refdef: style="text-align: center;"}
<img src="/images/2021-02-02-Computational-Optimal-Transport-Note-Part-1/Transport_Map_Visualization.png" alt="Transport_Map_Visualization" style="zoom:40%;" />
{:refdef}

有了传输映射的定义，可以进一步定义蒙日最优传输问题，求取最小化传输代价的前向映射 $$T$$

$$
\min _{T}\left\{\sum_{i} c\left(x_{i}, T\left(x_{i}\right)\right): T_{\#} \alpha=\beta\right\} \quad (x, y) \in \mathcal{X} \times \mathcal{Y}
$$

在离散情况下，可以指标映射 $$\sigma: [ n ] \rightarrow[m]$$ ，定义逆 inverse 为 $$\sigma^{-1}(j)$$ 映射到 $$j$$ 的指标，则有约束：

$$
\sum_{i \in \sigma^{-1}(j)} \mathbf{a}_{i}=\mathbf{b}_{j}
$$

当 $$n=m$$ 且 $$\mathbf{a}_{i}=\mathbf{b}_{j}=1 / n$$，之前在最优传输笔记中的结论有，这样的情况下，传输映射 $$T$$ 是一个双射，其是一个置换

当 $$n<m$$ 时，蒙日传输映射不一定存在，就比如可视化蒙日传输映射中的图例，若反向要求 $$y$$ 到 $$x$$ 的蒙日传输，这是不存在的

### Pushing-forward Operator

对于一个连续映射：$$T: \mathcal{X} \rightarrow \mathcal{Y}$$，可以再进一步的定义一个分布的映射（Pushing-forward Operator） $$T_{\#}: \mathcal{M}(\mathcal{X}) \rightarrow \mathcal{M}(\mathcal{Y})$$，比如离散的情况下，前向算子就是把 $$\alpha$$ 的支撑集变换位置；

$$
T_{\#} \alpha \stackrel{\text { def. }}{=} \sum_{i} \mathbf{a}_{i} \delta_{T\left(x_{i}\right)}
$$

对于 $$T: \mathcal{X} \rightarrow \mathcal{Y}$$，前向算子满足坐标变换公式：

$$
\forall h \in \mathcal{C}(\mathcal{Y}), \quad \int_{\mathcal{Y}} h(y) \mathrm{d} \beta(y)=\int_{\mathcal{X}} h(T(x)) \mathrm{d} \alpha(x)
$$

当然最朴素的，就是质量不变公式：$$\forall \; \beta \text { measurable }  B \subset \mathcal{Y}$$：

$$
\beta(B)=\alpha(\{x \in \mathcal{X}: T(x) \in B\})=\alpha\left(T^{-1}(B)\right)
$$

由于质量不变，因此前向算子的变换仍然是一个测度，即：

$$
\alpha \in \mathcal{M}_{+}^{1}(\mathcal{X}) \Rightarrow T_{\#} \alpha \in \mathcal{M}_{+}^{1}(\mathcal{Y})
$$

### Pull-back Operator

有后向算子，定义在函数上： $$T^{\#}: \mathcal{C}(\mathcal{Y}) \rightarrow \mathcal{C}(\mathcal{X})$$，当然为了满足类似前向算子的坐标变换公式，定义为 $$\forall g \in \mathcal{C}(\mathcal{Y})$$，$$T^{\#} g=g \circ T$$

$$
\forall(\alpha, g) \in \mathcal{M}(\mathcal{X}) \times \mathcal{C}(\mathcal{Y}), \quad \int_{\mathcal{Y}} g \mathrm{~d}\left(T_{\#} \alpha\right)=\int_{\mathcal{X}}\left(T^{\#} g\right) \mathrm{d} \alpha
$$

值得注意的是，后向算子定义在函数上，前向算子定义在测度上，这是不一样的：

{:refdef: style="text-align: center;"}
<img src="/images/2021-02-02-Computational-Optimal-Transport-Note-Part-1/Push_forward_pull_back.png" alt="Push_forward_pull_back" style="zoom:40%;" />
{:refdef}

### Measures and random variables

这个等价性的特性非常的重要，即：

一个 Radon 测度和一个随机变量是等价的，下面证明：

随机变量 $$X: \Omega \rightarrow \mathcal{X}$$，而概率空间上有概率测度 $$(\Omega, \mathbb{P})$$，对应的有 Radon 测度 $$\alpha \in \mathcal{M}_{+}^{1}(\mathcal{X})$$，由于概率测度的大小定义为：

$$
\forall A \in \mathcal{X}, P\left(X^{-1}(A)\right)=\alpha(A) 
$$

因此可以把 $$\alpha$$ 看作 $$\alpha=X_{\#} \mathbb{P}$$，由于事件域是固定的，因此 Radon 测度和随机变量是一一对应的

进一步的，复合映射也是成立的，对于 $$Y=T(X):$$ 

$$
 \omega \in \Omega \rightarrow T(X(\omega)) \in Y
$$

因此这是一个前向的复合，进一步的，在 $$Y$$ 中抽样 $$y$$ 等价于在 $$X$$ 中抽样 $$x$$ 后计算 $$y=T(x)$$

而反回来谈这个等价性，我们知道离散测度的“核”：Dirac 函数 $$\delta \in \mathcal{M}_{+}(\mathcal{X})$$ ，因此离散测度可以和一个随机变量等价，而连续测度 $$\alpha \in \mathcal{M}_{+}^{1}(\mathcal{X})$$，当然也是可以和一个随机变量等价的，而一个随机变量和一个分布是等价的，因此倒出了一个想法，能否通过最优传输代价来定义两个分布之间的距离，当一个离散，一个连续时，后面的 $$W$$ 距离部分会更详细的叙述这个问题

### Arbitrary Setting 

通过定义前向算子，就可以定义任意测度情况下的蒙日最优传输问题，有概率测度 $$(\alpha, \beta)$$，其支撑集为 $$(\mathcal{X}, \mathcal{Y})$$，寻求传输映射 $$ T: \mathcal{X} \rightarrow \mathcal{Y}$$ 使传输代价最小：

$$
\min _{T}\left\{\int_{\mathcal{X}} c(x, T(x)) \mathrm{d} \alpha(x): T_{\#} \alpha=\beta\right\}
$$

## Kantorovich Relaxation

在蒙日形式最优传输中，最大的问题就是一个始发地 $$x_i$$ 不能对应多个目的地 $$y_j$$，康托洛维奇形式摆脱了原先的传输映射应当是一个确定性（deterministic）的想法，变成一个随机性（probabilistic）的想法（可能说的是，原本传输映射非零的条件在康托洛维奇形式中松弛了，从 $$x_i$$ 对应 $$y_j$$ 的传输可能是0，这一点原本在传输映射中是不会发生的，用非零的松弛换了可以表示一个始发地可以传输到多个目的地的想法）

首先是概率向量的情况：

具体来说，是通过建立一个耦合矩阵 $$\mathbf{P} \in \mathbb{R}_{+}^{n \times m}$$ 来替代原先的传输映射，为了符合传输的特性，定义传输计划 $$\mathbf{U}(\mathbf{a}, \mathbf{b}) $$ 为：

$$
\mathbf{U}(\mathbf{a}, \mathbf{b}) \stackrel{\text { def. }}{=}\left\{\mathbf{P} \in \mathbb{R}_{+}^{n \times m}: \mathbf{P} \mathbb{1}_{m}=\mathbf{a} \text { and } \mathbf{P}^{\mathrm{T}} \mathbb{1}_{n}=\mathbf{b}\right\}
\\
\mathbf{P} \mathbb{1}_{m}=\left(\sum_{j} \mathbf{P}_{i, j}\right)_{i} \in \mathbb{R}^{n} \quad 

\quad \mathbf{P}^{\mathrm{T}} \mathbb{1}_{n}=\left(\sum_{i} \mathbf{P}_{i, j}\right)_{j} \in \mathbb{R}^{m}
$$

由于这样的定义，比起蒙日形式定义条件导致的不对称性，康托洛维奇形式是对称的，即：

$$
\mathbf{P} \in \mathbf{U}(\mathbf{a}, \mathbf{b}) \Rightarrow \mathbf{P}^{\mathrm{T}} \in \mathbf{U}(\mathbf{b}, \mathbf{a})
$$

具体的定义康托洛维奇形式最优传输问题：

$$
\mathrm{L}_{\mathrm{C}}(\mathrm{a}, \mathrm{b}) \stackrel{\mathrm{def.}}{=} \min _{\mathbf{P} \in \mathrm{U}(\mathrm{a}, \mathrm{b})}\langle\mathbf{C}, \mathbf{P}\rangle \stackrel{\mathrm{def}}{=} \sum_{i, j} \mathbf{C}_{i, j} \mathbf{P}_{i, j}
$$

这是一个线性规划问题，因此可以知道，康托洛维奇形式最优传输计划不一定是唯一的

正如之前最优传输笔记中提到的，当蒙日形式存在的时候，康托洛维奇形式可能比蒙日形式的传输代价更小，以指派问题为例子：

对于置换 $$\sigma \in \operatorname{Perm}(n)$$，可以定义传输映射：

$$
\forall(i, j) \in [ n ]^{2}, \quad\left(\mathbf{P}_{\sigma}\right)_{i, j}=\left\{\begin{array}{l}
1 / n \quad \text { if } \quad j=\sigma_{i} \\
0 \quad \text { otherwise. }
\end{array}\right.
$$

则蒙日形式的传输代价为：

$$
\left\langle\mathbf{C}, \mathbf{P}_{\sigma}\right\rangle=\frac{1}{n} \sum_{i=1}^{n} \mathbf{C}_{i, \sigma_{i}}
$$

而通过康托洛维奇形式来定义的话

$$
\mathbf{P} \in \mathbf{U}\left(\mathbb{1}_{n} / n, \mathbb{1}_{n} / n\right) \quad \mathbf{P}^{*}=\arg \min _{\mathbf{P} \in \mathbf{U}\left(\mathbb{1}_{n} / n, \mathbb{1}_{n} / n\right)}\langle\mathbf{C}, \mathbf{P}\rangle
$$

可以知道，置换矩阵集合是 $$ \mathbf{U}\left(\mathbb{1}_{n} / n, \mathbb{1}_{n} /n \right)$$ 的子集，因此在指派问题中，康托洛维奇最优传输代价不大于蒙日最优传输代价：

$$
\mathrm{L}_{\mathbf{C}}\left(\mathbb{1}_{n} / n, \mathbb{1}_{n} / n\right) \leq \min _{\sigma \in \operatorname{Perm}(n)}\left\langle\mathbf{C}, \mathbf{P}_{\sigma}\right\rangle
$$

其他的最优传输问题也是一个道理，更不要说如果蒙日形式不存在时了，如下图右边的指派问题：

{:refdef: style="text-align: center;"}
<img src="/images/2021-02-02-Computational-Optimal-Transport-Note-Part-1/assignment_plus.png" alt="assignment_plus" style="zoom:40%;" />
{:refdef}

通过上面定义的概率向量的情况，离散测度的情况是类似定义的

有离散测度 $$\alpha, \beta$$ 及其支撑集，在支撑集上定义传输代价矩阵 $$\mathrm{C}_{i, j} \stackrel{\text { def. }}{=} c\left(x_{i}, y_{j}\right)$$，离散测度的康托洛维奇传输问题等价于概率向量的情况

$$
\mathcal{L}_{c}(\alpha, \beta) \stackrel{\text { def. }}{=} \mathrm{L}_{\mathbf{C}}(\mathbf{a}, \mathbf{b})
$$

 然后是一般形式的康托洛维奇传输问题：

耦合测度定义在 $$\pi \in \mathcal{M}_{+}^{1}(\mathcal{X} \times \mathcal{Y})$$ 上，同时在耦合测度上做一些限制

1. 耦合测度要和之前的 $$\mathbf{P} \mathbb{1}_{m}=\mathbf{a} \quad \mathbf{P}^{\mathrm{T}} \mathbb{1}_{n}=\mathbf{b}$$ 一样，因此有：

   $$
   P_{\mathcal{X} \#} \pi=\alpha 
   \\
   P_{\mathcal{Y}_{\#}} \pi=\beta
   $$

2. 进一步的由于前向算子的性质，可以知道：

   $$
   \pi(A \times \mathcal{Y})=\alpha(A) \; \forall \; A \subset \mathcal{X}
   \\
   \pi(\mathcal{X} \times B)=\beta(B) \; \forall \;  B \subset \mathcal{Y}
   $$

然后就有康托洛维奇最优传输问题：

$$
\mathcal{L}_{c}(\alpha, \beta) \stackrel{\text { def. }}{=} \min _{\pi \in \mathcal{U}(\alpha, \beta)} \int_{\mathcal{X} \times \mathcal{Y}} c(x, y) \mathrm{d} \pi(x, y)
$$

又之前的结论知，一个测度等价于一个随机变量，因此康托洛维奇问题也可以通过随机变量的形式理解，即：

$$
\mathcal{L}_{c}(\alpha, \beta)=\min _{(X, Y)}\left\{\mathbb{E}_{(X, Y)}(c(X, Y)): X \sim \alpha, Y \sim \beta\right\}
$$

下图是蒙日形式传输问题的一个解：

{:refdef: style="text-align: center;"}
<img src="/images/2021-02-02-Computational-Optimal-Transport-Note-Part-1/Kantorovich_Solution.png" alt="Kantorovich_Solution" style="zoom:40%;" />
{:refdef}

下图是康托洛维奇形式传输问题的一个解：

{:refdef: style="text-align: center;"}
<img src="/images/2021-02-02-Computational-Optimal-Transport-Note-Part-1/Kantorovich_Solution.png" alt="Kantorovich_Solution" style="zoom:40%;" />
{:refdef}

### Metric Properties of Optimal Transport

因此正如 WGAN 中利用测地距离定义两个不相交的分布流形之间的距离，Optimal Transport 的一个重要性质就是用来定义分布的距离，当固定传输代价矩阵 $$\mathrm{C}$$ 后，可以证明最优传输给出了一个分布之间的距离

首先在离散情况下证明：

给出传输代价 $$\mathbf{C}=\mathbf{D}^{p}=\left(\mathbf{D}_{i, j}^{p}\right)_{i, j} \in \mathbb{R}^{n \times n}$$，$$\mathbf{D} \in \mathbb{R}_{+}^{n \times n}$$ 是 $$[n]$$ 上的一个距离，有需要满足一些限制：

- $$\mathbf{D} \in \mathbb{R}_{+}^{n \times n}$$ 是对称的
- $$\mathbf{D}_{i, j}=0$$ 当且仅当 $$i=j$$
- 三角不等式 $$\forall(i, j, k) \in [ n ]^{3}, \mathbf{D}_{i, k} \leq \mathbf{D}_{i, j}+\mathbf{D}_{j, k}$$

然后可以通过 $$\mathbf{D}$$ 诱导距离：

$$
\mathrm{W}_{p}(\mathbf{a}, \mathbf{b}) \stackrel{\text { def. }}{=} \mathrm{L}_{\mathbf{D}}^{p}(\mathbf{a}, \mathbf{b})^{1 / p}
$$

那么一样要验证 $$\mathrm{W}_{p}(\mathbf{a}, \mathbf{b})$$ 的对称性和正则性，以及三角不等式，仅叙述三角不等式：

有 $$\mathbf{a}, \mathbf{b}, \mathbf{c} \in \Sigma_{n}$$，以及 $$\mathbf{a}, \mathbf{b} $$ 之间的最优传输 $$ \mathbf{P} $$，$$ \mathbf{b} ,\mathbf{c} $$ 之间的最优传输 $$\mathbf{Q}$$，为了避免 $$\mathbf{b}$$ 的零坐标在后续证明中产生问题，定义： 

$$
\tilde{\mathbf{b}}_{j} \stackrel{\text { def. }}{=}
\begin{array}{l}
\left\{\begin{array}{ll}
\mathbf{b}_{j}  & \text { if } \mathbf{b}_{j}>0 \\
1 & \text { else }
\end{array}\right. \\
\end{array}{}
$$

通过 $$\tilde{\mathbf{b}}_{j}$$ 定义  $$\mathbf{S} \stackrel{\text { def. }}{=} \mathbf{P} \operatorname{diag}(1 / \tilde{\mathbf{b}}) \mathbf{Q} \in \mathbb{R}_{+}^{n \times n}$$，可以证明 $$\mathbf{S} \in \mathbf{U}(\mathbf{a}, \mathbf{c})$$：

$$
\mathbf{S} \mathbb{1}_{n}=\mathbf{P} \operatorname{diag}(1 / \tilde{\mathbf{b}}) \mathbf{Q} \mathbb{1}_{n}=\mathbf{P}(\mathbf{b} / \tilde{\mathbf{b}})=\mathbf{P} \mathbb{1}_{\operatorname{Supp}(\mathbf{b})}=\mathbf{a}
$$

其中 $$\mathbf{P} \mathbb{1}_{\operatorname{Supp}(\mathbf{b})}=\mathbf{P} \mathbb{1}=\mathbf{a}$$ 是由于，当 $$\mathbf{b}_{j}=0$$ 时，说明对应的 $$j$$ 处没有运入，即 $$\mathbf{P}_{i, j}=0$$， 则本来计算 $$\mathbf{P} \mathbb{1}$$，这部分也不产生影响，同理也有 $$\mathbf{S}^{\mathrm{T}} \mathbb{1}_{n}=\mathbf{c}$$，因此有 $$\mathbf{S} \in U(\mathbf{a}, \mathbf{c})$$，则有：

$$
\begin{aligned}
\mathrm{W}_{p}(\mathbf{a}, \mathbf{c}) &=\left(\min _{\mathbf{P} \in U(\mathbf{a}, \mathbf{c})}\left\langle\mathbf{P}, \mathbf{D}^{p}\right\rangle\right)^{1 / p} \leq\left\langle\mathbf{S}, \mathbf{D}^{p}\right\rangle^{1 / p}    &(\mathbf{S} \text{ cost not less than } \mathbf{D})
\\
&=\left(\sum_{i k} \mathbf{D}_{i k}^{p} \sum_{j} \frac{\mathbf{P}_{i j} \mathbf{Q}_{j k}}{\tilde{\mathbf{b}}_{j}}\right)^{1 / p} \leq\left(\sum_{i j k}\left(\mathbf{D}_{i j}+\mathbf{D}_{j k}\right)^{p} \frac{\mathbf{P}_{i j} \mathbf{Q}_{j k}}{\tilde{\mathbf{b}}_{j}}\right)^{1 / p}  &(\text{ triangle inequality } )
\\
& \leq\left(\sum_{i j k} \mathbf{D}_{i j}^{p} \frac{\mathbf{P}_{i j} \mathbf{Q}_{j k}}{\tilde{\mathbf{b}}_{j}}\right)^{1 / p}+\left(\sum_{i j k} \mathbf{D}_{j k}^{p} \frac{\mathbf{P}_{i j} \mathbf{Q}_{j k}}{\tilde{\mathbf{b}}_{j}}\right)^{1 / p}  &(\text{  Minkowski’s inequality })
\\
& =\mathrm{W}_{p}(\mathbf{a}, \mathbf{b})+\mathbf{W}_{p}(\mathbf{b}, c)
\end{aligned}
$$

因此结论得证，值得注意的是，当 $$0<p \leq 1$$ 时，$$\mathrm{D}^{p}$$ 是距离，但是 $$\mathbf{W}_{p}(\mathbf{a}, \mathbf{b})$$ 由于三角不等式不成立？导致不是距离，因此要求 $$p \geq 1$$ ，$$\mathbf{W}_{p}(\mathbf{a}, \mathbf{b})$$ 是距离

### Properties of Optimal Transport Distance

而当 $$\mathbf{W}_{p}(\mathbf{a}, \mathbf{b})$$ 推广到任意的测度（连续）是类似的定义方法，或者说是在一个框架下做定义的，这个启发我们是否能够通过 $$\mathbf{W}_{p}(\mathbf{a}, \mathbf{b})$$ 来定义连续测度和离散测度之间的距离？实际上 $$\mathbf{W}_{p}(\mathbf{a}, \mathbf{b})$$ 确实可以这样做，下面举两个体现最优传输距离比较好的例子，并说明相关的性质

#### Weak Convergence

对于传统的距离定义，比如 $$L^{2}$$ 距离，要求两个测度的支撑集一致 $$\operatorname{supp}(\alpha) = \operatorname{supp}(\beta)$$ ，最优传输定义的距离不要求这么严格，进一步的，就算 $$\operatorname{supp}(\alpha) = \operatorname{supp}(\beta)$$ 的情况下，强行扩张定义域，使得积分可积，也有可以体现最优传输距离比较好的例子：

研究离散均匀分布 $$\alpha$$， 定义在 $$\{0,1 / N, 2 / N, \ldots, 1\}$$ 上：

$$
\alpha=\sum_{i=1}^{N} \frac{1}{N} \delta x_{i} \quad \operatorname{supp}(\alpha)=\{0,1 / N,..., 1\}
$$

和连续均匀分布 $$\beta=U(0,1)$$，从实践的角度上来说，当 $$N \rightarrow \infty$$ 时，两个分布几乎是一样的：

{:refdef: style="text-align: center;"}
<img src="/images/2021-02-02-Computational-Optimal-Transport-Note-Part-1/Uniform-Distribution-Types.png" alt="Uniform-Distribution-Types" style="zoom:100%;" />
{:refdef}

但是取 $$L^2$$ 距离的话， $$L^2(\alpha,\beta)=1$$，对比最优传输距离为 $$\mathcal{W}_{p}(\alpha,\beta)=1/N$$，非常好的体现了两个分布渐进一致的性质，或者说，当 $$N \rightarrow \infty$$ 时有， $$ \alpha \rightarrow \beta$$ （弱收敛）

$$\mathcal{W}_{p}$$ 是弱收敛的距离，定义弱收敛：在紧空间中 $$\mathcal{X}$$，有一列 $$\left(\alpha_{k}\right)_{k}$$ 弱收敛到 $$\alpha \in \mathcal{M}_{+}^{1}(\mathcal{X})$$ $$\Leftrightarrow$$ $$\forall g \in \mathcal{C}(\mathcal{X}), \int_{\mathcal{X}} g \mathrm{~d} \alpha_{k} \rightarrow \int_{\mathcal{X}} g \mathrm{~d} \alpha$$

 那么最优传输距离是能够体现这样的性质的

#### Translations

最优传输距离，可以分离出 Translation 算子的影响，即把空间中的点沿一个向量变换 $$\tau $$ ：

$$
T_{\tau}: x \mapsto x-\tau
$$

则进一步可以利用 $$T_{\tau}$$ 去定义前向算子变换分布 $$T_{\tau \#} \alpha$$，发现变换的效果可以分离出来：

$$
\mathcal{W}_{2}\left(T_{\tau {\#}} \alpha, T_{\tau {\#}} \beta\right)^{2}=\mathcal{W}_{2}(\alpha, \beta)^{2}-2\left\langle\tau-\tau^{\prime}, \mathbf{m}_{\alpha}-\mathbf{m}_{\beta}\right\rangle+\left\|\tau-\tau^{\prime}\right\|^{2}
$$

这里的 $$\mathbf{m}_{\alpha} \stackrel{\text { def. }}{=} \int_{\mathcal{X}} x \mathrm{~d} \alpha(x) \in \mathbb{R}^{d}$$ 是分布 $$\alpha$$ 的期望，那么特别的有：

$$
\mathcal{W}_{2}(\alpha, \beta)^{2}=\mathcal{W}_{2}(\tilde{\alpha}, \tilde{\beta})^{2}+\left\|\mathbf{m}_{\alpha}-\mathbf{m}_{\beta}\right\|^{2}
$$

这里的 $$(\tilde{\alpha}, \tilde{\beta})$$ 是期望为原点的分布，计算上的性质很好

## Dual Problem

康托洛维奇问题是通过定义对偶问题：

$$
\mathrm{L}_{\mathbf{C}}(\mathbf{a}, \mathbf{b})=\max _{(\mathbf{f}, \mathbf{g}) \in \mathbf{R}(\mathbf{C})}\langle\mathbf{f}, \mathbf{a}\rangle+\langle\mathbf{g}, \mathbf{b}\rangle
\\
\mathbf{R}(\mathbf{C}) \stackrel{\text { def }}{=}\left\{(\mathbf{f}, \mathbf{g}) \in \mathbb{R}^{n} \times \mathbb{R}^{m}: \forall(i, j) \in [ n ] \times [ n ], \mathbf{f} \oplus \mathbf{g} \leq \mathbf{C}\right\} .
$$

康托洛维奇对偶性可以由拉格朗日对偶性得出，通过拉格朗日对偶性转化 $$\mathbf{R}(\mathbf{C}) $$ ，下面证明这仍然是原问题：

$$
\min _{\mathbf{P} \geq 0} \max _{(\mathbf{f}, \mathbf{g}) \in \mathbb{R}^{n} \times \mathbb{R}^{m}}\langle\mathbf{C}, \mathbf{P}\rangle+\left\langle\mathbf{a}-\mathbf{P} \mathbb{1}_{m}, \mathbf{f}\right\rangle+\left\langle\mathbf{b}-\mathbf{P}^{\mathrm{T}} \mathbb{1}_{n}, \mathbf{g}\right\rangle
$$

当考虑有限维的线性规划时，可以转换 $$\min\max$$ 为 $$\max\min$$ 有：

$$
\max _{(\mathbf{f}, \mathbf{g}) \in \mathbb{R}^{n} \times \mathbb{R}^{m}}\langle\mathbf{a}, \mathbf{f}\rangle+\langle\mathbf{b}, \mathbf{g}\rangle+\min _{\mathbf{P} \geq 0}\left\langle\mathbf{C}-\mathbf{f} \mathbb{1}_{m}{ }^{\mathrm{T}}-\mathbb{1}_{n} \mathbf{g}^{\mathrm{T}}, \mathbf{P}\right\rangle
$$

由于是取 $$\min$$ ，可以知：

$$
\min _{\mathbf{P} \geq 0}\langle\mathbf{Q}, \mathbf{P}\rangle=\left\{\begin{array}{ll}
0 & \text { if } \quad \mathbf{Q} \geq 0 \\
-\infty & \text { otherwise }
\end{array}\right.
$$

因此这个条件等价于 $$\mathbf{C}-\mathbf{f} \mathbb{1}_{m}{ }^{\mathrm{T}}-\mathbb{1}_{n} \mathbf{g}^{\mathrm{T}}=\mathbf{C}-\mathbf{f} \oplus \mathbf{g} \geq 0$$，这仍然是康托洛维奇对偶性问题，通过拉格朗日对偶性知到康托洛维奇最优传输计划的支撑集在正好达成对偶性的集合中：

$$
\left\{(i, j) \in  [n] \times [m]: \mathbf{P}_{i, j}>0\right\} 
\subset
\left\{(i, j) \in [n] \times [m]: \mathbf{f}_{i}+\mathbf{g}_{j}=\mathbf{C}_{i, j}\right\}
$$

### Intuitive Presentation

康托洛维奇对偶性，也可以由一个例子来把握一下，由之前的在最优传输笔记中所写的一样，康托洛维奇形式的原问题可以入如是理解：最小的代价去把矿上的资源推到各地的工厂中，问题的解为最优传输计划 $$\mathrm{P}^{\star}$$，最优传输代价为 $$\left\langle\mathbf{P}^{\star}, \mathbf{C}\right\rangle$$

当然也有存在这样的情况：矿主不是自己运的，（因为他不会算最优传输方案这样的理由）而是委托给别人运（当然假设这个世界中，自己运的实际代价不小于别人运的实际代价），转运商的方案是在矿$$i$$ 收一笔装载费用 $$\mathbf{f}_{i}$$ ，而在工厂 $$j$$ 收一笔卸货费 $$\mathbf{g}_{j}$$，而同时有两个分布 $$\mathbf{a}$$ 和 $$\mathbf{b}$$ 代表需要装走的矿物的分布和需要接受的工厂量的分布，则转运方案的费用是 $$\langle\mathbf{f}, \mathbf{a}\rangle+\langle\mathbf{g}, \mathbf{b}\rangle$$ 

当然，转运商给出的方案一定会使自己牟利最大，因此可以知道转运商的方案的成本下界就是最优传输方案，矿主自己需要把运输价钱给压低，最好的办法是算出原问题的最优传输代价 $$\mathrm{L}_{\mathbf{C}}(\mathbf{a}, \mathbf{b})$$，当然出于某些技术性问题，他不能算出具体的方案，此时有替代方案 :

比较 $$\mathrm{C}_{i, j}$$ 为矿主自己运 $$i \rightarrow j$$ 的代价，转运商的方案中对应为 $$\mathbf{f}_{i}+\mathbf{g}_{j}$$ ，若出现了 $$\mathbf{f}_{i}+\mathbf{g}_{j} > \mathrm{C}_{i, j}$$ 的情况，则这部分就是有压价空间的

换一个角度，已知矿主的压价方案，出于转运商的角度，首先他自己不会给出超过矿主自己运还贵的方案去做这件事情，对于任意传输方案 $$\mathbf{P}$$，有边际分布 $$\mathbf{a},\mathbf{b}$$，自然有约束：

$$
\begin{aligned}
\sum_{i, j} \mathbf{P}_{i, j} \mathbf{C}_{i, j} & \geq \sum_{i, j} \mathbf{P}_{i, j}\left(\mathbf{f}_{i}+\mathbf{g}_{j}\right)=\left(\sum_{i} \mathbf{f}_{i} \sum_{j} \mathbf{P}_{i, j}\right)+\left(\sum_{j} \mathbf{g}_{j} \sum_{i} \mathbf{P}_{i, j}\right) \\
&=\langle\mathbf{f}, \mathbf{a}\rangle+\langle\mathbf{g}, \mathbf{b}\rangle
\end{aligned}
$$

同时在满足这样的方案 $$\mathbf{f},\mathbf{g}$$ 中，为了不给矿主压价空间，满足 $$\mathbf{f}_{i}+\mathbf{g}_{j} \leq \mathbf{C}_{i, j}$$，即康托洛维奇形式的约束 $$ \mathbf{f} \oplus \mathbf{g} \leq \mathbf{C}$$

下图可视化的展示了康托洛维奇形式原问题和对偶问题的关系：

{:refdef: style="text-align: center;"}
<img src="/images/2021-02-02-Computational-Optimal-Transport-Note-Part-1/Primal_Dual.png" alt="Primal_Dual" style="zoom:40%;" />
{:refdef}

进一步的，对偶问题也可以推广到任意的测度上：

$$
\mathcal{L}_{c}(\alpha, \beta)=\sup _{(f, g) \in \mathcal{R}(c)} \int_{\mathcal{X}} f(x) \mathrm{d} \alpha(x)+\int_{\mathcal{Y}} g(y) \mathrm{d} \beta(y)
\\
\mathcal{R}(c) \stackrel{\text { def. }}{=}\{(f, g) \in \mathcal{C}(\mathcal{X}) \times \mathcal{C}(\mathcal{Y}): \forall(x, y), f(x)+g(y) \leq c(x, y)\}
$$

同时依然有最优传输的支撑集在对偶问题可行集和原问题可行集中交集中：

$$
\operatorname{Supp}(\pi) \subset\{(x, y) \in \mathcal{X} \times \mathcal{Y}: f(x)+g(y)=c(x, y)\}
$$

上面的有约束情况，在 $$\alpha,\beta$$ 构成分布，即：$$\int_{\mathcal{X}} \mathrm{d} \alpha=\int_{\mathcal{Y}} \mathrm{d} \beta=1$$，可以转化成无约束形式：

$$
\mathcal{L}_{c}(\alpha, \beta)=\sup _{(f, g) \in \mathcal{C}(\mathcal{X}) \times \mathcal{C}(\mathcal{Y})} \int_{\mathcal{X}} f \mathrm{~d} \alpha+\int_{\mathcal{Y}} g \mathrm{~d} \beta+\min _{\mathcal{X} \otimes \mathcal{Y}}(c-f \oplus g)
$$

## Brenier Theorem

康托洛维奇形式和蒙日形式等价性是非常重要的，至少到现在，证明其等价性需要证明在某些情况下，蒙日形式给出了康托洛维奇形式的一个下界，这其实很难做到。Brenier Theorem 给出条件：在 $$R^{d}$$ 中，传输距离定义为 $$p=2$$ 的情况，两个测度中至少测度 $$\alpha$$ 有一个能定义 $$\rho_{\alpha}$$，且有二阶矩，则在这样的条件下，康托洛维奇形式和蒙日形式是等价的

具体一些的话，在 $$\mathcal{X}=\mathcal{Y}=\mathbb{R}^{d}$$ 中有代价函数 $$c(x, y)=\|x-y\|^{2}$$，其中至少一个测度 $$\alpha$$ 有一个能定义 $$\rho_{\alpha}$$，且有二阶矩 康托洛维奇最优传输是唯一的，且其支撑集和蒙日传输的支撑集是一样的，为 $$\{(x, T(x)), T: \mathbb{R}^{d} \rightarrow \mathbb{R}^{d} \} $$，而康托洛维奇传输计划可以表示为 $$\pi=(\mathrm{Id}, T)_{\#}\alpha $$，比如：

$$
\forall h \in \mathcal{C}(\mathcal{X} \times \mathcal{Y}), \quad \int_{\mathcal{X} \times \mathcal{Y}} h(x, y) \mathrm{d} \pi(x, y)=\int_{\mathcal{X}} h(x, T(x)) \mathrm{d} \alpha(x)
$$

进一步的，还可以给出蒙日传输映射 $$T$$ 的形式，通过一个凸函数 $$\varphi(x)$$ 定义，为其梯度 $$T(x)=\nabla \varphi(x)$$，而 $$\varphi(x)$$ 由之前康托洛维奇对偶形式中的 $$f$$ 给出：$$\varphi(x)=\frac{\|x\|^{2}}{2}-f(x)$$ 。简要的过一遍证明的主体：

康托洛维奇传输代价有等价形式：$$\int c \mathrm{~d} \pi=C_{\alpha, \beta}-2 \int\langle x, y\rangle \mathrm{d} \pi(x, y)$$，而由于测度 $$\alpha,\beta$$ 已知，$$C_{\alpha, \beta}=\int\|x\|^{2} \mathrm{~d} \alpha(x)+\int\|y\|^{2} \mathrm{~d} \beta(y)$$ 这部分为常数，问题就等价于去求解：

$$
\max _{\pi \in \mathcal{U}(\alpha, \beta)} \int_{\mathcal{X} \times \mathcal{Y}}\langle x, y\rangle \mathrm{d} \pi(x, y)
$$

利用对偶性可以写出：

$$
\min _{(\varphi, \psi)}\left\{\int_{\mathcal{X}} \varphi \mathrm{d} \alpha+\int_{\mathcal{Y}} \psi \mathrm{d} \beta: \forall(x, y), \quad \varphi(x)+\psi(y) \geq\langle x, y\rangle\right\}
$$

可以知道 $$(\varphi, \psi)$$ 和 $$(f, g)$$ 的关系满足 $$(\varphi, \psi)=\left(\frac{\|\cdot\|^{2}}{2}-f, \frac{\|\cdot\|^{2}}{2}-g \right) $$

利用勒让德变换可知：$$\varphi^{*}(y) \stackrel{\text { def. }}{=} \sup _{x}\langle x, y\rangle-\varphi(x)$$，利用勒让德变换，定义约束 $$\forall y, \quad \psi(y) \geq \varphi^{*}(y) \stackrel{\text { def. }}{=} \sup _{x}\langle x, y\rangle-\varphi(x)$$。令 $$\psi=\varphi^{*}$$，则可以变换优化目标为：

$$
\min _{\varphi} \int_{\mathcal{X}} \varphi \mathrm{d} \alpha+\int_{\mathcal{Y}} \varphi^{*} \mathrm{~d} \beta
$$

可知 $$\varphi^{*}$$ 是凸的，那么上述的变换再做一次的话，则有 $$\varphi = \varphi^{* *}$$，那么在这样的约束条件下，知道 $$ \left\{(x, y): \varphi(x)+\varphi^{*}(y)=\langle x, y\rangle\right\}$$ 是凸的，上述集合是最优传输 $$\pi$$ 的支撑集，同时也是勒让德变换的最优解，因此 $$y \in \partial \varphi(x)$$

由于 $$\varphi $$ 是凸的，则几乎处处可微，由于 $$\alpha$$ 是一个密度，所以它也几乎处处可微，则可以定义最优传输 $$\pi=(\mathrm{Id}, \nabla \varphi)_{\#} \alpha$$

## Reference

1. Gabriel Peyre ,Marco Cuturi "[Computational Optimal Transport](https://arxiv.org/abs/1803.00567), Foundation and Trends"




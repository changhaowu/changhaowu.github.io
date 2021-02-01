---
layout: post
title: "Optimal Transport Note:Part 1"
date: 2021-01-20 
image: /images/cover/C_Scenery4.png       
tags: [Generative-Model]
toc: true
---

# Optimal Transport Note: Part 1

## Formulation of Optimal Transport

### Monge Formulation

最优传输的背景是蒙日考虑在建造防御工事时，如何花费最少的劳动力去把四散的土堆运输到其他处的防御工事处，在此之上抽象出了最优传输问题，最优传输问题总共有两种提法，蒙日形式（Monge Formulation）和康托洛维奇形式（Kantorovich Formulation），康托洛维奇形式更加完善，更加适合理论研究，蒙日形式则更加适合应用上的计算。

先给出蒙日形式的最优传输问题：

有概率空间 $$ (X, \Sigma_X , \mu) $$ 和 $$ (Y, \Sigma_Y , \nu) $$ 

定义代价函数 $$c: X \times Y \rightarrow[0,+\infty]$$，测量运输 $$x \in X $$ 到 $$ y \in Y$$ 的代价

定义传输映射 $$T: X \rightarrow Y$$ 将 $$u \in \mathcal{P}(X) $$ 传输到 $$  \nu \in \mathcal{P}(Y)$$，当

$$
\nu(B)=\mu\left(T^{-1}(B)\right) \quad  \forall \; \nu \text { -measurable } B
$$

这样定义保证了传输映射必须双射，而测度不变，直观上来说，就是传输映射 $$T$$，从 $$X$$ 中取走多少土，就相应有多少土运到 $$Y$$ 中，如下图所示

{:refdef: style="text-align: center;"}
<img src="/images/2021-01-20-Optimal-Transport-Note-Part-1/Visualing_Transport_Map.png" alt="Visualing_Transport_Map" style="zoom:30%;" />
{:refdef}

在上面的定义下，称 $$T$$ 传输 $$\mu$$ 到 $$\nu$$ ，记 $$\nu=T_{\#} \mu$$

然后对于传输映射就有两条性质：

对于 $$ \mu \in \mathcal{P}(X), T: X \rightarrow Y, S: Y \rightarrow Z $$，以及 $$ f \in L^{1}(Y)$$

1. 变量变换公式：分别在原像集和像集的角度下，（$$f \equiv 1$$ 则为上面的质量不变，推广质量不变到期望不变？）

   $$
   \int_{Y} f(y) \mathrm{d}\left(T_{\#} \mu\right)(y)=\int_{X} f(T(x)) \mathrm{d} \mu(x)
   $$

2. 映射复合公式：推广到存在中转站这样的情况下而定义

   $$
   (S \circ T)_{\#} \mu=S_{\#}\left(T_{\#} \mu\right)
   $$

上面的定义都很自然又严谨，但是很可惜，由于蒙日形式下要求传输映射 $$T$$ 可逆，其不一定存在

比如在 $$x_{1}$$ 处有 $$1$$ 单位沙堆，而在 $$ y_{1},y_{2} $$ 处分别有两个 $$\frac{1}{2}$$ 单位的防御工事要建造

换言之 $$\mu=\delta_{x_{1}}$$ 而 $$\nu=\frac{1}{2} \delta_{y_{1}}+\frac{1}{2} \delta_{y_{2}}$$ ，由于$$ \nu\left(\left\{y_{1}\right\}\right)=\frac{1}{2} $$，其不可能等于 $$\mu\left(T^{-1}\left(y_{1}\right)\right) \in\{0,1\}$$，因此传输映射 $$T$$ 不存在

于是就可以定义蒙日形式的最优传输问题：$$ T: X \rightarrow Y \text { subject to } \nu=T_{\#} \mu$$

$$
\text { minimise } \mathbb{M}(T)=\int_{X} c(x, T(x)) \mathrm{d} \mu(x)
$$

###  Kantorovich Formulation

由于蒙日形式下，定义的最优传输 $$x \mapsto T(x)$$ ，由于传输映射需要保证映射的特性，或者直观上来说， $$x_{1}$$ 处的土堆不能分割，只能全部传输到另一个点 $$y_{1}$$ 处，需要更加灵活的定义

因此康托洛维奇定义了传输计划 $$\pi \in \mathcal{P}(X \times Y)$$ ，传输计划一样要服从传输质量不变性的约束，但在此之上，传输计划使得 $$x_{1}$$ 处的土堆可以运输到多个目的地 $$\{y_{1},...,y_{n}\}$$ 处，只需要满足 $$\mu({x_1})=\nu(\{y_{1},...,y_{n}\})$$ 即可

或者考虑联合分布和边际分布的概念，在概率空间 $$ (X, \Sigma_X , \mu) $$ 和 $$ (Y, \Sigma_Y , \nu) $$ 的基础上，有$$\pi \in \mathcal{P}(X \times Y)$$ ，记 $$\mathrm{d} \pi(x, y)$$ 是从 $$x$$ 传输到 $$y$$ 的质量，服从

$$
\pi(A \times Y)=\int_{A \times Y} d\pi\left(x,y\right)=\mu(A) \quad \pi(X \times B)=\int_{X \times B} d\pi\left(x,y\right)=\nu(B)
$$

记 $$\Pi(\mu, \nu)$$ 为传输方案的集合，比起传输映射 $$T$$ 可能不存在的问题，传输计划 $$\Pi(\mu, \nu)$$ 永远非空，因为有一个平凡解 $$\pi^{*}$$ ,取定 $$\{y^*\} \in \Sigma_{Y}$$ ，对应的 $$X$$ 上的起点在满足约束 $$\int_{X } d\pi\left(x,y^*\right)=\nu(y^*)$$ 对 $$ \nu(y^*)$$ 成比例取值即可，如下图所示

{:refdef: style="text-align: center;"}
<img src="/images/2021-01-20-Optimal-Transport-Note-Part-1/Trivial_Plan.png" alt="Trivial Pan" style="zoom:40%;" />
{:refdef}

定义好了传输计划后，就可以定义康托洛维奇形式的最优传输问题：$$ \mu \in \mathcal{P}(X) ,\nu \in \mathcal{P}(Y)$$

$$
\text { minimise } \mathbb{K}(\pi)=\int_{\mathrm{X} \times Y} c(x, y) \mathrm{d} \pi(x, y) 
\quad
\text{subject to} \quad \pi \in \Pi(\mu, \nu)
$$

下面证明蒙日形式与康托洛维奇形式的关系：假设蒙日形式最优存在，$$T^{\dagger}: X \rightarrow Y$$，定义 $$d \pi(x, y)=\mathrm{d} \mu(x) \delta_{y=T^{\dagger}(x)}$$ 

$$
\begin{array}{l}
\pi(A \times Y)=\int_{A} \delta_{T^{\dagger}(x) \in Y} \mathrm{~d} \mu(x)=\mu(A) \\
\pi(X \times B)=\int_{X} \delta_{T^{\dagger}(x) \in B} \mathrm{~d} \mu(x)=\mu\left(\left(T^{\dagger}\right)^{-1}(B)\right)=T_{\#}^{\dagger} \mu(B)=\nu(B)
\end{array}
$$

于是 $$\pi \in \Pi(\mu, \nu)$$ ，

$$
\int_{X \times Y} c(x, y) \mathrm{d} \pi(x, y)=\int_{X} \int_{Y} c(x, y) \delta_{y=T^{\dagger}(x)} \mathrm{d} y \mathrm{~d} \mu(x)=\int_{X} c\left(x, T^{\dagger}(x)\right) \mathrm{d} \mu(x)
$$

于是有

$$
\inf \mathbb{K}(\pi) \leq \inf \mathbb{M}(T)
$$

而当传输计划与传输映射等价的时候，即 $$d \pi^{\dagger}(x, y)=\mathrm{d} \mu(x) \delta_{y=T^{\dagger}(x)}$$ 时，此时有 $$\inf \mathbb{K}(\pi) = \inf \mathbb{M}(T)$$ ，此时蒙日形式与康托洛维奇形式是等价的

最优传输的一个应用是，利用最优传输的插值：

$$
\begin{aligned}
\mu_{t}&=\left((1-t) \mathrm{Id}+t T^{\dagger}\right)_{\#} \mu
\\
\mu_{0}(B)&=\left(\mathrm{Id}\right)_{\#} \mu(B)=\mu(\mathrm{Id}^{-1}(B))=\mu(B) 
\\
\mu_{1}(B)&=\mu_{1}\left(T^{\dagger-1}(B)\right)=\nu(B)
\end{aligned}
$$

 其效果会比单纯的在欧氏空间中插值：

$$
\mu_{t}^{E}=(1-t) \mu+t \nu
$$

在可视化后的效果上更好一些：

{:refdef: style="text-align: center;"}
<img src="/images/2021-01-20-Optimal-Transport-Note-Part-1/OT_interpolation.png" alt="OT_interpolation" style="zoom:40%;" />
{:refdef}

## Special Cases

一般意义下的最优传输问题，还需要康托洛维奇对偶性等工具，但是在此之前，有两种特殊的情况，不用对偶性就可以解决，于是先摘了这些“低垂的果实” X:)

### Optimal Transport in One Dimension

在一维情况下，有概率空间 $$ (X, \Sigma_X , \mu) $$ 和 $$ (Y, \Sigma_Y , \nu) $$ 下，进而利用 $$\mu,\nu$$  可以定义右连续，不减的 $$c.d.f $$ $$F(x),G(y)$$ ，有性质：

$$
F(x)=\int_{-\infty}^{x} \mathrm{~d} \mu=\mu((-\infty, x])
\\
F(-\infty)=0 \quad F(+\infty)=1
\\
$$

同时，可以定义广义逆 $$F^{-1}$$ 

$$
F^{-1}(t)=\inf \{x \in \mathbb{R}: F(x)>t\}
\\
F^{-1}(F(x)) \geq x \quad F\left(F^{-1}(t)\right) \geq t
$$

进一步当 $$F$$ 可逆时

$$
F^{-1}(F(x))=x \quad F\left(F^{-1}(t)\right)=t
$$

以上的定义，对于 $$\nu$$ 来说，也是一样的再做一遍

然后就有了 Theorem 2.1

####  Theorem  2.1 

$$\mu, \nu \in \mathcal{P}(\mathbb{R})$$ 其 $$c.d.f$$ 分别是 $$F,G$$， 认为 $$c(x, y)=d(x-y)$$ 是凸的且连续的，$$\pi^{\dagger} \in  \mathcal{P}(\mathbb{R}^{2})$$ 且有 $$c.d.f \quad H(x, y)=\min \{F(x), G(y)\}$$ ，则 $$\pi^{\dagger} \in \Pi(\mu, \nu)$$ 且 $$\pi^{\dagger} $$ 是康托洛维奇形式最优传输问题的解，且在代价函数 $$c(x,y)$$ 下的传输代价为

$$
\min _{\pi \in \Pi(\mu, \nu)} \mathbb{K}(\pi)=\int_{0}^{1} d\left(F^{-1}(t)-G^{-1}(t)\right) \mathrm{d} t
$$

#### Corollary 2.2

1. 当 $$ c(x, y)=   \lvert x-y  \rvert  $$ ，则最优传输代价也等于两个 $$ c.d.f $$ 的 $$ L^1 $$ 距离：

   $$
   \inf _{\pi \in \Pi(\mu, \nu)} \mathbb{K}(\pi)=\int_{\mathbb{R}}|F(x)-G(x)| \mathrm{d} 
   $$

   {:refdef: style="text-align: center;"}
   <img src="/images/2021-01-20-Optimal-Transport-Note-Part-1/abs_cost_equalivence.png" alt="abs_cost_equalivence" style="zoom:40%;" />
   {:refdef}

   如图所示，描述积分区域可以用两种方法：

   $$
   \begin{aligned}
   \mathcal{A} &=\left\{(x, t): \min \left\{F^{-1}(t), G^{-1}(t)\right\} \leq x \leq \max \left\{F^{-1}(t), G^{-1}(t)\right\}, t \in[0,1]\right\} \\ 
   &= \{(x, t): \min \{F(x), G(x)\} \leq t \leq \max \{F(x), G(x)\}, x \in \mathbb{R}\}
   \end{aligned}
   $$

   且 $$ \max \{a, b\}-\min \{a, b\}= \lvert a-b  \rvert$$ 即为代价函数即可证明

2. 若传输计划等价于传输映射 $$\min _{\pi \in \Pi(\mu, \nu)} \mathbb{K}(\pi)=\min _{T: T_{\#} \mu=\nu} \mathbb{M}(T)$$ ，则 $$T^{\dagger}=G^{-1} \circ F$$ 是蒙日形式的最优传输映射：

   $$
   \inf _{T: T_{\#} \mu=\nu} \mathbb{M }(T)=\mathbb{M}\left(T^{\dagger}\right)
   $$

   1. 第一部分证明 $$T^{\dagger}_{\#} \mu=\nu$$ ：

      利用之前的复合映射公式，知$$T^{\dagger}_{\#} \mu =G_{\#}^{-1}\left(F_{\#} \mu\right)$$ ，由于 $$F$$ 连续 ，$$\exists x_t,\forall t \in (0,1)  ,F\left(x_{t}\right)=t $$  ，于是对于 $$F_{\#} \mu$$ 有：
      
      $$
      \begin{aligned}
      F_{\#} \mu([0, t]) &=\mu(\{x: F(x) \leq t\}) \\
      &=\mu\left(\left\{x: x \leq x_{t}\right\}\right) \\
      &=F\left(x_{t}\right) \\
      &=t
      \\
      &\Rightarrow F_{\#} \mu=\mathcal{L}_{[0,1]}
      \end{aligned}
      $$
      
      于是问题变成证明 $$T^{\dagger}_{\#} \mu =G_{\#}^{-1}\left(\mathcal{L}_{[0,1]}\right)$$ 
      
      $$
      \begin{aligned}
      G_{\#}^{-1} \mathcal{L}\left\lfloor_{[0,1]}((-\infty, y])\right.&=\mathcal{L}\left\lfloor_{[0,1]}\left(\left\{t: G^{-1}(t) \leq y\right\}\right)\right.\\
      &=\mathcal{L}\left\lfloor_{[0,1]}(\{t: G(y) \geq t\})\right.\\
      &=G(y) \\
      &=\nu((-\infty, y])
      \\
      &\Rightarrow T^{\dagger}_{\#} \mu =G_{\#}^{-1}\left(F_{\#} \mu\right)
      \end{aligned}
      $$

   2. 第二部分证明 $$T^{\dagger}$$ 是蒙日形式的最优传输，利用之前的质量不变公式和 $$F_{\#} \mu=\mathcal{L}_{[0,1]}$$
   
      $$
      \begin{aligned}
      \inf _{\pi \in \Pi(\mu, \nu)} \mathbb{K}(\pi) &=\int_{0}^{1} d\left(F^{-1}(t)-G^{-1}(t)\right) \mathrm{d} t \\
      &=\int_{\mathbb{R}} d\left(x-G^{-1}(F(x))\right) \mathrm{d} \mu(x) \\
      &=\int_{\mathbb{R}} d\left(x-T^{\dagger}(x)\right) \mathrm{d} \mu(x) \\
      & \geq \inf _{T: T_{\#} \mu=\nu} \mathbb{M}(T)
      \end{aligned}
      $$
      
      同时 $$\inf _{T: T_{\#} \mu=\nu} \mathbb{M}(T) \geq \min _{\pi \in \Pi(\mu, \nu)} \mathbb{K}(\pi)$$ ，因此  $$T^{\dagger}=G^{-1} \circ F$$ 是蒙日最优传输

#### Proposition 2.3

定义一个很重要的性质，集合的单调性（这是对于某个测度 $$d$$ ），由简单的一个二维情况做例子：

对于 $$\Gamma \subset \mathbb{R}^{2}$$，$$\forall \left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right) \in \Gamma$$ , $$\Gamma$$ 是单调的当：

$$
d\left(x_{1}-y_{1}\right)+d\left(x_{2}-y_{2}\right) \leq d\left(x_{1}-y_{2}\right)+d\left(x_{2}-y_{1}\right)
$$

然后是 Proposition 2.3 的内容：

有 $$\mu, \nu \in \mathcal{P}(\mathbb{R})$$，假设在代价函数 $$c(x, y)=d(x-y)$$ 意义下的最优传输计划 $$\pi^{\dagger} \in \Pi(\mu, \nu)$$，对于任何支撑集中的点 $$\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right) \in \operatorname{supp}\left(\pi^{\dagger}\right)$$，有

$$
d\left(x_{1}-y_{1}\right)+d\left(x_{2}-y_{2}\right) \leq d\left(x_{1}-y_{2}\right)+d\left(x_{2}-y_{1}\right)
$$

利用反证法证明，若能在支撑集中构造一个 $$\pi^{\dagger}$$ 的下界 $$\pi^{*}$$ ，且证明 $$\pi^{*}  \in \Pi(\mu, \nu)$$

假设在支撑集中存在不单调的点 $$\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right) \in \operatorname{supp}\left(\pi^{\dagger}\right)$$，那么就有：

$$
d\left(x_{1}-y_{1}\right)+d\left(x_{2}-y_{2}\right)-d\left(x_{1}-y_{2}\right)-d\left(x_{2}-y_{1}\right) \geq \eta
$$

然后在 $$ (X, \Sigma_X , \mu) $$ 和 $$ (Y, \Sigma_Y , \nu) $$ 上分别构造两个闭区间 $$I_{1}, I_{2}, J_{1}, J_{2}$$，且满足：

1. $$x_{i} \in I_{i}, y_{i} \in J_{i}, i=1,2$$
2. $$\forall x \in I_{i}, y \in J_{j}, i, j=1,2,d(x-y) \geq d\left(x_{i}-y_{j}\right)-\varepsilon$$ ，且 $$\varepsilon<\frac{\eta}{4}$$
3. $I_{i} \times J_{j}$ 不相交;
4. $\pi^{\dagger}\left(I_{1} \times J_{1}\right)=\pi^{\dagger}\left(I_{2} \times J_{2}\right)=\delta>0$

这样的闭区间可以通过取的很小构造出，利用这些小区间去构造比 $$\pi^{\dagger}$$ 小的测度 $$\tilde{\pi}$$，而在小区间之外的部分，则利用原先的最优传输 $$\pi^{\dagger}$$ ，因此还需要定义 $$\pi^{\dagger}$$ 的投影：

$$
\begin{array}{ll}
\tilde{\mu}_{1}=P_{\#}^{X} \pi^{\dagger}\left\lfloor I_{1} \times J_{1},\right. & \tilde{\mu}_{2}=P_{\#}^{X} \pi^{\dagger}\left\lfloor I_{2} \times J_{2}\right. \\
\tilde{\nu}_{1}=P_{\#}^{Y} \pi^{\dagger}\left\lfloor_{I_{1} \times J_{1}},\right. & \tilde{\nu}_{2}=P_{\#}^{Y} \pi^{\dagger}\left\lfloor I_{2} \times J_{2}\right.
\end{array}
$$

于是利用最优传输投影 $$\pi^{\dagger}$$ 的测度，构造 $$\tilde{\pi}_{12} \in \Pi\left(\tilde{\mu}_{1}, \tilde{\nu}_{2}\right), \tilde{\pi}_{21} \in \Pi\left(\tilde{\mu}_{2}, \tilde{\nu}_{1}\right)$$，并定义：

$$
\tilde{\pi}(A \times B)=\left\{\begin{array}{ll}
\pi^{\dagger}(A \times B) & \text { if }(A \times B) \cap\left(I_{i} \times J_{j}\right)=\emptyset \text { for all } i, j \\
0 & \text { if } A \times B \subseteq I_{i} \times J_{i} \text { for some } i \\
\pi^{\dagger}(A \times B)+\tilde{\pi}_{12}(A \times B) & \text { if } A \times B \subseteq I_{1} \times J_{2} \\
\pi^{\dagger}(A \times B)+\tilde{\pi}_{21}(A \times B) & \text { if } A \times B \subseteq I_{2} \times J_{1}
\end{array}\right.
$$

$$\tilde{\pi}$$ 把小区间的测度挖掉，对于 $$(A \times B) \cap\left(I_{i} \times J_{j}\right) \neq \emptyset$$ 且 $$A \times B \nsubseteq\left(I_{i} \times J_{j}\right)$$ 的情况，利用补集定义：

$$
\tilde{\pi}(A \times B)=\tilde{\pi}\left((A \times B) \cap\left(I_{i} \times J_{j}\right)\right)+\tilde{\pi}\left((A \times B) \cap\left(I_{i} \times J_{j}\right)^{c}\right)
$$

下验证 $$\tilde{\pi} \in \Pi(\mu, \nu)$$，取 $$\tilde{\pi}(\mathbb{R} \times B)$$ 研究：

1. 当 $$B \cap\left(J_{1} \cup J_{2}\right)=\emptyset$$
   
   $$
   \tilde{\pi}(\mathbb{R} \times B)=\pi^{\dagger}(\mathbb{R} \times B)=\nu(B)
   $$

2. 当 $$B \subseteq J_{1}$$
   
   $$
   \begin{aligned}
   \tilde{\pi}(\mathbb{R} \times B) &=\tilde{\pi}\left(\left(\mathbb{R} \backslash\left(I_{1} \cup I_{2}\right)\right) \times B\right)+\tilde{\pi}\left(I_{1} \times B\right)+\tilde{\pi}\left(I_{2} \times B\right) \\
   &=\pi^{\dagger}\left(\left(\mathbb{R} \backslash\left(I_{1} \cup I_{2}\right)\right) \times B\right)+0+\pi^{\dagger}\left(I_{2} \times B\right)+\tilde{\pi}_{21}\left(I_{2} \times B\right) 
   \\
   & \; \Big\Downarrow \; \tilde{\pi}_{21}\left(I_{2} \times B\right)=\tilde{\nu}_{1}(B)=\pi^{\dagger}\left(I_{1} \times\left(B \cap J_{1}\right)\right)=\pi^{\dagger}\left(I_{1} \times B\right)
   \\
   &=\pi^{\dagger}\left(\left(\mathbb{R} \backslash I_{1}\right) \times B\right)+\pi^{\dagger}\left(I_{1} \times B\right) \\
   &=\pi^{\dagger}(\mathbb{R} \times B) \\
   &=\nu(B)
   \end{aligned}
   $$
   
   对于 $$B \subseteq J_{2}$$ 是一样的，有 $$\tilde{\pi}(\mathbb{R} \times B)=\nu(B)$$，同理有 $$\tilde{\pi}(A \times \mathbb{R})=\mu(A)$$，推出 $$\tilde{\pi} \in \Pi(\mu, \nu)$$

下证 $$\tilde{\pi}$$ 是最优传输：

$$
\begin{aligned}
& \int_{\mathbb{R} \times \mathbb{R}} d(x-y) \mathrm{d} \pi^{\dagger}(x, y)-\int_{\mathbb{R} \times \mathbb{R}} d(x-y) \mathrm{d} \tilde{\pi}(x, y) 
\\
&=\int_{I_{1} \times J_{1} \cup I_{2} \times J_{2}} d(x-y) \mathrm{d} \pi^{\dagger}(x, y)-\int_{I_{1} \times J_{2}} d(x-y) \mathrm{d} \tilde{\pi}_{12}(x, y) 
-\int_{I_{2} \times J_{1}} d(x-y) \mathrm{d} \tilde{\pi}_{21}(x, y) 
\\
& \geq \delta\left(d\left(x_{1}-y_{1}\right)-\varepsilon\right)+\delta\left(d\left(x_{2}-y_{2}\right)-\varepsilon\right)-\delta\left(d\left(x_{1}-y_{2}\right)+\varepsilon\right)-\delta\left(d\left(x_{2}-y_{1}\right)+\varepsilon\right) 
\\
&\geq \delta(\eta-4 \varepsilon) 
\\
&>0
\end{aligned}
$$

与 $$\pi^{\dagger}$$ 是最优传输矛盾，因此知道 $$\operatorname{supp}\left(\pi^{\dagger}\right)$$ 在传输代价 $$d$$ 的意义下单调

#### Proof of Theorem 2.1

有传输代价函数 $$d$$ 严格凸，连续，由康托洛维奇传输问题最优解的存在性，知存在 $$\pi^{*} \in \prod(\mu, \nu)$$ 为最优传输计划，下证 $$\pi^{*}=\pi^{\dagger} $$

1.  $$\operatorname{supp}\left(\pi^{\dagger}\right)$$ 在传输代价 $$d$$ 的意义下单调，由传输代价函数 $$d$$ 严格凸， $$\operatorname{supp}\left(\pi^{\dagger}\right)$$ 就有更强的性质：
   
   $$
   \forall \left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right) \in \Gamma,  x_{1}<x_{2} \Rightarrow y_{1} \leq y_{2}
   $$
   
   这一点利用 $$d$$ 严格凸证明，反证法假设 $$y_{1} \leq y_{2}$$，设 $$a=x_{1}-y_{1}, b=x_{2}-y_{2},\delta=x_{2}-x_{1}$$，由支撑集单调：
   
   $$
   d(a)+d(b) \leq d(b-\delta)+d(a+\delta) 
   $$
   
   设 $$t=\frac{\delta}{b-a} \in \left( 0,1 \right) $$ ，则有 $$b-\delta=(1-t) b+t a, a+\delta=t b+(1-t) a$$，利用 Jenson 不等式知：
   
   $$
   d(b-\delta)+d(a+\delta)<(1-t) d(b)+t d(a)+t d(b)+(1-t) d(a)=d(b)+d(a)
   $$
   
   这与单调性矛盾，则支撑集有 $$\forall\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right) \in \Gamma, x_{1}<x_{2} \Rightarrow y_{1} \leq y_{2}$$

2. 利用支撑集的性质，可以证明 $$\pi^{*}((-\infty, x],(-\infty, y])=\min \{F(x), G(y)\}$$ ，即 $$\pi^{\dagger}=\pi^{*}$$：

   令 $$A=(-\infty, x] \times(y,+\infty), B=(x,+\infty) \times(-\infty, y]$$，由$$\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right) \in \Gamma , x_{1} \leq x_{2} \Rightarrow y_{1} \leq y_{2}$$，若有 $$\left(x_{0}, y_{0}\right) \in \Gamma$$，则
   
   $$
   \Gamma \subset\left\{(x, y): x \leq x_{0}, y \leq y_{0}\right\} \cup\left\{(x, y): x \geq x_{0}, y \geq y_{0}\right\}
   $$
   
   {:refdef: style="text-align: center;"}
   <img src="/images/2021-01-20-Optimal-Transport-Note-Part-1/optplan_supp.png" alt="optplan_supp" style="zoom:40%;" />
    {:refdef}
   
   如图所示，由于支撑集的性质，有 $$\pi(A) * \pi(B)=0$$，知道 $$\pi^{*}$$测度不大于：
   
   $$
   \begin{aligned}
   \pi^{*}((-\infty, x] \times(-\infty, y])=\min \{& \pi^{*}(((-\infty, x] \times(-\infty, y]) \cup A),
   \left.\pi^{*}(((-\infty, x] \times(-\infty, y]) \cup B)\right\} .
   \end{aligned}
   $$
   
   进一步的，由于
   
   $$
   \begin{array}{l}
   \pi^{*}(((-\infty, x] \times(-\infty, y]) \cup A)=\pi((-\infty, x] \times \mathbb{R})=F(x) \\
   \pi^{*}(((-\infty, x] \times(-\infty, y]) \cup B)=\pi(\mathbb{R} \times(-\infty, y])=G(y)
   \end{array}
   $$
   
   那么 $$\pi^{*}((-\infty, x] \times(-\infty, y])=\min \{F(x), G(y)\}=\pi^{\dagger}((-\infty, x] \times(-\infty, y])$$，由此可知 $$\pi^{\dagger}$$ 为康托洛维奇最优传输

3.  最后证一维最优传输的等价性，$$\int_{\mathbb{R} \times \mathbb{R}} d(x-y) \mathrm{d} \pi^{\dagger}(x, y)=\int_{0}^{1} d\left(F^{-1}(t)-G^{-1}(t)\right) \mathrm{d} t$$，也等价于 $$\pi^{\dagger}=\left(F^{-1}, G^{-1}\right)_{\#} \mathcal{L}\lfloor[0,1]$$
   
   $$
   \begin{aligned}
   \left(F^{-1}, G^{-1}\right)_{\#} \mathcal{L}\lfloor[0,1]((-\infty, x] \times(-\infty, y])&=\mathcal{L} L_{[0,1]}\left(\left(F^{-1}, G^{-1}\right)^{-1}((-\infty, x] \times(-\infty, y])\right) \\
   &=\mathcal{L}\left\lfloor_{[0,1]}\left(\left\{t: F^{-1}(t) \leq x \text { and } G^{-1}(t) \leq y\right\}\right)\right.\\
   &=\mathcal{L}\lfloor[0,1](\{t: F(x) \geq t \text { and } G(y) \geq t\})\\
   &=\min \{F(x), G(y)\} \\
   &=\pi^{\dagger}((-\infty, x] \times(-\infty, y])
   \end{aligned}
   $$
   
   进一步的，由于变量变换公式：
   
   $$
   \int_{\mathbb{R} \times \mathbb{R}} d(x-y) \mathrm{d} \pi^{\dagger}(x, y)=\int_{\mathbb{R} \times \mathbb{R}} d(x-y) \mathrm{d}\left(\left(F^{-1}, G^{-1}\right)_{\#} \mathcal{L}\right)(x, y)=\int_{0}^{1} d\left(F^{-1}(t)-G^{-1}(t)\right) \mathrm{d} t
   $$
   


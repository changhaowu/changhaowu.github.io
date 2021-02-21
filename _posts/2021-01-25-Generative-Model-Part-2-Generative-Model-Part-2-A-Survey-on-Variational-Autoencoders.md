---
layout: post
title: "Generative Model Part 2：A Survey on Variational Autoencoder"
date: 2021-01-25
image: /images/cover/C_ Abstraction3.jpeg         
tags: [Generative-Model]
toc: false
published: false
---

{: class="table-of-content"}
* TOC
{:toc}

# Generative Model Part 2：A Survey on Variational Autoencoders 

"VAE marrys graphical models and deep learning" ---Diederik P. Kingma

正如作者在《An Introduction to Variational Autoencoders》所言，VAE结合了概率模型，图模型，神经网络。

## 1.  Introduction 

### 1.1  Probabilistic Models and Variational Inference

概率模型中，$$\mathbf{X}$$ 代表所有观察数据的集合，也是需要联合分布建模的对象，需要去逼近真实的分布 $$p^{*}(\mathbf{x})$$ ，利用一个由 $$\boldsymbol{\theta}$$ 控制的模型：

$$
\mathbf{x} \sim p_{\boldsymbol{\theta}}(\mathbf{x})
$$

而概率模型中，学习的本质是，寻找最适合的参数 $$\boldsymbol{\theta}$$ 来逼近真实分布  $$p^{*}(\mathbf{x})$$ 

$$
p_{\boldsymbol{\theta}}(\mathbf{x}) \approx p^{*}(\mathbf{x})
$$

因此，希望 $$p^{*}(\mathbf{x})$$ 足够灵活以逼近一个足够精确的模型

上述模型被称为非条件概率模型，与此相对应的，还有一种模型被称为条件概率模型：

$$
p_{\boldsymbol{\theta}}(\mathbf{y} \mid \mathbf{x}) \approx p^{*}(\mathbf{y} \mid \mathbf{x})
$$

常被用于回归，分类问题，尽管条件概率模型与非条件概率模型会在理论上，是两种概率模型，但是实际上，非条件概率模型在实践上，是由观察数据集 $$\mathbf{X}$$ 去建立 $$p_{\boldsymbol{\theta}}(\mathbf{x})$$ ，因此与条件概率模型是等价的，在后面会详细叙述。

### 1.2  Parameterizing Conditional Distributions with Neural Networks

尽管提出了理论上的模型，但是需要具体的模型去进行计算，概率模型的建模才算完成

可微神经网络，或者说神经网络，是常用的计算可行的，泛化性好的方案，作为函数拟合器，基于深度神经网络的模型，一样能做到可以学习概率密度函数。因此基于深度神经网络的概率模型，由于其计算可行性，以及对于神经网络有相当好的优化方案如随机梯度下降，因此作为文中的具体模型，写作 $$NeuralNet(.)$$

比如利用神经网络去建模一个图片分类模型 $$p_{\boldsymbol{\theta}}(y \mid \mathbf{x})$$ ，$$y$$ 代表类， $$\mathbf{x}$$ 代表图片：

$$
\begin{aligned}
\mathbf{p} &=\text { NeuralNet }(\mathbf{x}) \\
p_{\boldsymbol{\theta}}(y \mid \mathbf{x}) &=\text { Categorical }(y ; \mathbf{p})
\end{aligned}
$$

特别注意的是，为了使得输出是一个概率模型，会利用如 softmax 层等作为 $$NeuralNet(.)$$ 的最后一层来规范输出使 $$\sum_{i} p_{i}=1$$

### 1.3  Directed Graphical Models and Neural Networks

到具体的概率模型建模时，使用有向无环图建立概率变量之间的联系，称为概率图模型，比如下图这样的：

{:refdef: style="text-align: center;"}
<img src="/images/2021-01-25-Generative Model Part 2：Generative Model Part 2：A Survey on Variational Autoencoders.md/440px-Graph_model.svg.png" alt="440px-Graph_model.svg" style="zoom:50%;" />
{:refdef}

于是在有向无环图建模下的概率模型，其联合概率可以写成：

$$
p_{\boldsymbol{\theta}}\left(\mathbf{x}_{1}, \ldots, \mathbf{x}_{M}\right)=\prod_{j=1}^{M} p_{\boldsymbol{\theta}}\left(\mathbf{x}_{j} \mid P a\left(\mathbf{x}_{j}\right)\right)
$$

$$P a\left(\mathbf{x}_{j}\right)$$ 指代的是节点 $$j$$ 的所有父节点，而对于根节点，定义其 $$P a\left(\mathbf{x}_{j}\right)$$ 为空集

为了具体的参数化有向无环图概率模型，利用神经网络建模，神经网络的输入是一个随机变量 $$\mathbf{x}$$ 的父节点  $$P a(\mathbf{x})$$，输出的是概率分布 $$\eta$$ ：

$$
\begin{aligned}
\boldsymbol{\eta} &=\operatorname{Neural} \operatorname{Net}(P a(\mathbf{x})) \\
p_{\boldsymbol{\theta}}(\mathbf{x} \mid P a(\mathbf{x})) &=p_{\boldsymbol{\theta}}(\mathbf{x} \mid \boldsymbol{\eta})
\end{aligned}
$$

下面叙述如何学习这样的模型的参数

### 1.4  Learning in Fully Observed Models with Neural Nets

在有向无环图概率模型中，如果在图中所有的随机变量都在数据中被观察到了，那么需要做到就是常规的 MLE 优化，计算，微分（Straightforward！）

#### 1.4.1 Dataset

对采样过程有 $$i.i.d$$ 假设

$$
\mathcal{D}=\left\{\mathbf{x}^{(1)}, \mathbf{x}^{(2)}, \ldots, \mathbf{x}^{(N)}\right\} \equiv\left\{\mathbf{x}^{(i)}\right\}_{i=1}^{N} \equiv \mathbf{x}^{(1: N)}
$$

因此，利用取log分解连乘有：

$$
\log p_{\boldsymbol{\theta}}(\mathcal{D})=\sum_{\mathbf{x} \in \mathcal{D}} \log p_{\boldsymbol{\theta}}(\mathbf{x})
$$

#### 1.4.2 Maximum Likelihood and Minibatch SGD

在 ML 标准中，优化在给出标准后，取寻找最优参数 $$\theta^*$$ 使得标准最优，比如

$$
\theta^* = \arg\min_{\theta} \sum_{\mathbf{x} \in \mathcal{D}} \log p_{\boldsymbol{\theta}}(\mathbf{x})
$$

而常用的求解算法是随机梯度下降算法（SGD），当在整个数据集上进行一次梯度计算的话， 称为 batch gradient descent，但是当数据集很大的时候，更加适合用 minibatch SGD 来处理

### 1.5 Intractabilities

DLVM，深度隐变量模型，建立在隐空间上，隐空间中的变量是难以直接观察得到的，往往很难在数据中直接体现，但是确实存在着隐空间，以mnist手写体数据库来说的话，尽管手写体的维度时 $$28 \times 28$$ 维，但是实际上在实践过程中，在手写过程中，大脑去构建手写体数字的结构时，想的是笔画多重，弯折怎么写，这部分思想过程可以归纳到隐空间建模上

生成模型也是一样的，直接建立 $$p(\mathbf{x})$$ 很难，但是可以通过一组隐变量来建模：

$$
p_{\boldsymbol{\theta}}(\mathbf{x})=\int p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z}) d \mathbf{z} = \int p_{\boldsymbol{\theta}}(\mathbf{z}) p_{\boldsymbol{\theta}}(\mathbf{x} \mid \mathbf{z}) d \mathbf{z}
$$

但是在实际建模中，上述的模型时很难计算的，即 DLVM 的不可处理性

最大似然估计在 DLVM 需要解析的，或者估计出 $$p_{\boldsymbol{\theta}}(\mathbf{x})$$，再通过优化 $$p_{\boldsymbol{\theta}}(\mathbf{x})$$ 找出 $$\theta^*$$，但是由于求解积分 $$p_{\boldsymbol{\theta}}(\mathbf{x})=\int p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z}) d \mathbf{z}$$ 的过程，这个不能直接解析的做出，或估计出，通过 MLE 估计来优化是不可处理的，同时连带的，由于后验分布有计算公式：

$$
p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x})=\frac{p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})}{p_{\boldsymbol{\theta}}(\mathbf{x})}
$$

因此后验分布也是不能计算的，如 MAP 等方法同样不能使用，为了解决不可处理性，生成模型中有几类方案，下面主要关注 Variational Autoencoders

## 2. Variational Autoencoders

VAE 从最大边际后验 $$p_{\theta}(\mathbf{x})$$ 出发，想法有借鉴自动编码器（Autoencoder），但是由于边际后验不可处理性，VAE 的思路是通过做一个参数推断模型 $$q_{\phi}(\mathbf{z} \mid \mathbf{x})$$ 来替换 $$p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x})$$ ，具体来说，就是用 $$q_{\phi}(\mathbf{z} \mid \mathbf{x})$$ 逼近 $$p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x})$$ ：

$$
q_{\phi}(\mathbf{z} \mid \mathbf{x}) \approx p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x})
$$

{:refdef: style="text-align: center;"}
<img src="/images/2021-01-25-Generative Model Part 2：Generative Model Part 2：A Survey on Variational Autoencoders.md/VAE-illustration.png" alt="VAE-illustration" style="zoom:30%;" />
{:refdef}

正如上图所示，如果强调 VAE 的自动编码器的属性，则推断分布 $$q_{\phi}(\mathbf{z} \mid \mathbf{x})$$ 构成了编码器，解码器则是 $$ p_{\boldsymbol{\theta}}(\mathbf{x} \mid \mathbf{z})$$，当然如果能够直接求解 $$ p_{\boldsymbol{\theta}}(\mathbf{x} \mid \mathbf{z})$$ 是最好的，但是上述解码器是无法直接优化的，因此才需要再做一个编码器的推断分布 $$q_{\phi}(\mathbf{z} \mid \mathbf{x})$$ 

在具体问题中，通过一个有向无环图结构来建模推断分布：

$$
q_{\phi}(\mathbf{z} \mid \mathbf{x})=q_{\phi}\left(\mathbf{z}_{1}, \ldots, \mathbf{z}_{M} \mid \mathbf{x}\right)=\prod_{j=1}^{M} q_{\phi}\left(\mathbf{z}_{j} \mid P a\left(\mathbf{z}_{j}\right), \mathbf{x}\right)
$$

为了具体的建模这个逼近，VAE 采用了两步，即 $$Evidence \; Lower \; Bound$$ 来作为优化目标，同时在具体优化过程中采用了 $$ Reparameterization\; Trick$$ 来解决梯度的求解问题

### 2.1 Loss function (Evidence Lower Bound)

边际后验 $$\log p_{\boldsymbol{\theta}}(\mathbf{x}) $$ 与 $$q_{\phi}(\mathbf{z} \mid \mathbf{x})$$ 独立，因此通过重写成期望，可以推导出 $$\log p_{\boldsymbol{\theta}}(\mathbf{x}) $$ 等价于 ELBO 加一个KL散度

$$
\begin{aligned}
\log p_{\boldsymbol{\theta}}(\mathbf{x}) 
&=\mathbb{E}_{q_{\phi}(\mathbf{z} \mid \mathbf{x})}\left[\log p_{\boldsymbol{\theta}}(\mathbf{x})\right] 
\\
&=\mathbb{E}_{q_{\phi}(\mathbf{z} \mid \mathbf{x})}\left[\log \left[\frac{p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})}{p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x})}\right]\right]
\\
&=\mathbb{E}_{q_{\phi}(\mathbf{z} \mid \mathbf{x})}\left[\log \left[\frac{p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})}{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})} \frac{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})}{p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x})}\right]\right]
\\
&= \mathbb{E}_{q_{\phi}(\mathbf{z} \mid \mathbf{x})}\left[\log \left[\frac{p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})}{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})} \frac{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})}{p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x})}\right]\right]
\\
&= \underbrace{\mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})}\left[\log \left[\frac{p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})}{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})}\right]\right]}_{=\mathcal{L}_{\boldsymbol{\theta}, \boldsymbol{\phi}}(\mathbf{x})\\(\mathrm{ELBO})}
+
\underbrace{\mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})}\left[\log \left[\frac{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})}{p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x})}\right]\right]}_{=D_{K L}\left(q_{\phi}(\mathbf{z} \mid \mathbf{x}) \| p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x})\right)}

\end{aligned}
$$

那么最大后验近似于优化 $$\mathcal{L}_{\boldsymbol{\theta}, \boldsymbol{\phi}}(\mathbf{x}) $$ (ELBO)

$$
\begin{aligned}
\mathcal{L}_{\boldsymbol{\theta}, \boldsymbol{\phi}}(\mathbf{x}) 
&= \mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})}\left[\log p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})-\log q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})\right]
\\
 &=\log p_{\boldsymbol{\theta}}(\mathbf{x})-D_{K L}\left(q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) \| p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x})\right) \\
& \leq \log p_{\boldsymbol{\theta}}(\mathbf{x})
\end{aligned}
$$

而第二项KL散度，很巧妙的同时表达了两个距离：

1. ELBO和边际后验中间的距离
2. 推断分布 $$q_{\phi}(\mathbf{z} \mid \mathbf{x})$$ 与后验 $$ p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x})$$ 的距离

优化目标从边际后验到  $$\mathcal{L}_{\boldsymbol{\theta}, \boldsymbol{\phi}}(\mathbf{x}) $$ ，比起不可处理的边际后验， $$\mathcal{L}_{\boldsymbol{\theta}, \boldsymbol{\phi}}(\mathbf{x}) $$ 能够通过随机梯度下降进行优化，理论上来说优化 $$\mathcal{L}_{\boldsymbol{\theta}, \boldsymbol{\phi}}(\mathbf{x}) $$ 可以同时达成两个目标：

1. 优化边际分布 $$\log p_{\boldsymbol{\theta}}(\mathbf{x}) $$ 的变分下界，使得边际分布不小，近似的达成最大似然的效果
2. 使 $$\mathcal{L}_{\boldsymbol{\theta}, \boldsymbol{\phi}}(\mathbf{x}) $$ 第二项 $$ -D_{K L}\left(q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) \| p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x})\right)$$ 变大，即使推断分布和后验分布的距离变小，得到一个更准确的推断分布

### 2.2 Stochastic Gradient-Based Optimization of the ELBO (Reparameterization Trick)

#### 2.2.1 Reparameterization Trick

给一个 $$i.i.d$$ 数据集 $$\mathcal{D}$$，数据集产生 $$\mathcal{L}_{\theta, \phi}(\mathcal{D})$$ 等价于 $$\sum_{\mathbf{x} \in \mathcal{D}} \mathcal{L}_{\theta, \phi}(\mathbf{x})$$，那么下面的讨论中，就变成计算 $$\mathcal{L}_{\theta, \phi}(\mathbf{x})$$，这样是等价的，采用梯度法去优化 $$\sum_{\mathbf{x} \in \mathcal{D}} \mathcal{L}_{\theta, \phi}(\mathbf{x})$$

梯度 $$\nabla_{\boldsymbol{\theta}, \phi} \mathcal{L}_{\boldsymbol{\theta}, \boldsymbol{\phi}}(\mathbf{x})$$ ，对于 $$\theta$$ 有：

$$
\begin{aligned}
\nabla_{\boldsymbol{\theta}} \mathcal{L}_{\boldsymbol{\theta}, \phi}(\mathbf{x}) &=\nabla_{\boldsymbol{\theta}} \mathbb{E}_{q_{\phi}(\mathbf{z} \mid \mathbf{x})}\left[\log p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})-\log q_{\phi}(\mathbf{z} \mid \mathbf{x})\right] \\

&=\mathbb{E}_{q_{\phi}(\mathbf{z} \mid \mathbf{x})}\left[\nabla_{\boldsymbol{\theta}}\left(\log p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})-\log q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})\right)\right] \\

& = \nabla_{\boldsymbol{\theta}}\left(\log p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})-\log q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})\right) \\

&=\nabla_{\boldsymbol{\theta}}\left(\log p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})\right)
\end{aligned}
$$

但是问题在于，对于 $$\phi$$，梯度难以直接计算

$$
\begin{aligned}
\nabla_{\phi} \mathcal{L}_{\theta, \phi}(\mathbf{x}) &=\nabla_{\phi} \mathbb{E}_{q_{\phi}(\mathbf{z} \mid \mathbf{x})}\left[\log p_{\theta}(\mathbf{x}, \mathbf{z})-\log q_{\phi}(\mathbf{z} \mid \mathbf{x})\right] \\
& \neq \mathbb{E}_{q_{\phi}(\mathbf{z} \mid \mathbf{x})}\left[\nabla_{\phi}\left(\log p_{\theta}(\mathbf{x}, \mathbf{z})-\log q_{\phi}(\mathbf{z} \mid \mathbf{x})\right)\right]
\end{aligned}
$$

VAE 提供的解决办法是去构造一个 $$\nabla_{\phi} \mathcal{L}_{\theta, \phi}(\mathbf{x})$$ 无偏估计 $$\nabla_{\boldsymbol{\theta}, \boldsymbol{\phi}} \tilde{\mathcal{L}}_{\boldsymbol{\theta}, \boldsymbol{\phi}}(\mathbf{x} ; \boldsymbol{\epsilon})$$ ，即另一个技巧 $$Reparameterization \; Trick$$

$$\nabla_{\phi} \mathcal{L}_{\theta, \phi}(\mathbf{x})$$ 是由于计算 $$q_{\phi}(\mathbf{z} \mid \mathbf{x})$$ 的期望，再此基础上算 $$\phi$$ 的偏导数就导致难以解析的计算，$$Reparameterization \; Trick$$ 设计了一个无偏统计量来估计 $$\nabla_{\phi} \mathcal{L}_{\theta, \phi}(\mathbf{x})$$

{:refdef: style="text-align: center;"}
<img src="/images/2021-01-25-Generative Model Part 2：Generative Model Part 2：A Survey on Variational Autoencoders.md/Reparameterization-Trick.png" alt="Reparameterization-Trick" style="zoom:40%;" />
{:refdef}

如上图所示，构造一个噪声变量 $$\epsilon$$ 有分布 $$p(\boldsymbol{\epsilon})$$，定义新的映射关系：$$\mathbf{z}=\mathbf{g}(\boldsymbol{\epsilon}, \boldsymbol{\phi}, \mathbf{x})$$ 满足

$$
\mathbb{E}_{q_{\phi}(\mathbf{z} \mid \mathbf{x})}[f(\mathbf{z})]=\mathbb{E}_{p(\boldsymbol{\epsilon})}[f(\mathbf{z})]
$$

在此基础上进行求导就可以有一个 $$\nabla_{\boldsymbol{\phi}} f(\mathbf{z})$$ 的估计：

$$
\begin{aligned}
\nabla_{\phi} \mathbb{E}_{q_{\phi}(\mathbf{z} \mid \mathbf{x})}[f(\mathbf{z})] &=\nabla_{\phi} \mathbb{E}_{p(\boldsymbol{\epsilon})}[f(\mathbf{z})] \\
&=\mathbb{E}_{p(\boldsymbol{\epsilon})}\left[\nabla_{\boldsymbol{\phi}} f(\mathbf{z})\right] \\
& \simeq \nabla_{\boldsymbol{\phi}} f(\mathbf{z})
\end{aligned}
$$

更进一步，代入具体的函数 $$f(z) = \log p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})-\log q_{\phi}(\mathbf{z} \mid \mathbf{x})$$，构造一个蒙特卡洛估计
$$\tilde{\mathcal{L}}_{\boldsymbol{\theta}, \phi}(\mathbf{x})$$：

$$
\begin{aligned}
\boldsymbol{\epsilon} & \sim p(\boldsymbol{\epsilon})
\\
\mathbf{z} &=\mathbf{g}(\boldsymbol{\phi}, \mathbf{x}, \boldsymbol{\epsilon}) 
\\
\tilde{\mathcal{L}}_{\boldsymbol{\theta}, \phi}(\mathbf{x}) &=\log p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})-\log q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})
\end{aligned}
$$

可以证明 $$\tilde{\mathcal{L}}_{\boldsymbol{\theta}, \phi}(\mathbf{x})$$ 是无偏的：

$$
\begin{aligned}
\mathbb{E}_{p(\boldsymbol{\epsilon})}\left[\nabla_{\boldsymbol{\theta}, \phi} \tilde{\mathcal{L}}_{\boldsymbol{\theta}, \boldsymbol{\phi}}(\mathbf{x} ; \boldsymbol{\epsilon})\right] &=\mathbb{E}_{p(\boldsymbol{\epsilon})}\left[\nabla_{\boldsymbol{\theta}, \boldsymbol{\phi}}\left(\log p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})-\log q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})\right)\right] \\
&=\nabla_{\boldsymbol{\theta}, \boldsymbol{\phi}}\left(\mathbb{E}_{p(\boldsymbol{\epsilon})}\left[\log p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})-\log q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})\right]\right) \\
&=\nabla_{\boldsymbol{\theta}, \phi} \mathcal{L}_{\boldsymbol{\theta}, \boldsymbol{\phi}}(\mathbf{x})
\end{aligned}
$$

#### 2.2.2 Computation of Inference Distribution

在计算 $$\tilde{\mathcal{L}}_{\boldsymbol{\theta}, \phi}(\mathbf{x}) =\log p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})-\log q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})$$ 的过程中，需要计算编码器分布 $$\log q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})$$ 

$$
\begin{aligned}
\boldsymbol{\epsilon} & \sim p(\boldsymbol{\epsilon})
\\
\mathbf{z} &=\mathbf{g}(\boldsymbol{\phi}, \mathbf{x}, \boldsymbol{\epsilon}) 
\\
p(\boldsymbol{\epsilon}) &= q_{\phi}(\mathbf{z} \mid \mathbf{x})\left|\operatorname{det}\left(\frac{\partial \mathbf{z}}{\partial \boldsymbol{\epsilon}}\right)\right|

\end{aligned}
$$

因此可以看出，选择一个合适的可逆变化 $$g(\epsilon)$$ 会使得计算简化

$$
\log q_{\phi}(\mathbf{z} \mid \mathbf{x})=\log p(\boldsymbol{\epsilon})-\log \left|\operatorname{det}\left(\frac{\partial \mathbf{z}}{\partial \boldsymbol{\epsilon}}\right)\right|
$$

下面举一个具体计算中构造推断分布和 $$Reparameterization \; Trick$$ 的例子：

推断分布（编码器分布）取作高斯分布，高斯分布的参数通过编码器计算得到：

$$
\begin{aligned}
q_{\phi}(\mathbf{z} \mid \mathbf{x})  &= \mathcal{N}(\mathbf{z} ; \boldsymbol{\mu}, \operatorname{diag}\left(\sigma^{2}\right))
\\
(\boldsymbol{\mu}, \log \boldsymbol{\sigma}) &=\text { EncoderNeuralNet}_{\boldsymbol{\phi}}(\mathbf{x}) 
\\
q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) &=\prod_{i} q_{\boldsymbol{\phi}}\left(z_{i} \mid \mathbf{x}\right)=\prod_{i} \mathcal{N}\left(z_{i} ; \mu_{i}, \sigma_{i}^{2}\right)
\end{aligned}
$$

采用 $$Reparameterization \; Trick$$：

$$
\begin{array}{l}
\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I}) \\
\mathbf{z}=\mu+\sigma \odot \epsilon
\end{array}
$$

因此就可以计算后验分布 $$\log q_{\phi}(\mathbf{z} \mid \mathbf{x})$$ 为：

$$
\begin{aligned}
\log q_{\phi}(\mathbf{z} \mid \mathbf{x}) &=\log p(\boldsymbol{\epsilon})-\log d_{\boldsymbol{\phi}}(\mathbf{x}, \boldsymbol{\epsilon}) \\
&=\sum_{i} \log \mathcal{N}\left(\epsilon_{i} ; 0,1\right)-\log \sigma_{i}
\end{aligned}
$$

### 2.3 Estimation of the Marginal Likelihood

边际似然可以利用推断分布去构造：

$$
\log p_{\boldsymbol{\theta}}(\mathbf{x})=\log \mathbb{E}_{q_{\phi}(\mathbf{z} \mid \mathbf{x})}\left[p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z}) / q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})\right]
$$

而在数值上，上式可以利用蒙特卡洛法进行估计：

$$
\log p_{\boldsymbol{\theta}}(\mathbf{x}) \approx \log \frac{1}{L} \sum_{l=1}^{L} p_{\boldsymbol{\theta}}\left(\mathbf{x}, \mathbf{z}^{(l)}\right) / q_{\boldsymbol{\phi}}\left(\mathbf{z}^{(l)} \mid \mathbf{x}\right)
$$

当 $$L=1 $$ 时，这个式子就是 $$\mathcal{L}_{\theta, \phi}(\mathcal{x})$$

### 2.4 Summary of Training 

VAE，和GAN一样，最后通过从隐空间采样，然后用解码器映射到数据空间，生成数据，但是VAE仍然会去训练

#### 2.4.1 Optimization issues


---
layout: post
title: "Generative Model Part 2：A Survey on Variational Autoencoder"
date: 2021-01-25
image: /images/cover/C_ Abstraction3.jpeg         
tags: [Generative-Model]
toc: false
published: true
---


{: class="table-of-content"}
* TOC
{:toc}

# Generative Model Part 2：A Survey on Variational Autoencoder

"VAE marrys graphical models and deep learning" ---Diederik P. Kingma

正如VAE的作者在《An Introduction to Variational Autoencoders》所言，VAE结合了概率模型，图模型，神经网络，成功的实现了对于复杂分布的拟合，建立在神经网络这样的标准的，高效的拟合器上，也可以利用梯度下降算法进行训练

## 1.  Introduction 

这一部分中，主要叙述关于概率模型，神经网络，图模型的一些基础

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

因此，利用取 $$\log$$ 分解连乘有：

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

训练完成的 VAE ，从隐空间中采样 $$\boldsymbol{z}$$，利用解码器 $$p_{\boldsymbol{\theta}}(\mathbf{x} \mid \mathbf{z})$$ 去复建出 $$\boldsymbol{X}$$，其模型如下所示

{:refdef: style="text-align: center;"}
<img src="/images/2021-01-25-Generative Model Part 2：Generative Model Part 2：A Survey on Variational Autoencoders.md/latent_encoder_graph.png" alt="latent_encoder_graph" style="zoom:30%;" />
{:refdef}

但是由于后验不可处理性，生成模型并无法直接根据数据集直接求解 $$p_{\boldsymbol{\theta}}(\mathbf{x} \mid \mathbf{z})$$ ，有几种解决办法来处理这个问题。VAE 仍然从最大边际后验 $$p_{\theta}(\mathbf{x})$$ 的想法出发，但是优化后验的一个下界来使后验变大（见后文 ELBO 部分），想法有借鉴自动编码器（Autoencoder），但是由于边际后验 $$p_{\boldsymbol{\theta}}(\mathbf{x})$$ 不可处理性。VAE补全了自动编码器的前半部分，通过做一个参数推断模型 $$q_{\phi}(\mathbf{z} \mid \mathbf{x})$$ 来替换 $$p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x})$$ 模拟编码器
$$
q_{\phi}(\mathbf{z} \mid \mathbf{x}) \approx p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x})
$$

{:refdef: style="text-align: center;"}
<img src="/images/2021-01-25-Generative Model Part 2：Generative Model Part 2：A Survey on Variational Autoencoders.md/VAE-illustration.png" alt="VAE-illustration" style="zoom:30%;" />
{:refdef}

正如上图所示，如果强调 VAE 的自动编码器的属性，则推断分布 $$q_{\phi}(\mathbf{z} \mid \mathbf{x})$$ 构成了编码器，解码器则是 $$ p_{\boldsymbol{\theta}}(\mathbf{x} \mid \mathbf{z})$$，当然如果能够直接求解 $$ p_{\boldsymbol{\theta}}(\mathbf{x} \mid \mathbf{z})$$ 是最好的，但是上述解码器是无法直接优化的，因此才需要再做一个编码器的推断分布 $$q_{\phi}(\mathbf{z} \mid \mathbf{x})$$ ，有了推断分布后，就可以从其中采样 $$z$$，再镜像的利用解码器 $$ p_{\boldsymbol{\theta}}(\mathbf{x} \mid \mathbf{z})$$ 映射回数据空间后，利用交叉熵去算出 loss，利用 SGD 去优化编码器和解码器的参数 $$\phi,\theta$$

在具体问题中，通过一个有向无环图结构来建模推断分布：

$$
q_{\phi}(\mathbf{z} \mid \mathbf{x})=q_{\phi}\left(\mathbf{z}_{1}, \ldots, \mathbf{z}_{M} \mid \mathbf{x}\right)=\prod_{j=1}^{M} q_{\phi}\left(\mathbf{z}_{j} \mid P a\left(\mathbf{z}_{j}\right), \mathbf{x}\right)
$$

为了具体的建模上述逼近，VAE 采用了两步，即 $$Evidence \; Lower \; Bound$$ 来作为优化目标，同时在具体优化过程中采用了 $$ Reparameterization\; Trick$$ 来解决梯度的求解问题

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
2. 使 $$\mathcal{L}_{\boldsymbol{\theta}, \boldsymbol{\phi}}(\mathbf{x}) $$ 第二项 $$ -D_{K L}\left(q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) \| p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x})\right)$$ 变大，即使推断分布和后验分布的距离变小，得到一个更准确的编码器分布，一旦能够准确建模编码器分布，有助于优化解码器分布 $$ p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x})$$ 

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

但是问题在于，由于之前对于 $$\phi$$，由于 $$z$$ 是 $$\phi$$ 的函数，这样梯度难以解析的直接计算

$$
\begin{aligned}
\nabla_{\phi} \mathcal{L}_{\theta, \phi}(\mathbf{x}) &=\nabla_{\phi} \mathbb{E}_{q_{\phi}(\mathbf{z} \mid \mathbf{x})}\left[\log p_{\theta}(\mathbf{x}, \mathbf{z})-\log q_{\phi}(\mathbf{z} \mid \mathbf{x})\right] \\
& \neq \mathbb{E}_{q_{\phi}(\mathbf{z} \mid \mathbf{x})}\left[\nabla_{\phi}\left(\log p_{\theta}(\mathbf{x}, \mathbf{z})-\log q_{\phi}(\mathbf{z} \mid \mathbf{x})\right)\right]
\end{aligned}
$$

VAE 提供的解决办法是去构造一个 $$\nabla_{\phi} \mathcal{L}_{\theta, \phi}(\mathbf{x})$$ 无偏估计 $$\nabla_{\boldsymbol{\theta}, \boldsymbol{\phi}} \tilde{\mathcal{L}}_{\boldsymbol{\theta}, \boldsymbol{\phi}}(\mathbf{x} ; \boldsymbol{\epsilon})$$ ，即另一个技巧 $$Reparameterization \; Trick$$

$$\nabla_{\phi} \mathcal{L}_{\theta, \phi}(\mathbf{x})$$ 是由于计算 $$q_{\phi}(\mathbf{z} \mid \mathbf{x})$$ 的期望，再此基础上算 $$\phi$$ 的偏导数就导致难以解析的计算，这样就无法执行梯度下降算法，于是作者在 $$Reparameterization \; Trick$$ 中设计了一个无偏统计量来估计 $$\nabla_{\phi} \mathcal{L}_{\theta, \phi}(\mathbf{x})$$：

{:refdef: style="text-align: center;"}
<img src="/images/2021-01-25-Generative Model Part 2：Generative Model Part 2：A Survey on Variational Autoencoders.md/Reparameterization-Trick.png" alt="Reparameterization-Trick" style="zoom:40%;" />
{:refdef}

如上图所示，构造一个噪声变量 $$\epsilon$$ 有分布 $$p(\boldsymbol{\epsilon})$$，定义新的映射关系：$$\mathbf{z}=\mathbf{g}(\boldsymbol{\epsilon}, \boldsymbol{\phi}, \mathbf{x})$$ 满足，其中 $$g(.)$$ 未知，但是 $$\epsilon$$ 相对的比较简单

$$
\mathbb{E}_{q_{\phi}(\mathbf{z} \mid \mathbf{x})}[f(\mathbf{z})]=\mathbb{E}_{p(\boldsymbol{\epsilon})}[f(\mathbf{z})]
$$

这样就可以把 $$z$$ 的随机性转移到 $$\epsilon$$ 上，先在抽象一些的问题设定下研究 $$Reparameterization \; Trick$$，在此基础上对 $$\phi$$ 求导，可以有一个 $$\nabla_{\boldsymbol{\phi}} f(\mathbf{z})$$ 的估计：

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

可以证明 $$\nabla_{\boldsymbol{\theta}, \phi} \tilde{\mathcal{L}}_{\boldsymbol{\theta}, \phi}(\mathbf{x})$$ 是 $$\nabla_{\boldsymbol{\theta}, \phi} \tilde{\mathcal{L}}_{\boldsymbol{\theta}, \boldsymbol{\phi}}(\mathbf{x} ; \boldsymbol{\epsilon})$$ 无偏估计：

$$
\begin{aligned}
\mathbb{E}_{p(\boldsymbol{\epsilon})}\left[\nabla_{\boldsymbol{\theta}, \phi} \tilde{\mathcal{L}}_{\boldsymbol{\theta}, \boldsymbol{\phi}}(\mathbf{x} ; \boldsymbol{\epsilon})\right] &=\mathbb{E}_{p(\boldsymbol{\epsilon})}\left[\nabla_{\boldsymbol{\theta}, \boldsymbol{\phi}}\left(\log p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})-\log q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})\right)\right] \\
&=\nabla_{\boldsymbol{\theta}, \boldsymbol{\phi}}\left(\mathbb{E}_{p(\boldsymbol{\epsilon})}\left[\log p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})-\log q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})\right]\right) \\
&=\nabla_{\boldsymbol{\theta}, \phi} \mathcal{L}_{\boldsymbol{\theta}, \boldsymbol{\phi}}(\mathbf{x})
\end{aligned}
$$

在实际计算中，类似SGD的思想一样，利用多次比较“模糊”的计算来替换单次“精准”的计算，即直接计算：
$$
\nabla_{\boldsymbol{\theta, \phi}}\log p_{\theta}(\mathbf{x}  \mid \mathbf{z} )+\log p_{\phi}(\mathbf{z} )-\log q(\mathbf{z}  \mid \mathbf{x} )
\quad
\mathbf{z}  \text{ ～ } q(\mathbf{z}  \mid \mathbf{x} )
$$
当然这样说可能还是有一些抽象，下面会具体的举一个具体的隐空间 $$z$$ 和相应的 $$\epsilon$$ 的例子

#### 2.2.2 Computation of Inference Distribution

在计算 $$\tilde{\mathcal{L}}_{\boldsymbol{\theta}, \phi}(\mathbf{x}) =\log p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})-\log q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})$$ 的过程中，需要计算编码器/推断分布 $$\log q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})$$ 

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

下面举一个具体计算中构造编码器分布 $$\log q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})$$ 和 $$Reparameterization \; Trick$$ 的例子：

设隐空间 $$\mathbf{z} $$ 服从正态分布，而条件分布 $$q_{\phi}(\mathbf{z} \mid \mathbf{x})$$ 的参数通过编码器计算得到：

$$
\begin{aligned}
q_{\phi}(\mathbf{z} \mid \mathbf{x})  &= \mathcal{N}(\mathbf{z} ; \boldsymbol{\mu}, \operatorname{diag}\left(\sigma^{2}\right))
\\
(\boldsymbol{\mu}, \log \boldsymbol{\sigma}) &=\text { EncoderNeuralNet}_{\boldsymbol{\phi}}(\mathbf{x}) 
\\
q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) &=\prod_{i} q_{\boldsymbol{\phi}}\left(z_{i} \mid \mathbf{x}\right)=\prod_{i} \mathcal{N}\left(z_{i} ; \mu_{i}, \sigma_{i}^{2}\right)
\end{aligned}
$$

这样直接算下去的话，之后无法反向传播，采用 $$Reparameterization \; Trick$$：

$$
\begin{array}{l}
\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I}) \\
\mathbf{z}=\mu+\sigma \odot \epsilon
\end{array}
$$

进一步就可以计算编码器分布 $$\log q_{\phi}(\mathbf{z} \mid \mathbf{x})$$ 为：
$$
\begin{aligned}
\log q_{\phi}(\mathbf{z} \mid \mathbf{x}) &=\log p(\boldsymbol{\epsilon})-\log d_{\boldsymbol{\phi}}(\mathbf{x}, \boldsymbol{\epsilon}) \\
&=\sum_{i} \log \mathcal{N}\left(\epsilon_{i} ; 0,1\right)-\log \sigma_{i}
\end{aligned}
$$

### 2.3 Estimation of the Marginal Likelihood

边际似然可以利用编码器分布去构造：

$$
\log p_{\boldsymbol{\theta}}(\mathbf{x})=\log \mathbb{E}_{q_{\phi}(\mathbf{z} \mid \mathbf{x})}\left[p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z}) / q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})\right]
$$

而在数值上，上式可以利用蒙特卡洛法进行估计：

$$
\log p_{\boldsymbol{\theta}}(\mathbf{x}) \approx \log \frac{1}{L} \sum_{l=1}^{L} p_{\boldsymbol{\theta}}\left(\mathbf{x}, \mathbf{z}^{(l)}\right) / q_{\boldsymbol{\phi}}\left(\mathbf{z}^{(l)} \mid \mathbf{x}\right)
$$

当 $$L=1 $$ 时，这个式子就是 $$\mathcal{L}_{\theta, \phi}(\mathcal{x})$$

### 2.5 Summary of Training

VAE，和GAN一样，最后通过从隐空间采样，然后用解码器映射到数据空间，生成数据，但是比起GAN通过去迭代训练生成器和判别器的结构，最后把生成器当作映射，VAE更加强调对于数据分布的理解

- VAE 假设隐空间 $$\mathbf{z}$$ 的编码器分布 $$q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})$$ 服从一个高斯分布（强调模型的表达能力选择半三角作为高斯分布方差，反之选择对角矩阵作为方差），利用编码器把 $$\mathbf{x} $$ 从数据空间映射到隐空间，输出分布的 $$\mu,\sigma$$

- 通过 $$Reparameterization \;trick$$ 把 $$\mathbf{z} $$ 的随机性转到 $$\epsilon$$ 上，采样 $$\mathbf{z} $$

- $$\mathbf{z} $$ 经过解码器，利用交叉熵计算出后验分布 $$p(\mathbf{x}  \mid \mathbf{z} )$$，最后计算产生损失进行反向传播
  $$
  \mathcal{L}_{\theta, \phi}(\mathcal{\mathbf{x} }) = \log p_{\boldsymbol{\theta}}(\mathbf{x} \mid \mathbf{z}) +\log p(\mathbf{z}) -\log q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) \quad (L=1 )
  $$

上式中，非常值得注意的一点是，后两项其实不去限定在 $$L=1 $$ 的条件下的话等价于 $$-D_{K L}\left(q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})\| p_{\theta}(\mathbf{z})\right)$$ ，这样在训练过程中，隐空间的先验分布 $$p_{\theta}(\mathbf{z})$$ 和解码器用来训练自己的分布 $$q_{\boldsymbol{\theta}}(\mathbf{x} \mid \mathbf{z})$$ 不会偏离很远，这样就可以在时候进行生成时，隐空间中采样的 $$\mathbf{z}$$ 会离之前解码器学习过的样本的数据流形太远

下面借用 Tensorflow.org 开源的 [Convolutional Variational Autoencoder](https://www.tensorflow.org/tutorials/generative/cvae) 的实现

#### 2.5.1 Data Preposscessing

数据预处理中，对图像进行灰度->黑白处理，这样最后输出利用伯努利二项分布进行输出即可：

```python
def preprocess_images(images):
  images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
  return np.where(images > .5, 1.0, 0.0).astype('float32')

train_images = preprocess_images(train_images)
test_images = preprocess_images(test_images)

```

#### 2.5.2 Network Architecture

CVAE网络结构是一个一对耦合的解码器和编码器，解码器由两层卷机层和一层全联接层构成，解码器把数据映射到隐空间上（输出隐空间的参数），这里假设一个高斯分布作为隐空间的分布，为了实现 $$Reparameterization \; Trick$$ 则给出了另一个服从标准正态分布的参数 $$\epsilon$$：

$$
z=\mu+\sigma \odot \epsilon
$$

然后解码器是编码器的镜像，镜像的全联接层后接了一层反卷积层

利用解码器和编码器，可以进一步写出 encode(.), decode(.), sample(.),, reparameterize(.) 并封装到 CVAE 类中

```python
class CVAE(tf.keras.Model):
  """Convolutional variational autoencoder."""

  def __init__(self, latent_dim):
    super(CVAE, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            # No activation
            tf.keras.layers.Dense(latent_dim + latent_dim),
        ]
    )

    self.decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            # No activation
            tf.keras.layers.Conv2DTranspose(
                filters=1, kernel_size=3, strides=1, padding='same'),
        ]
    )

  @tf.function
  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=True)

  def encode(self, x):
    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z, apply_sigmoid=False):
    logits = self.decoder(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs
    return logits
```

#### 2.5.3 Loss Function

然后计算损失函数的过程中，方差取的是 $$log var$$ 防止数值上的问题，后验分布利用交叉熵计算即可，为了加速计算，只一次采样 $$z$$ ：

$$
\log p_{\boldsymbol{\theta}}(\mathbf{x} \mid \mathbf{z}) = - H(q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}), p_{\boldsymbol{\theta}}(\mathbf{x} \mid \mathbf{z}))
$$

```python
def compute_loss(model, x):
  mean, logvar = model.encode(x)
  z = model.reparameterize(mean, logvar)
  x_logit = model.decode(z)
  cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
  logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
  logpz = log_normal_pdf(z, 0., 0.)
  logqz_x = log_normal_pdf(z, mean, logvar)
  return -tf.reduce_mean(logpx_z + logpz - logqz_x)

```

### 2.4 Challenges

VAE，和GAN一样，最后通过从隐空间采样，然后用解码器映射到数据空间，生成数据，但是和GAN不一样的是，GAN训练的是可能性（probability），即利用判别器去判断生成数据是真实数据的可能性，VAE仍然会去训练数据的分布（distribution），那么这一点也给VAE的训练带来的困难

#### 2.4.1 Optimization issues

整个训练过程中，VAE 都要去算数据的后验分布 $$p(x \mid z)$$ 来进行优化，但是一开始训练时，解码器的效果会很差，无法把隐空间中的采样 $$z$$ 映射到真实的数据流形上，那么此时计算出的 $$\mathcal{L}_{\theta, \phi}(\mathcal{x}) $$ 中 $$\log p_{\boldsymbol{\theta}}(\mathbf{x} \mid \mathbf{z})$$ 很小，且优化很难，为了使 $$\mathcal{L}_{\theta, \phi}(\mathcal{x}) $$ 变大

$$
\mathcal{L}_{\theta, \phi}(\mathcal{\mathbf{x} }) = \mathbb{E}_{q_{\phi}(\mathbf{z} \mid \mathbf{x})}[\log p_{\boldsymbol{\theta}}(\mathbf{x} \mid \mathbf{z}) +\log p(\mathbf{z}) -\log q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})]
$$

后半部分 $$\mathbb{E}_{q_{\phi}(\mathbf{z} \mid \mathbf{x})}[\log p(\mathbf{z}) /q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})] =  -D_{K L}\left(q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})\| p_{\theta}(\mathbf{z})\right)$$ 就会变大，又由KL散度的非负性，此时模型就会到达一个无意义的均衡点 $$\log p(\mathbf{z}) \approx \log q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) $$，此时编码器 $$q_{\phi}(\mathbf{z} \mid \mathbf{x})$$ 没有性能去做有意义的编码，继而解码器很难把隐变量  $$\mathbf{z}$$ 复建回数据流形上对应的 $$\mathbf{x}$$

解决方法有在训练过程中，首先冻结（前面乘一个0～1之前的控制系数）KL散度这一正则项，在训练过程中逐步把KL散度解冻（但是解冻的速度选择就很有炼丹的味道，当解冻速度太快时，会落到上述的无意义均衡中，当解冻的速度慢了的话，那么会浪费时间，并且解码器确实无法得到有意义的训练）

## Reference

1. Diederik P. Kingma and Max Welling (2019), “An Introduction to Variational Autoencoders”, Foundations and TrendsR in Machine Learning 
2.  [Diederik P Kingma](https://arxiv.org/search/stat?searchtype=author&query=Kingma%2C+D+P), [Max Welling](https://arxiv.org/search/stat?searchtype=author&query=Welling%2C+M) (2013), "Auto-Encoding Variational Bayes" (2013)
3.  [Carl Doersch](https://arxiv.org/search/stat?searchtype=author&query=Doersch%2C+C) (2016) "Tutorial on Variational Autoencoders" 
4. Tensorflow Author  [Convolutional Variational Autoencoder](https://www.tensorflow.org/tutorials/generative/cvae) Tensorflow Author








---
layout: post
title: "Generative Model Part 3：Towards Deeper Understanding on Variational Autoencoders"
date: 2021-02-2
image: /images/cover/C_ Abstraction1.jpeg         
tags: [Generative-Model]
toc: false
published: false
---

{: class="table-of-content"}
* TOC
{:toc}

# Generative Model Part 3：Towards Deeper Understanding on Variational Autoencoders

概括一下 VAE 的构建过程，如下图所示：

{:refdef: style="text-align: center;"}
<img src="/images/2021-02-23-Generative Model Part 2：Generative Model Part 3：Towards Deeper Understanding of Variational Autoencoder.md/VAE-training.png" alt="VAE-training" style="zoom:35%;" />
{:refdef}


在训练过程中，即希望复建的误差小，同时希望隐空间上的两个分布不要差的太远使得解码器失效，这在训练中对于编码器和解码器的能力提出了很高的要求：

在前篇对于 VAE 的介绍中，利用 $$\mathcal{L}_{\theta, \phi}(\mathbf{x})$$ 来近似 $$\log p_{\boldsymbol{\theta}}(\mathbf{x})$$ ，成功的优化了编码器参数 $$\phi$$ 和解码器参数 $$\theta$$ ，但是在实践中，需要去选择一个表达能力比较强的模型，下面先证明一下这样做的必要性：

可以通过经验分布函数 $$q_{\mathcal{D}}(\mathbf{x})$$ 可以改写似然函数：

$$
\begin{aligned}
\log p_{\boldsymbol{\theta}}(\mathcal{D}) &=\frac{1}{N_{\mathcal{D}}} \sum_{\mathbf{x} \in \mathcal{D}} \log p_{\boldsymbol{\theta}}(\mathbf{x}) \\
&=\mathbb{E}_{q_{\mathcal{D}}(\mathbf{x})}\left[\log p_{\boldsymbol{\theta}}(\mathbf{x})\right]
\end{aligned}
$$

那么经验分布和实际分布的 KL 散度（两者之差）等价于一个负的似然，即两者越相似，似然函数越大

$$
\begin{aligned}
D_{K L}\left(q_{\mathcal{D}}(\mathbf{x}) \| p_{\boldsymbol{\theta}}(\mathbf{x})\right) &=-\mathbb{E}_{q_{\mathcal{D}}(\mathbf{x})}\left[\log p_{\boldsymbol{\theta}}(\mathbf{x})\right]+\mathbb{E}_{q_{\mathcal{D}}(\mathbf{x})}\left[\log q_{\mathcal{D}}(\mathbf{x})\right] \\
&=-\log p_{\boldsymbol{\theta}}(\mathcal{D})+\text { constant }
\end{aligned}
$$

其中常数是 $$\text{constant} = -\mathcal{H}\left(q_{\mathcal{D}}(\mathbf{x})\right)$$，进一步的可以建立隐空间和数据空间的联合分布 $$(\mathbf{x} ,\mathbf{z})$$，有两种途径可以达到建模，分别是经验分布/编码器分布 $$ q_{\mathcal{D}}, \phi(\mathbf{x}, \mathbf{z})$$ 和生成分布 $$p_{\theta}(\mathbf{x}, \mathbf{z})$$，而两者之间的差也可以等价的写成：

$$
\begin{array}{l}
D_{K L}\left(q_{\mathcal{D}, \phi}(\mathbf{x}, \mathbf{z}) \| p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})\right) 
\\
=-\mathbb{E}_{q_{\mathcal{D}}(\mathbf{x})}\left[\mathbb{E}_{q_{\phi}(\mathbf{z} \mid \mathbf{x})}\left[\log p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})-\log q_{\phi}(\mathbf{z} \mid \mathbf{x})\right]-\log q_{\mathcal{D}}(\mathbf{x})\right] 
\\
=-\mathcal{L}_{\theta, \phi}(\mathcal{D})+\text { constant }
\end{array}
$$

也就是说，经验分布和隐空间的联合分布和生成分布的差越小，$$\mathcal{L}_{\theta, \phi}(\mathcal{D})$$ 越大，进一步的说，可以说 VAE 训练的 $$ELBO$$ 目标函数与一般的 $$ML $$ 目标函数 $$D_{K L}\left(q_{\mathcal{D}}(\mathbf{x}) \| p_{\boldsymbol{\theta}}(\mathbf{x})\right)$$ 有下列关系，中间会有一个 $$gap$$

$$
\begin{array}{l}
D_{K L}\left(q_{\mathcal{D}, \phi}(\mathbf{x}, \mathbf{z}) \| p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})\right) \\
=D_{K L}\left(q_{\mathcal{D}}(\mathbf{x}) \| p_{\boldsymbol{\theta}}(\mathbf{x})\right)+\mathbb{E}_{q \mathcal{D}(\mathbf{x})}\left[D_{K L}\left(q_{\mathcal{D}, \phi}(\mathbf{z} \mid \mathbf{x}) \| p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x})\right)\right] \\
\geq D_{K L}\left(q_{\mathcal{D}}(\mathbf{x}) \| p_{\boldsymbol{\theta}}(\mathbf{x})\right)
\end{array}
$$

$$D_{K L}\left(q_{\mathcal{D}, \phi}(\mathbf{x}, \mathbf{z}) \| p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})\right)$$ 和 $$\mathcal{L}_{\theta, \phi}(\mathcal{D})$$ 的等价性，使理解一些模型发生的现象变得容易：

比如 VAE 的生成模型生成的结果比较模糊，这里需要用到 $$KL$$ 散度的不对称性，当某些点 $$(\mathbf{x}, \mathbf{z})$$ 在 $$\text{supp} (q_{\mathcal{D}, \phi}(\mathbf{x}, \mathbf{z}))$$ 中，为了限制 $$KL$$ 散度变大，$$p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})$$ 会在该点分配一定质量，这是一件好事情，但是相反的，如果 $$D_{K L}\left(q_{\mathcal{D}, \phi}(\mathbf{x}, \mathbf{z}) \| p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})\right)$$ 并不能限制生成分布 $$p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})$$ 在 $$\text{supp} (q_{\mathcal{D}, \phi}(\mathbf{x}, \mathbf{z}))$$ 外分配一些质量，这些点最后在生成时，就会导致出现模糊的采样点，此时就需要一个更强的生成模型或者推断模型来解决这个问题

## 1. Improving the Flexibility of Inference Model

那么一个计算性质上好的推断模型，应当能够做到：

1. $$q_{\phi}(\mathbf{z} \mid \mathbf{x})$$ 能够高效计算，尤其是微分计算
2. $$q_{\phi}(\mathbf{z} \mid \mathbf{x})$$ 中采样 $$\mathbf{z}$$ 的计算量不大
3. 在高维情况下（往往数据空间中都是这样的情况），$$q_{\phi}(\mathbf{z} \mid \mathbf{x})$$ 的计算应当能够做到并行化

抛去计算可行性，最重要的一点是，$$q_{\phi}(\mathbf{z} \mid \mathbf{x})$$ 的表达力足够强以至于做得到拟合后验分布 $$ p_{}(\mathbf{z} \mid \mathbf{x})$$ :

$$
\log p_{\theta }(\mathbf{x}) = 
\mathbb{E}_{q_{\phi}(\mathbf{z} \mid \mathbf{x})}\left[\log \left[\frac{p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})}{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})}\right]\right]+\mathbb{E}_{q_{\phi}(\mathbf{z} \mid \mathbf{x})}\left[\log \left[\frac{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})}{p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x})}\right]\right]
$$

这样第二项 $$D_{K L}\left(q_{\phi}(\mathbf{z} \mid \mathbf{x}) \| p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x})\right) \geq 0$$ 就会小，到达一个更紧的逼近 $$\mathcal{L}_{\theta, \phi}(\mathbf{x}) \rightarrow \log p_{\theta }(\mathbf{x}) $$

有两种技巧来提升后验估计的准确性：Auxiliary Latent Variables 和 Normalizing Flows.

### 1.1. Auxiliary Latent Variables

事实上，隐空间的维度对于模型生成质量是非常重要的，在Mnist数据集上的实验体现了这一点，在控制训练集一致的情况下，相同的隐空间采样，左边一列解码器的输入为5，右边为10维，如图所示显然10维的生成效果更加好

{:refdef: style="text-align: center;"}
<img src="/images/2021-02-23-Generative-Model-Part-3-Towards-Deeper-Understanding-on-Variational-Autoencoders/image_at_epoch_0010.png" alt="image_at_epoch_0010" style="zoom:100%;" />
{:refdef}


那么除了单纯的堆砌独立的隐空间维度，是否还有一种增加表达能力的方法呢？这就引出了Auxiliary Latent Variables 这种增强后验估计的方法，在原本的隐空间 $$ \mathbf{z}$$ 外增加了一个隐空间 $$ \mathbf{u}$$ 

在构造推断分布的时候，同时输出两个隐空间的联合分布：$$q_{\phi}(\mathbf{u}, \mathbf{z} \mid \mathbf{x})$$，对比原先的推断分布进一步的有分解

$$
q_{\phi}(\mathbf{u}, \mathbf{z} \mid \mathbf{x})=q_{\phi}(\mathbf{u} \mid \mathbf{x}) q_{\phi}(\mathbf{z} \mid \mathbf{u}, \mathbf{x})
\\
q_{\phi}(\mathbf{z} \mid \mathbf{x})=\int q_{\phi}(\mathbf{u}, \mathbf{z} \mid \mathbf{x}) d \mathbf{u}
$$

同时在构造生成分布的时候也是一样的做法：

$$
p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z}, \mathbf{u})=p_{\boldsymbol{\theta}}(\mathbf{u} \mid \mathbf{x}, \mathbf{z}) p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})
$$

那么由于多一个隐空间的关系，因此模型表达能力更加好，可以减少第二项 $$\mathbb{E}_{q_{\mathcal{D}}(\mathbf{x}, \mathbf{z})}\left[D_{K L}\left(q_{\mathcal{D}, \boldsymbol{\phi}}(\mathbf{u} \mid \mathbf{x}, \mathbf{z}) \| p_{\boldsymbol{\theta}}(\mathbf{u} \mid \mathbf{x}, \mathbf{z})\right)\right]$$，因此可以得出一个更紧的逼近

### 1.2. Normalizing Flows

一种方法是增加一个隐空间来增加模型的表达能力，另一种就是保证信息从数据空间映射到隐空间的过程中，尽可能的不损失，其中的极致就是完全不损失信息，或者在数学上，这种被称为


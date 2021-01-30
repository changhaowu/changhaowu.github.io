---
layout: post
title: "Generative Model Part 2：A Survey on Variational Autoencoder"
date: 2021-01-25
image: /images/cover/C_ Abstraction3.jpeg         
tags: [Generative-Model]
toc: true
published: false
---

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

<img src="/images/2021-01-25-Generative Model Part 2：Generative Model Part 2：A Survey on Variational Autoencoders.md/440px-Graph_model.svg.png" alt="440px-Graph_model.svg" style="zoom:50%;" />

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

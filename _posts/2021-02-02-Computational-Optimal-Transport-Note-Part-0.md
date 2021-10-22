---
layout: post
title: "Computational Optimal Transport Note Part 0: Introduction"
date: 2021-02-01
image: /images/cover/C_Scenery3.jpeg   
tags: [OTNotes]
toc: false
---


{: class="table-of-content"}
* TOC
{:toc}

# Computational Optimal Transport Note Part 0: Introduction

## Abstract 

在 《Optimal Transport Notes》 中主要是从理论的角度去研究最优传输问题的，而在《Computational Optimal Transport》中主要侧重于最优传输的数值方法，同时也介绍了引导设计这些算法的理论基础。

最优传输问题（Optimal Transport）是 Monge 在解决一个建城防的工程问题时提出的，在二十世纪上半叶又由 Kantorovich 等人在经济学，逻辑学方面作出了工作，在 1949 年 Dantzig 在线性规划的框架下，提出了最优传输的数值解法，而后在 1990 年代 Brenier 又进一步完善了理论框架，同时间 Earth Mover Distance 在计算机视觉上产生了应用。近年来，最优传输在机器学习领域（生成模型），图形学（shape manipulation）的应用解决了很多的问题

## Notation

- $$[n]: $$ 整数集 $$\{1, \ldots, n\}$$
- $$\mathbb{1}_{n, m}$$ ： ones(n,m)
- $$\mathbb{I}_{n}:$$ 单位 $$n \times n$$ 方阵
- $$\operatorname{diag}(u)$$ ： $$u \in \mathbb{R}^{n}$$ 的对角阵
- $$\sum_{n}$$ ：$$\mathbb{R}_{+}^{n}$$ 中的概率向量 $$e.g \quad \mathbb{R}=3: (0.1, 0.1, 0.8)$$
- $$(\mathbf{a}, \mathbf{b})$$：$$\Sigma_{n} \times \Sigma_{m}$$ 中的直方图
- $$(\alpha, \beta)$$：$$(\mathcal{X}, \mathcal{Y})$$ 上的测度
- $$\frac{\mathrm{d} \alpha}{\mathrm{d} \beta}$$：在 $$a$$ 处测度 $$\alpha$$ 相对于 $$\beta$$ 的密度
- $$\rho_{\alpha}=\frac{\mathrm{d} \alpha}{\mathrm{d} x}$$：在 $$a$$ 处测度 $$\alpha$$ 相对于勒贝格测度的密度 
- $$\left(\alpha=\sum_{i} \mathbf{a}_{i} \delta_{x_{i}}, \beta=\sum_{j} \mathbf{b}_{j} \delta_{y_{j}}\right)$$：支撑集为 $$x_{1}, \ldots, x_{n} \in \mathcal{X}$$ 和 $$y_{1}, \ldots, y_{m} \in \mathcal{Y}$$ 的离散测度
- $$c(x, y)$$：传输测度，以矩阵的形式定义 $$\mathbf{C}_{i, j}=\left(c\left(x_{i}, y_{j}\right)\right)_{i, j}$$ 支撑集为 $$\alpha, \beta$$
- $$\pi$$：$$(\alpha, \beta)$$ 之间的耦合测度，有限制：$$\forall A \subset \mathcal{X}, \pi(A \times \mathcal{Y})=\alpha(A)$$，$$\forall B \subset \mathcal{Y}, \pi(\mathcal{X} \times B)=\beta(B)$$，离散形式下有 $$\pi=\sum_{i, j} \mathbf{P}_{i, j} \delta_{\left(x_{i}, y_{j}\right)}$$ 
- $$\mathcal{U}(\alpha, \beta)$$：耦合测度的集合
- $$\mathcal{R}(c)$$：对偶势能的集合
- $$T: \mathcal{X} \rightarrow \mathcal{Y}$$：蒙日传输映射，有 $$T_{\#} \alpha=\beta$$
- $$\left(\alpha_{t}\right)_{t=0}^{1}$$：时间动态测度，$$\alpha_{t=0}=\alpha_{0} $$，$$\alpha_{t=1}=\alpha_{1}$$
- $$v$$：Brenier 公式的速度，$$J=\alpha v$$ 指代动量
- $$(f, g)$$：对偶势能
- $$(\mathbf{u}, \mathbf{v}) \stackrel{\text { def. }}{=}\left(e^{\mathbf{f} / \varepsilon}, e^{\mathbf{g} / \varepsilon}\right)$$：Sinkhorn 放缩
- $$\langle\cdot, \cdot\rangle:$$ 向量指代欧几里得空间中的点积，$$\langle A, B\rangle \stackrel{\text { def. }}{=} \operatorname{tr}\left(A^{\top} B\right)$$ 矩阵情况下是 Frobenius 范数
- $$f \oplus g(x, y) \stackrel{\text { def. }}{=} f(x)+g(y)$$：$$f: \mathcal{X} \rightarrow \mathbb{R}, g: \mathcal{Y} \rightarrow \mathbb{R}$$，$$f \oplus g: \mathcal{X} \times \mathcal{Y} \rightarrow \mathbb{R}$$
- $$\mathbf{f} \oplus \mathbf{g} \stackrel{\text { def. }}{=} \mathbf{f} \mathbb{1}_{m}^{\top}+\mathbb{1}_{n} \mathbf{g}^{\top} \in \mathbb{R}^{n \times m}$$：离散情况 
- $$\alpha \otimes \beta$$：$$\mathcal{X} \times \mathcal{Y}$$ 上的乘积测度 $$\int_{\mathcal{X} \times \mathcal{Y}} g(x, y) \mathrm{d}(\alpha \otimes \beta)(x, y) \stackrel{\text { def }}{=} \int_{\mathcal{X} \times \mathcal{Y}} g(x, y) \mathrm{d} \alpha(x) \mathrm{d} \beta(y)$$ 
- $$\mathbf{a} \otimes \mathbf{b} \stackrel{\text { def. }}{=} \mathbf{a} \mathbf{b}^{\top} \in \mathbb{R}^{n \times m}$$
- $$\mathbf{u} \odot \mathbf{v}=\left(\mathbf{u}_{i} \mathbf{v}_{i}\right) \in \mathbb{R}^{n} \text { for }(\mathbf{u}, \mathbf{v}) \in\left(\mathbb{R}^{n}\right)^{2}$$

## Reference

1. Gabriel Peyre, Marco Cuturi." [Computational Optimal Transport](https://arxiv.org/abs/1803.00567), Foundation and Trends"


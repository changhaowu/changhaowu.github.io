---
layout: post
title: "Neural ODEs"
date: 2021-12-07
image: images/cover/C_face1.png   
tags: [Numerical Computation]
toc: false
published: false

---

{: class="table-of-content"}
* TOC
{:toc}
Adjoint States Methods 在数值计算中非常的有用，其在1960年代由 Pontryagin 提出以更高效的计算梯度，在如地震学，冰川学等学科的数值模拟中广泛应用，而近来神经网络的优化问题范式同样符合 Adjoint States Methods 使用范畴，当在神经网络中使用时，其等价于 'Back Propagation'，更巧妙的是，在 NIPS 2018 best paper winner 'Neural Ordinary Differential Equations' 中，Adjoint States Methods 的应用使得梯度对于内存的计算复杂度从 O(n) 降至 O(1)，换言之，比起需要把整个网络的计算图存下来的 Back Propagation, Adjoint States Methods 把梯度下降变成了一个逐层迭代求解的问题

于是现在优化问题的定义开始，现在有输入 $x \in \mathbb{R}^{n_{x}}$，控制 $p \in \mathbb{R}^{n_{p}}$, 有优化问题：
$$
\arg_{p} \min J(x, p) \ \ \ J: \mathbb{R}^{n_{x}} \times \mathbb{R}^{n_{p}} \rightarrow \mathbb{R}
\\
s.t \ \ \ f(x, p)=0  \ \ \ f: \mathbb{R}^{n_{x}} \times \mathbb{R}^{n_{p}} \rightarrow \mathbb{R}^{n_{x}}
$$
如果以神经网络的视角来看的话，$J$ 即代表损失函数，而控制 $p$ 其实就是控制网络的参数，逐层来定义网络中的 hidden layers

为了对研究的情况做一些分类， 可以从 $J(x, p)$ 的角度来看，在 ‘16-90-computational-methods-in-aerospace-engineering’ 中， Qiqi Wang 教授定义了五种情况：

1. 解析的 $J(x, p)$，这样直接计算即可

2. 存在 intermediate state 的 $x_i$ 的 $J(x_i, p)$​, 神经网络就属于这种情况
   $$
   x_{n}=x_{n}\left(x_{n-1}, p\right)
   \\
   x_{0}=x_{0}(p)
   $$

3. $x$ 由一个ODE控制
   $$
   J(x(t), p)   \ \ \    s.t \ \frac{d x}{d t}=f(x, p)
   $$

4. $x,p$ 由一个隐函数 $f(x, p)=0$ 控制

5. $x$ 由一个PDE控制, $J(x(y,t), p) $

下文的行文结构做如此安排：通过第一种情况来叙述 Adjoint States Method 的思想，通过第二种情况来展现 Back Propagation 和 Adjoint States Method 的等价性，通过第三种情况来分析，




---
layout: post
title: "Adjoint Methods, Automatic Differentiation"
date: 2021-12-07
image: images/cover/C_face1.png   
tags: [Numerical-Computation]
toc: false
published: true
---

{: class="table-of-content"}
* TOC
{:toc}
Adjoint States Methods 在数值计算中非常的有用，其在1960年代由 Pontryagin 提出以更高效的计算梯度，在如地震学，冰川学等学科的数值模拟中广泛应用，而近来神经网络的优化问题范式同样符合 Adjoint States Methods 使用范畴，当在神经网络中使用时，他还有个更有名的名字 'Back Propagation'，更巧妙的是，在 NIPS 2018 best paper winner 'Neural Ordinary Differential Equations' 中，Adjoint States Methods 的应用使得梯度对于内存的计算复杂度从 O(n) 降至 O(1)，换言之，从要把整个网络的计算图存下来的一般 Back Propagation, Adjoint States Methods 把梯度下降变成了一个逐层迭代求解的问题

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

下文的行文结构做如此安排：通过第一种情况来叙述 Adjoint States Method 的思想，通过第二种情况来展现 Back Propagation 和 Adjoint States Method 的等价性，通过第三种情况来分析 Neural ODEs 中利用 Adjoint States Method 来减少计算内存复杂度的方法

### Analytical Function Case

通过定义拉格朗日函数 $\mathcal{L}(x, p, \lambda) \equiv J(x,p)+\lambda^{T} f(x, p)$， 同时由于 $f(x, p) = 0$, 可知 $x = x(p)$，这样可以化代价函数 $J(x,p)$ 为 $J(x(p))$

$$
\begin{aligned} 
\mathrm{d}_{p} f(x)=\mathrm{d}_{p} \mathcal{L} &=\partial_{x} J \mathrm{~d}_{p} x+\mathrm{d}_{p} \lambda^{T} f+\lambda^{T}\left(\partial_{x} f \mathrm{~d}_{p} x+\partial_{p} f\right) 
\\
&=J_{x} x_{p}+ \lambda^{T}\left(f_{x} x_{p}+f_{p}\right) \quad \text { because } f=0 \text { everywhere }
\\ 
&=\left(J_{x}+\lambda^{T} f_{x}\right) x_{p}+\lambda^{T} f_{p}  
\end{aligned}
$$

为了方便计算 $d_p f $，其中 $f_{p},J_{p}$ 都是可以直接解析计算的，而 $x_{p}$ 则需要通过整个网络迭代重新计算，为了计算梯度的便捷性，得到了拉格朗日乘子 $f_{x}^{T} \lambda=-J_{x}^{T}$，那么 $d_p J=\lambda^{T} f_p $ 

而 $\lambda = -(f_{x}^{-1})^{T}J_{x}^{T}$ ，这种共轭转置被称为 adjoint variable，这也是这样计算梯度的方法被称为 adjoint state method 的原因

### Intermediated States Case

现在有如下计算图:

{:refdef: style="text-align: center;"}
<img src="/images/2021-12-07-Adjoint-Methods-and-Auto-Differentiation/intermediate_states_graph.png" alt="intermediate_states_graph" style="zoom:40%;" />
{:refdef}

通过 BP 算法我们已经知道了 $d_p J$ 的形式了，而通过 adjoint method 可以得到同样的结果：

神经网络逐层转播的过程可以形成如下约束

$$
\left\{\begin{array}{l}
x_{0}=x_{0}(p) \\
x_{i}=x_{i}\left(x_{i-1}, p\right)
\end{array}\right.
$$

于是有拉格朗日函数 

$L(x, p, \lambda)=J\left(x_{n}\right)+\lambda_{n}^{T}\left(x_{n}-x_{n}\left(x_{n-1}, p\right)\right)+\cdots+\lambda_{1}^{\top}\left(x_{1}-x_{1}\left(x_{0}, p\right)\right)+\lambda_{0}^{\top}\left(x_{0}-x_{}\left(p\right)\right) $

对其做扰动，可以得到

$$
\begin{aligned} 
\delta L=\frac{\partial J}{\partial x_{n}} \delta x_{n} &+\lambda_{n}^{T}\left(\delta x_{n}-\frac{\partial x_{n}}{\partial x_{n-1}} \delta x_{n-1}-\frac{\partial x_{n}}{\partial p} \delta p\right) 
\\ &+\lambda_{n-1}^{T}\left(\delta x_{n-1}-\frac{\partial x_{n-1}}{\partial x_{n-2}} \delta x_{n-2}-\frac{\partial x_{n-1}}{\partial p} \delta p\right) 
\\ &+\cdots 
\\ &+\lambda_{0}^{T}\left(\delta x_{0} + \frac{d x_{0}}{d p} \delta p\right).
\end{aligned}
$$

类似最简单情况的想法，为了减少后续计算 $\frac{\partial L}{\partial p}$ 的计算量，需要把反复计算的 $\delta x_{i}$ 消去，由此得到了关于各个 $\lambda_{i}^{T}$ 的约束，代入并消去多余项后有：

$$
\frac{\partial J}{\partial p}=\sum_{i }\frac{\partial E}{\partial x_i} \frac{\partial x_i}{\partial p}
$$

### ODE controlled Case

这种对偶求解梯度的方法在特殊情况下会有意料之外的效果，在前 面两节我们证明了这种对偶方法求解的结果和原问题求解的梯度是等价的，而在这节中，借助特殊情况的约束函数 $f(x, p) = 0$，adjoint method 在特殊情况下展现出了比原问题求梯度更加优秀的数值计算性质

由于考虑到ODE的约束 $\frac{d z}{d t}=f_{\theta}(s, x, z)$，其中 $s$ 为网络深度的输入，参数 $\theta$ 为网络的控制， 则很自然的使用一个对时间积分的约束，写出拉格朗日函数：

$$
L:=J-\int_{0} \mathbf{a}^{\top}(\tau)\left[\dot{\mathbf{z}}(\tau)-f_{\theta}\left(s, \mathbf{x}, \mathbf{z}(\tau)\right)\right] \mathrm{d} \tau
$$

由于构造的拉格朗日乘子，类似之前的情况，$\mathrm{d} \mathcal{L} / \mathrm{d} \theta=\mathrm{d} \ell / \mathrm{d} \theta$，展开上式：

$$
\begin{aligned} \mathcal{L} &=\ell-\left.\mathbf{a}^{\top}(\tau) \mathbf{z}(\tau)\right|_{0} ^{S}+\int_{0}^{S}\left(\dot{\mathbf{a}}^{\top} \mathbf{z}+\mathbf{a}^{\top} f_{\theta}\right) \mathrm{d} \tau \\ &=L(\mathbf{z}(S))-\left.\mathbf{a}^{\top}(\tau) \mathbf{z}(\tau)\right|_{0} ^{S}+\int_{0}^{S}\left(\dot{\mathbf{a}}^{\top} \mathbf{z}+\mathbf{a}^{\top} f_{\theta}+l\right) \mathrm{d} \tau \end{aligned}
$$

计算上式的梯度：

$$
\begin{aligned} \frac{\mathrm{d} \ell}{\mathrm{d} \theta} &=\frac{\mathrm{d} \mathcal{L}}{\mathrm{d} \theta}=\frac{\partial L(\mathbf{z}(S))}{\partial \mathbf{z}(S)} \frac{\mathrm{d} \mathbf{z}(S)}{\mathrm{d} \theta}-\mathbf{a}^{\top}(S) \frac{\mathrm{d} \mathbf{z}(S)}{\mathrm{d} \theta}-\mathbf{a}^{\top}(0) \frac{\mathrm{d} \mathbf{z}(0)}{\mathrm{d} \theta} \\ &+\int_{0}^{S}\left[\dot{\mathbf{a}}^{\top} \frac{\mathrm{d} \mathbf{z}}{\mathrm{d} \theta}+\mathbf{a}^{\top}\left(\frac{\partial f_{\theta}}{\partial \theta}+\frac{\partial f_{\theta}}{\partial \mathbf{z}} \frac{\mathrm{d} \mathbf{z}}{\mathrm{d} \theta}+\frac{\partial f_{\theta}}{\partial \mathbf{x}} \frac{\mathrm{d} \mathbf{x}}{\mathrm{d} \theta}+\frac{\partial f_{\theta}}{\partial \tau} \frac{\mathrm{d} f}{\partial \theta}\right)+\frac{\partial l}{\partial \mathbf{z}} \frac{\mathrm{d} \mathbf{z}}{\mathrm{d} \theta}+\frac{\partial l}{\partial \tau} \frac{\mathrm{d} f}{\mathrm{~d} \theta}\right] \mathrm{d} \tau \end{aligned}
$$

整理得到了：

$$
\begin{aligned} 
\frac{\mathrm{d} \ell}{\mathrm{d} \theta} &=\left[\frac{\partial L}{\partial \mathbf{z}(S)}-\mathbf{a}^{\top}(S)\right] \frac{\mathrm{d} \mathbf{z}(S)}{\mathrm{d} \theta}+
\\ 
&+\int_{0}^{S}\left(\dot{\mathbf{a}}^{\top}+\mathbf{a}^{\top} \frac{\partial f_{\theta}}{\partial \mathbf{z}}+\frac{\partial l}{\partial \mathbf{z}}\right) \frac{\mathrm{d} \mathbf{z}}{\mathrm{d} \theta} \mathrm{d} \tau 
\\ 
&+\int_{0}^{S} \mathbf{a}^{\top} \frac{\partial f_{\theta}}{\partial \theta} \mathrm{d} \tau \end{aligned}
$$

同样的通过计算对偶问题，得到了一个关于 $\mathbf{z}(s)$ 以及控制 $\mathbf{z}(s)$ 的参数 $\theta$ 的梯度的动力系统

$$
\left\{\begin{array}{l}
\mathbf{a}^{\top}(s) = \frac{\partial L}{\partial \mathbf{z}(s)} 
\\
\dot{\mathbf{a}}^{\top}(s)=-\mathbf{a}^{\top}(s) \frac{\partial f_{\theta}}{\partial \mathbf{z}}-\frac{\partial l}{\partial \mathbf{z}} 
\\
\frac{\mathrm{d} \ell}{\mathrm{d} \theta}=\int_{0}^{S} \mathbf{a}^{\top} \frac{\partial f_{\theta}}{\partial \theta} \mathrm{d} \tau
\end{array}\right.
$$

本来问题会回到：$\frac{\mathrm{d} \ell}{\mathrm{d} \theta}=\int_{0}^{S} \mathbf{a}^{\top} \frac{\partial f_{\theta}}{\partial \theta} \mathrm{d} \tau$  再代入条件：$\mathbf{a}^{\top}(s) = \frac{\partial L}{\partial \mathbf{z}(s)} $ ，再把这个结果按照计算机的运算的情况拆成一个离散的 $[0,S]$ 区间上的求和：

$$
\frac{\mathrm{d} \ell}{\mathrm{d} \theta}=\Sigma_{p} \mathbf{a}^{\top} \frac{\partial f_{\theta}}{\partial \theta} = \Sigma_{p} \frac{\partial L}{\partial \mathbf{z}}  \frac{\partial f_{\theta}}{\partial \theta}
$$

上式是等价于 Backpropagation 的，但是由于 nerual ode 的状态是由一个动力系统控制的，则可以把梯度下降问题转化成一个 ODE solver 问题：

$$
\frac{d \mathbf{a}_{a u g}(t)}{d t}
 =-\left[\begin{array}{lll}\mathbf{a}^{\top}(t) & \mathbf{a}^{\top}_{\theta}(t) & \mathbf{a}_{t}^{\top}(t)\end{array}\right] \frac{\partial f_{a u g}}{\partial[\mathbf{z}, \theta, t]}(t)
 =-\left[\begin{array}{lll}\mathbf{a}^{\top} \frac{\partial f}{\partial \mathbf{z}} & \mathbf{a}^{\top} \frac{\partial f}{\partial \theta} & \left.\mathbf{a}^{\top} \frac{\partial f}{\partial t}\right](t)\end{array}\right.
$$

然后选取合适的 ODE solver 就得到了沿着解轨线的优化所需要的梯度，可以很清楚的看到，这样的求梯度对于内存的压力很小，与网络的深度 $t$ 无关


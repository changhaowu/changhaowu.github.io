---
layout: post
title: "Frequency Fourier and Galerkin"
date: 2021-12-25
image: images/cover/C_Street1.jpeg              
tags: [NLP]
toc: false
published: true

---

{: class="table-of-content"}
* TOC
{:toc}
## Frequency Fourier and Galerkin 

为了到达对于 Neural ODE 的更深入的理解，需要跳出残差方程的约束，把每层神经网络都看作一个算子，于是可以和 PDE 联系到一起，利用 PDE 中已有的丰富的方法和理论来辅助优化神经网络。本文的图文资料来源于 [Frequency Principle: Fourier Analysis Sheds Light on Deep Neural Networks](http://arxiv.org/abs/1901.06523) 和许志钦老师关于 Frequency Principle 的

讲座，以及  [Bruno Levy](http://alice.loria.fr/index.php/bruno-levy.html) 教授关于 [Function Space](http://www.gretsi.fr/peyresq12/documents/3-maillage4.pdf) 的讲座以及维基百科提供的详尽资料

### Low Frenquency and Function Space

#### Low Frequency basis is adequate for fitting

Generally speaking, “Frequency" in pixels image corresponds to the rate of change of intensity across neighbouring pixels. 由此为基础，把图片通过傅立叶变换迁移到频域后滤掉高频部分来降噪等得到了理论保证，如下图所示

{:refdef: style="text-align: center;"}
<img src="/images/2021-12-25-Frequency-Fourier-Galerkin/Image_frequency.png" alt="Image_frequency" style="zoom:30%;" />
{:refdef}

左图只有一种颜色，所有的像素之间没有变化，因此只有低频信号，然而从自然中拍摄的右图中，为了表示各物体之间的差别，物体的边缘存在着edge，各个物体的颜色也不尽相同，像素之间存在着变化，则最后图像中就有比较高频的部分，而其中最极端的情况则为在一张白纸上，直接用黑笔画一条竖线，就会产生最高频的情况：

{:refdef: style="text-align: center;"}
<img src="/images/2021-12-25-Frequency-Fourier-Galerkin/black_line_in_white_paper.png" alt="black_line_in_white_paper" style="zoom:35%;" />
{:refdef}

很自然的可以想到，高频的部分中有很多的噪声，同时低频的部分噪声则相对较少，如果从函数拟合的角度去看高频低频部分，则会有更贴近神经网络的结论出现：

{:refdef: style="text-align: center;"}
<img src="/images/2021-12-25-Frequency-Fourier-Galerkin/spatial_domain.gif" alt="fourier_domain" style="zoom:80%;" />
{:refdef}

上图是按照训练时间的先后顺序来展示的，因此当要去拟合一个函数的时候，首先学习到的信号总是低频的，这些信号学习的是要去拟合函数的 landscape，然后再是到的 detail 部分, landscape 和 detail，这不就恰恰对应着之前的图片中低频信号和高频信号的关系嘛。

{:refdef: style="text-align: center;"}
<img src="/images/2021-12-25-Frequency-Fourier-Galerkin/fourier_domain.gif" alt="fourier_domain" style="zoom:80%;" />
{:refdef} 

在上图傅立叶域上的可视化更好的展现了这个结论，因此神经网络学习时，首先学到的是低频的信号，然后再是高频的信号，而之前从图片里得到的结论是，高频信号往往对应着噪声，那么函数拟合中随着训练的深入，得到的过拟合的情况是否能理解成，拟合器过度学习了样本集中的噪声，导致了泛化性能的下降呢？

{:refdef: style="text-align: center;"}
<img src="/images/2021-12-25-Frequency-Fourier-Galerkin/Typical-relationship-for-capacity-and-generalization-error.jpeg" alt="Typical-relationship-for-capacity-and-generalization-error" style="zoom:60%;" />
{:refdef} 

那么这提供了一种思想，即当我们想利用傅立叶基去拟合函数的时候，除了本身计算能力有限导致的妥协，需要把无限的函数拟合问题转化成有限维的情况来做，同时本身这样做就是合理的，因为这避免了泛化误差的提升

#### What's function space

一般的向量空间的例子，不妨就用欧几里得空间好了，对一个三维的欧式空间，其中任意的向量能由基表示：

$$
\begin{aligned}
&V=x e_{1}+y e_{2}+z e_{3} \\
&x=V \cdot e_{1} \\
&y=V \cdot e_{2} \\
&z=V \cdot e_{3}
\end{aligned}
$$

其中的 $\cdot$ 运算为内积  $V \cdot W=V_{x} W_{x}+V_{y} W_{y}+V_{z} W_{z}$

内积如此定义，有一个更物理意义上好的效果，如果现在有一组两个基 $\{e_{1},e_{2}\}$张成了一个二维的欧式空间，同时有一个三维的向量 $v$ ，利用投影得到 $\{e_{1},e_{2}\}$ 对其的最佳逼近 $W=\left(V \cdot e_{1}\right) e_{1}+\left(V \cdot e_{2}\right) e_{2}$。通过内积可以定义向量空间上的投影

{:refdef: style="text-align: center;"}
<img src="/images/2021-12-25-Frequency-Fourier-Galerkin/projection_space.png" alt="projection_space" style="zoom:25%;" />
{:refdef} 

一般的向量空间的理解很直观，那么把研究的对象换成函数，则问题就变成了：

- 如何定义函数空间中的基
- 如何定义函数空间中的内积

第一个问题研究有很多思路了，比如多项式基，以及之前定义的傅立叶基，问题在于如何定义函数空间中的内积，借助 $\delta$ 函数作为基 ，向量可以表示为 $u = \sum_i u_i \delta_i$

$$
u \cdot v=\sum u_{i} v_{i}
$$

{:refdef: style="text-align: center;"}
<img src="/images/2021-12-25-Frequency-Fourier-Galerkin/dot_product_vector.png" alt="dot_product_vector" style="zoom:30%;" />
{:refdef} 

则对应到函数的情况则为：

$$
f \cdot g=\int f(t) g(t) d t
$$

{:refdef: style="text-align: center;"}
<img src="/images/2021-12-25-Frequency-Fourier-Galerkin/inner_product_function.png" alt="inner_product_function" style="zoom:30%;" />
{:refdef} 

那么傅立叶基就是把函数在如下基上做投影 $f(x)=\Sigma \alpha_{i} \phi_{i}(x)$：

$$
\begin{aligned}
&\phi_{0}(x)=1 \\
&\phi_{2 k}(x)=\sin (2 k \pi x) \\
&\phi_{2 k+1}(x)=\cos (2 k \pi x)
\end{aligned}
$$

### Fourier Transform and Application

由傅立叶基的特性发现，不同频率的傅立叶基之间是两两正交的，$\phi_{i}(x) \cdot \phi_{j}(x) = \delta_{ij}$，这让傅立叶基是一组正交基。这一良好的性质使傅立叶变换的可视化意义非常的好，傅立叶变换把时域上的信号转化到频域上，定义方法是

$$
\hat{f}(\xi)=\int_{-\infty}^{\infty} f(x) e^{-2 \pi i t \xi} d t
$$

如果引入复数的话，由于 $\cos \varphi=\frac{e^{i \varphi}+e^{-i \varphi}}{2}, \quad \sin \varphi=\frac{e^{i \varphi}-e^{-i \varphi}}{2 i}$，则傅立叶基的表述会有一个更优美的表述：

$$
f(t)=\sum_{n=-\infty}^{\infty} c_{n} e^{i 2 \pi n t}
\\
c_{n}=\frac{a_{n}-i b_{n}}{2}, \quad c_{-n}=\frac{a_{n}+i b_{n}}{2}
$$

在复变函数上，$\langle f, g\rangle=\int_{a}^{b} f(t) \overline{g(t)} \mathrm{d} t$ 内积的定义需要稍加修正，由于傅立叶基之间是正交的，这样积分的结果就是把每个基上对应需要的系数滤出来，如下图所示：

{:refdef: style="text-align: center;"}
<img src="/images/2021-12-25-Frequency-Fourier-Galerkin/Fourier_transform_time_and_frequency_domains_(small).gif" alt="Fourier_transform_time_and_frequency_domains_(small)" style="zoom:100%;" />
{:refdef} 

具体把上图的展开写开的话，$f(t)=\sum_{k=-\infty}^{\infty} a_{k} e^{j k \omega_{0} t}$ 是一个傅立叶基表示的展开：
$$
F(\omega)=\sum_{k=-\infty}^{\infty} a_{k} \int_{-\infty}^{\infty} e^{j\left(k \omega_{0}-\omega\right) t} d t=2 \pi \sum_{k=-\infty}^{\infty} a_{k} \delta\left(\omega-k \omega_{0}\right)
$$
{:refdef: style="text-align: center;"}
<img src="/images/2021-12-25-Frequency-Fourier-Galerkin/fourier_transform_fourier_series.png" alt="fourier_transform_fourier_series" style="zoom:10%;" />
{:refdef} 

对于一类特殊定义的变换，如傅立叶变换，拉普拉斯变换等，具有一个很特殊的定理，函数卷积的傅立叶变换是函数傅立叶变换的乘积：
$$
\mathcal{F}\{f * g\}=\mathcal{F}\{f\} \cdot \mathcal{F}\{g\}
$$
这样可以简化卷积的运算量，对于长度为 $ m $ 的序列，按照卷积的定义进行计算，需要做 $2 n-1$ 组对位乘法，其计算复杂度为 $\mathcal {O}(n^{2})$ ；而利用傅里叶变换将序列变换到频域上后，只需要一组对位乘法，利用傅里叶变换的快速算法之后，总的计算复杂度为 $\mathcal {O}(n\log n)$

### Galerkin basis, Road to application in Neural Network?





## Reference


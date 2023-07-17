---
layout: post
title: "Geometric Deep Learning Notes"
date: 2021-07-05
image: images/cover/C_Street8.JPG               
tags: [DL-Theory]
toc: false
published: false

---

{: class="table-of-content"}
* TOC
{:toc}
由 Erlangen Programme 可以知道，几何的研究可以归结为对于各种不变量的研究，比如欧几里得几何中，以长度和角度在变换中作为不变量，欧几里得几何中的平移和旋转不会改变这些性质，而在仿射变换中，平行的性质得到了保持，继而可以得到下面的关系：
$$
\text { 刚体变换群 } \triangleleft \text { 等距变换群 } \triangleleft \text { 共形变换群 } \triangleleft \text { 拓扑同胚群. }
$$
这样的框架非常的优美，而比起 Erlangen Program 这样的框架，对于神经网络的研究则显得没有一条主线，这可能会导致重复无效的研究，在 ‘Geometric Deep Learning Grids, Groups, Graphs, Geodesics, and Gauges’ 一文中，Bronstein 教授提出了利用几何不变量来研究神经网络，继而得到一个好的框架


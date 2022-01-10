---
layout: post
title: "2022-01-07-Nabla-Laplacian"
date: 2022-01-07
image: images/cover/C_Street8.JPG               
tags: [NLP]
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

  @tf.function
  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=True)


```

#### 

{:refdef: style="text-align: center;"}
<img src="/images/2021-02-02-Computational-Optimal-Transport-Note-Part-1/Transport_Map_Visualization.png" alt="Transport_Map_Visualization" style="zoom:40%;" />
{:refdef}



## Reference

[1] C.-C. Jay Kuo [Understanding Convolutional Neural Networks with A Mathematical Model](https://arxiv.org/abs/1609.04112) 

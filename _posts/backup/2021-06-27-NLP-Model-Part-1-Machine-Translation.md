---
layout: post
title: "Natural Language Process Model Part 1:Neural Machine Translation"
date: 2021-06-27
image: images/cover/C_Object1.JPG               
tags: [NLP]
toc: false
published: false

---

{: class="table-of-content"}
* TOC
{:toc}


# Natural Language Process Model Part 1:Neural Machine Translation

考虑到神经网络的拟合能力，如果能用深度学习（统计方法）去进行机器翻译的学习是一种很有诱惑力的想法，也确实有人如此做了，当然通过神经网络方法去做机器翻译的发展都是有一个过程的，从最简单的RNN encoder-decoder模型，到后来的transformer模型，这样的发展符合人类对于一件事情认知 $$\rightarrow$$ 假说 $$\rightarrow$$ 实验 $$\rightarrow$$ 超越的循序渐进的过程

无障碍的沟通是非常重要的事情，如果能设想一个人类之间沟通无障碍的未来的话，人类之间的隔阂会变小，就像传说中的巴别塔一般，无所不能的NT会诞生吧 X)

<img src="/images/2021-06-27-NLP-Model-Part-1-Machine-Translation/1920px-Pieter_Bruegel_the_Elder_-_The_Tower_of_Babel_(Vienna)_-_Google_Art_Project.jpeg" alt="1920px-Pieter_Bruegel_the_Elder_-_The_Tower_of_Babel_(Vienna)_-_Google_Art_Project" style="zoom:30%;" />

4 他们说，“来吧，我们要建造一座城和一座塔，塔顶通天，为了扬我们的名，免得我们被分散到世界各地。”

5 但是耶和华降临看到了世人所建造的城和塔。

6 耶和华说，“看哪，他们都是一样的人，说着同一种语言，如今他们既然能做起这事，以后他们想要做的事就没有不成功的了。”

7 让我们下去，在那里打乱他们的语言，让他们不能知晓别人的意思。

8 于是耶和华使他们分散到了世界各地，他们也就停止建造那座城。

9 因为耶和华在那里打乱了天下人的言语，使众人分散到了世界各地，所以那座城名叫巴别。

——创世记11:4–9

## Intuition：Neroscience shows strong evidence of the existence of Interlingua

当然，即使是最开始使用的RNN encoder-decoder结构也并非无源之水，而是源自于对于真正人脑神经网络在进行翻译过程的一项观察：人去进行翻译，比如法语到英语，翻译者在翻译过程执行的所做的，可以背归纳为以下的行为

1. 从原语言中把意思进行解码到到一个中间语言 （Decoding the meaning of the source text）
2. 再从中间语言将意味编码的目标语言 （and Re-encoding this meaning in the target language）

当然说起来很简单，为了保证意思不失真，同时又符合句法等限制，这个过程远比描述起来要复杂，

## Reference

[1] https://en.wikipedia.org/wiki/Tower_of_Babel

​	


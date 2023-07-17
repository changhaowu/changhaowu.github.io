---
layout: post
title: "Markov Decision Process in Reinforcement Learning"
date: 2020-07-03
image: images/cover/F5.jpg
tags: [Reinforcement-Learning]
toc: false
published: false
---

{: class="table-of-content"}
* TOC
{:toc}

## 引言

在这篇文章中，主要讨论利用马尔可夫过程中的一种特例，马尔科夫决策过程（markov decision process）在强化学习（Reinforcement learning）中的应用。马尔可夫决策过程是一类特殊的扩充的马尔可夫链，其马尔可夫性切合了生活中许多可以通过强化学习解决的问题，因此提供了去解一大类问题的范式。

这篇文章中将会围绕马尔可夫决策过程去研究其模型如何建立，马尔可夫决策过程建立的模型的一些指标以及相应的性质，最后如何去提供一种相应的解法来解决强化学习问题。

## 马尔可夫决策过程

马尔可夫决策过程其实是马尔可夫链的一类拓展，其特殊之处在于添加了行为（action）和回报（reward），action使得马尔科夫决策过程有机会选择而reward使得马尔科夫决策过程有“动机”，当每个状态时的行为只有一种的时候，并且没有所谓的回报之说的时候，马尔可夫决策过程就退化回了马尔可夫链。

### 智能体-环境交互

{:refdef: style="text-align: center;"}
<img src="/images/2020-07-03-Markov-Decision-Process-in-Reinforcement-Learning/Screen Shot 2020-06-26 at 4.52.15 PM.png" style="zoom:20%;" />
{: refdef}

Agent（智能体/模型）在（Environment）环境中进行行为.  环境是基于其模型给出反应的，当然模型会有可知和不可知之分，可知的情况下，其实就没有随机性了，问题可能就可以转化成规划问题，（当然也会有魔方这种环境完全可知，但是仍然不易解决的模型，但是这部分的问题本文不做讨论），而Markov Chain这种离散时间的问题结构，就变成了动态规划问题。

但是往往，由于环境提供的不完全的信息，环境的模型是不可知的，只能通过实验知道特定决策的结果。就比如下图这样的画风：

{:refdef: style="text-align: center;"}
<img src="/images/2020-07-03-Markov-Decision-Process-in-Reinforcement-Learning/Screen Shot 2020-06-26 at 5.07.10 PM.png" style="zoom:28%;" />
{: refdef}

归纳成比较数学的语言来说，就是智能体会处于状态 $$s$$ 下($$s \in \mathcal{S}$$) ，$$s$$ 是状态空间中的一个子集，然后在该状态 $$s$$ 下，智能体去选择一个  $$a$$ 动作 ($$a \in \mathcal{A}$$) ，是的自己从现在的状态转移到下一个状态。当然转移的过程有时也是随机的，建立在转移概率矩阵 ($$P$$) 上。而一旦一个行为完成, 环境会反馈出一个 奖励 $$r$$ ($$r \in \mathcal{R}$$) 

### 马尔可夫决策过程的定义

马尔可夫决策过程的空间定义成为四元组 $$<S,A,P,R>$$ ，以及衰减因子 $$\gamma$$

- $$\mathcal{S}$$ - 状态空间;
- $$\mathcal{A}$$ - 行为空间;
- $$P$$ - 转移概率;
- $$R$$ - 回报函数;
- $$\gamma$$ - 衰减因子，一般用来在 $$P$$ 和 $$R$$ 不完全的情况下，使模型更看重短期之内的收益的决策，而在现实问题中，$$P$$ 和 $$R$$ 的信息也的确一般不完全。

马尔可夫决策过程的目标，就是去最大化其环境反馈的收益，比如围棋赢了，那么就获得收益 1，该目标定义成模型的长期回报总和 $$ G_t=R_{t}+R_{t+1}+...+R_{T} $$,当然这是在有终止时间状态 $$T$$ 的情况下，当然 $$T=\infty$$ 的情况下，可以拓展成无限级数，然后之间定义的衰减因子，此时起到了一种调节的作用：
​        

$$ G_t=R_{t}+\gamma R_{t+1}+...+\gamma^{T-1}R_{T}+...=\sum_{k=0}^{\infty}\gamma^kR_{t+k+1} $$

- 衰减因子 $$\gamma$$ （ $$0\leq \gamma\geq1$$）的存在是的长期收益不会在数值上发散
- 衰减因子的存在使得长期收益和短期收益之间更容易做出选择眼前就可以拿到的短期收益，尤其当 $$\gamma=0$$ 时，系统会完全变成短视，即只顾下一个时间 $$t+1$$ 的收益

从信息的角度来说，从当前的状态可以从环境中获取的为了规划未来的行动的信息量和从过去到现在从环境中的获取的为了规划未来的行动的信息量是一样的，这样的随机过程的性质被称为Markov性质，这样和教材上定义的单步齐次马氏链的性质是一致的，比较正式的来说就是：

本来为了规划在 $$t$$ 时刻为了规划 $$t+1$$ 的行动，所需的应当是过去所有积累的信息量即： 

$$ \operatorname{Pr}\{R_{t+1}=r, S_{t+1}\mid S,A,R\} = \operatorname{Pr}\left\{R_{t+1}=r, S_{t+1}=s^{\prime} \mid S_{0}, A_{0}, R_{1}, \ldots,  R_{t}, S_{t}, A_{t}\right\} $$

但是其等价于：

$$ \operatorname{Pr}\{R_{t+1}=r, S_{t+1}\mid S,A,R\} = \operatorname{Pr}\left\{R_{t+1}=r, S_{t+1}=s^{\prime} \mid  S_{t}, A_{t}\right\} $$

这样的话，下一步的状态只需要用当前状态即可进行建模，这样定义的好处，即是有了现成的随机过程的范式可以直接利用，其次，很多生活中遇到的实际问题，确实是有Markov性的，比如围棋，后面的布局只需要看现在棋盘上的棋局，与过去的无关。

马尔可夫性的问题使得模型的状态表征的信息冗余度很小，并且下一状态只需要根据当前状态去建模的性质，使得模型的复杂度也不会过快的上升。

### 模型&性质

当给出了智能体在 $$(s, a, s', r)$$ 状态下的情况后，我们可以做如下定义：

- 转移概率：

  $$ P(s', r \vert s, a) = \mathbb{P} [S_{t+1} = s', R_{t+1} = r \vert S_t = s, A_t = a] $$

- 状态转移函数：

  $$ P_{ss'}^a = P(s' \vert s, a) = \mathbb{P} [S_{t+1} = s' \vert S_t = s, A_t = a] = \sum_{r \in \mathcal{R}} P(s', r \vert s, a) $$

- 期望收益：

$$ R(s, a) = \mathbb{E} [R_{t+1} \vert S_t = s, A_t = a] = \sum_{r\in\mathcal{R}} r \sum_{s' \in \mathcal{S}} P(s', r \vert s, a) $$

当有了状态 $$s$$ 后，智能体会根据某种准则选择后面的行为，即决策准则（policy），这是一个映射 $$s \rightarrow a$$

$$ \pi(a \vert s) = \mathbb{P}_\pi [A=a \vert S=s] $$

再给出了决策后，就可以定义想应的状态 $$s$$ 的价值（$$G_t$$ 采用之前的定义）：

$$ V_{\pi}(s) = \mathbb{E}_{\pi}[G_t \vert S_t = s]=\mathbb{E}_{\pi}[\sum_{k=0}^{\infty}\gamma^kR_{t+k+1} \vert S_t = s] $$

将动作也纳入到模型中就有动作价值函数

$$ Q_{\pi}(s, a) = \mathbb{E}_{\pi}[G_t \vert S_t = s, A_t = a]  $$

$$V_\pi$$ 和 $$Q_\pi$$ 是能够通过估计得出的，比如可以通过门特卡罗方法估计出的值，其是收敛到 $$V_\pi(s)$$，当然对于大量的情况，智能体就会通过参数估计一个价值函数 （值得一提的是，围棋的价值函数应当是非常不连续，数值上的性质很差，例证就是围棋，往往会有所谓的一招走错，满盘皆输的棋局）	

当然，也可以计算承继的一步的状态价值函数和此时的value的关系:

$$ \begin{aligned} V(s) &= \mathbb{E}[G_t \vert S_t = s] \\ &= \mathbb{E} [R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots \vert S_t = s] \\ &= \mathbb{E} [R_{t+1} + \gamma (R_{t+2} + \gamma R_{t+3} + \dots) \vert S_t = s] \\ &= \mathbb{E} [R_{t+1} + \gamma G_{t+1} \vert S_t = s] \\ &= \mathbb{E} [R_{t+1} + \gamma V(S_{t+1}) \vert S_t = s] \end{aligned} $$

对于动作价值函数，一样可以做类似的处理：

$$ \begin{aligned} Q(s, a) &= \mathbb{E} [R_{t+1} + \gamma V(S_{t+1}) \mid S_t = s, A_t = a] \\ &= \mathbb{E} [R_{t+1} + \gamma \mathbb{E}{a\sim\pi} Q(S{t+1}, a) \mid S_t = s, A_t = a] \end{aligned} $$

然后就可以导出Bell公式：

$$ \begin{aligned} V_{\pi}(s) &= \sum_{a \in \mathcal{A}} \pi(a \vert s) Q_{\pi}(s, a) \ Q_{\pi}(s, a) \\&= R(s, a) + \gamma \sum_{s' \in \mathcal{S}} P_{ss'}^a V_{\pi} (s') \ V_{\pi}(s) \\&= \sum_{a \in \mathcal{A}} \pi(a \vert s) \big( R(s, a) + \gamma \sum_{s' \in \mathcal{S}} P_{ss'}^a V_{\pi} (s') \big) \ Q_{\pi}(s, a)\\ &= R(s, a) + \gamma \sum_{s' \in \mathcal{S}} P_{ss'}^a \sum_{a' \in \mathcal{A}} \pi(a' \vert s') Q_{\pi} (s', a') \end{aligned} $$

{:refdef: style="text-align: center;"}
<img src="/images/2020-07-03-Markov-Decision-Process-in-Reinforcement-Learning/Screen Shot 2020-06-26 at 11.56.55 PM.png" style="zoom:32%;" />
{: refdef}

### 模型的求解

那么以上定义完成了以后，马尔可夫决策问题就等价于：在时刻 $$t$$ , 以及状态 $$s$$ 下，在决策中去找一个一个策略 $$\pi$$，使得：
​
$$ V_{*}(s)=\max_{\pi}V_{\pi}(s) \;\;\; \forall s \in S $$

那么如果找得到这个最优策略的话：

$$ \begin{aligned} V_{*}(s) &=\max _{a \in \mathcal{A}(s)} q_{\pi_{*}}(s, a) \\
&=\max _{a} \mathbb{E}_{\pi^{*}}\left[G_{t} \mid S_{t}=s, A_{t}=a\right] \\
&=\max _{a} \mathbb{E}_{\pi^{*}}\left[\sum_{k=0}^{\infty} \gamma^{k} R_{t+k+1} \mid S_{t}=s, A_{t}=a\right] \\
&=\max _{a} \mathbb{E}_{\pi^{*}}\left[R_{t+1}+\gamma \sum_{k=0}^{\infty} \gamma^{k} R_{t+k+2} \mid S_{t}=s, A_{t}=a\right.\\
&=\max _{a} \mathbb{E}\left[R_{t+1}+\gamma V_{*}\left(S_{t+1}\right) \mid S_{t}=s, A_{t}=a\right] \\
&=\max _{a \in \mathcal{A}(s)} \sum_{s^{\prime}, r} p\left(s^{\prime}, r \mid s, a\right)\left[r+\gamma V_{*}\left(s^{\prime}\right)\right] \end{aligned} $$

{:refdef: style="text-align: center;"}
<img src="/images/2020-07-03-Markov-Decision-Process-in-Reinforcement-Learning/Screen Shot 2020-06-27 at 12.30.06 AM.png" style="zoom:32%;" />
{: refdef}

## 价值函数估计方法

之前给出了马尔可夫决策过程建模的RL问题的范式，但是实际上，受制于不完全的信息，智能体需要对每步的 $V(s)$ 和 $ Q(s)$ 做估计。不完全的来说，有如下几种方法：

### 蒙特卡洛方法

蒙特卡洛方法的想法很简单，不需要去将环境建模，而是通过实验返回的值去逼近实际的 $$V(s)$$ 和 $$Q(s)$$，实际为了计算估计的 $$G_t$$，蒙特卡洛方法需要去遍历 $$S_1, A_1, R_2, \dots, S_T$$ 来计算 $$G_t = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}$$

推断的 $$V(s)$$ 为：

$$V(s) = \frac{\sum_{t=1}^T \mathbb{1}[S_t = s] G_t}{\sum_{t=1}^T \mathbb{1}[S_t = s]}  $$

推断的 $$Q(s)$$ 为：

$$ Q(s, a) = \frac{\sum_{t=1}^T \mathbb{1}[S_t = s, A_t = a] G_t}{\sum_{t=1}^T \mathbb{1}[S_t = s, A_t = a]} $$

由于蒙特卡洛方法的特点，原则上每个状态和动作应当被计算复数次来减小估计的误差，下图为蒙特卡洛方法估计出的 Black jack 的价值函数：

{:refdef: style="text-align: center;"}
<img src="/images/2020-07-03-Markov-Decision-Process-in-Reinforcement-Learning/Screen Shot 2020-06-27 at 1.32.46 AM.png" style="zoom:33%;" />
{: refdef}

具体的算法一般采用 $$\pi$$ 和 $$Q$$ 互相增益 $$Improvement-Evaluation$$ （EM?）的方式：

1. 用当前的价值函数估计最好的策略

$$ \pi(s) = \arg\max_{a \in \mathcal{A}} Q(s, a) $$

2. 在该策略下生成新的采样

3. 用新的采样去计算相应的价值函数：

$$ q_\pi(s, a) = \frac{\sum_{t=1}^T \big( \mathbb{1}[S_t = s, A_t = a] \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1} \big)}{\sum_{t=1}^T \mathbb{1}[S_t = s, A_t = a]} $$

{:refdef: style="text-align: center;"}
<img src="/images/2020-07-03-Markov-Decision-Process-in-Reinforcement-Learning/Screen Shot 2020-06-27 at 1.44.20 AM.png" style="zoom:33%;" />
{: refdef}

### TD学习

TD Learning (Temporal-Difference Learning) 是一种综合了动态规划和蒙特卡洛方法优点的方法，即可以利用经验学习，也可以如同动态规划一样，利用之前得到的结论，TD学习不需要如同蒙特卡洛方法一样，探索到 $$T$$

最简单的TD学习的迭代形式就是：​

$$ V\left(S_{t}\right) \leftarrow V\left(S_{t}\right)+\alpha\left[R_{t+1}+\gamma V\left(S_{t+1}\right)-V\left(S_{t}\right)\right]  $$

TD学习的目标就是 $$R_{t+1} + \gamma V(S_{t+1}$$)，而对应的，其动作效应函数：

$$ Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha (R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t))   $$

## 参考文献

[1]  Finite Markov Decision Process   https://en.wikipedia.org/wiki/Markov_decision_process

[2] Temporal difference learning https://en.wikipedia.org/wiki/Temporal_difference_learning 

[2] Q-learning [https://en.wikipedia.org/wiki/Q-learning](Q-learning)

[3] Lil-log Blog [A-long-peek-into-reinforcement-learning](a-long-peek-into-reinforcement-learning)

[4] Richard S. Sutton and Andrew G. Barto Second edition, in progress [Reinforcement Learning:An Introduction](http://incompleteideas.net/book/bookdraft2017nov5.pdf) 

[5] Martijn van Otterlo and Marco Wiering [Reinforcement Learning and Markov Decision Processes](https://www.ai.rug.nl/~mwiering/Intro_RLBOOK.pdf)

[6] [Intro to Reinforcement Learning](https://www.youtube.com/watch?v=IkEF4LpH5Ys&list=PLySQw_vQ73PyDY68KF0HdCzcILBoHVTvD)   Bolei Zhou


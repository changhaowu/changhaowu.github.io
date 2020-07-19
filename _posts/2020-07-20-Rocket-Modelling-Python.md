---
layout: post
title: "Python火箭模型建立"
date: 2020-07-11
image: images/cover/A3.jpg
tags: [Numerical Computation]
toc: true

---
#### 符号

| Notation                    | Meaning                       |
| --------------------------- | ----------------------------- |
| $$ t \quad$$                | time                          |
| $$y$$                       | altitude                      |
| $$v \quad$$                 | velocity, positive upwards    |
| $$F \quad$$                 | total force, positive upwards |
| $$D \quad$$                 | aerodynamic drag              |
| $$T \quad$$                 | propulsive thrust             |
| $$\Delta t \quad$$          | time step                     |
| $$\rho$$                    | air density                   |
| $$g$$                       | gravitational acceleration    |
| $$m$$                       | mass                          |
| $$C_{D}\quad$$              | drag coefficient              |
| $$A \quad$$                 | drag reference area           |
| $$\dot{m}_{\text {fuel }}$$ | fuel mass flow rate           |
| $$u_{e} \quad$$             | exhaust velocity              |
| $$i$$                       | time index                    |

#### 火箭模型

自己去运算，甚至搭建一个火箭是一件很有意思的事情。无论是哪一种火箭，其受力基本上都是三种，火箭发动机产生的推力，火箭在大气中收到的阻力，以及火箭附近的星体对其的引力：

{:refdef: style="text-align: center;"}
<img src="/images/2020-07-20-Rocket-Modelling-Python/rktfor.gif" alt="rktfor" style="zoom:80%;" />
{:refdef}


#### 模型火箭模型

最简单的系统，就是不考虑地心地固坐标系的情况下，讲地面考虑成平地，同时忽略质量的瞬时变动，继续使用牛顿第二系统去建立模型，只能适用于小型模型火箭，往往不会进入地球轨道，同时其极限高度也比较小，往往被限制在对流层内，因此阻力中的大气情况比较好估算。

{:refdef: style="text-align: center;"}
<img src="/images/2020-07-20-Rocket-Modelling-Python/rktflight.gif" alt="rktflight" style="zoom:80%;" />
{:refdef}

建立其动力模型：

$$
\begin{array}{l}
\left\{\begin{array}{l}
y^{\prime}=v \cos (\theta) \\
x^{\prime}=v \sin (\theta) \\
m \vec{v}^{\prime}=\vec{T}-\vec{D}-\vec{G} \\
m v_{y}^{\prime}=(T-D)\cos(\theta)-G\\
m v_{x}^{\prime}=(T-D)\sin(\theta) \\
D=\rho c_D A v^2 \\
m^{\prime}=-{m}_{f u e l}^{\prime} \\
T = m^{\prime}_{f u e l} u_{e} \\
m^{\prime}_{f u e l} = c \\
G = mg\frac{R_e^2}{(y+R_e)^2}
\end{array}\right. \\
\end{array}
$$

整理后得出：

$$
\begin{array}{l}
\left\{\begin{array}{l}
y^{\prime}=v_y  \\
x^{\prime}=v_x  \\
v^{\prime}_{y}= (- \frac{m^{\prime}}{m} u_{e} - \frac{\rho c_D A}{m}v^2)cos(\theta) - g\frac{R_e^2}{(y+R_e)^2} \\
v^{\prime}_{x}= (- \frac{m^{\prime}}{m} u_{e} - \frac{\rho c_D A}{m}v^2)sin(\theta) \\
{\theta}^{\prime} = {\arctan(\frac{v_x}{v_y})}^{\prime} \\
m^{\prime}= -c \\
\end{array}\right. \\
\end{array}
$$

到达最高点当：$$ \frac{d \theta}{d t}=0$$

#### 轨道火箭模型

如果要考虑到轨道火箭甚至向火星发射的火箭，由于其入轨后在地表的投影，距离发射基地已经有相当的距离，不能把地球处理成平地，而需要以地心为中心，建立地心地固坐标系（这部分以NASA在1998年发射的火星气象卫星作为图例）

{:refdef: style="text-align: center;"}
<img src="/images/2020-07-20-Rocket-Modelling-Python/launch2.gif" alt="launch2" style="zoom:80%;" />
{:refdef}

气象火箭在SECO-1阶段大致进入轨道，而在地图上的体现如下图，为状态1到状态2:

{:refdef: style="text-align: center;"}
<img src="/images/2020-07-20-Rocket-Modelling-Python/msp98track93.gif" alt="msp98track93" style="zoom:60%;" />
{:refdef}

为此可视化的得出，需要采用如地心地固坐标系这样的处理：

{:refdef: style="text-align: center;"}
<img src="/images/2020-07-20-Rocket-Modelling-Python/ECEF.png" alt="ECEF" style="zoom:55%;" />
{:refdef}

为了方便处理，暂时不考虑地球各点微弱的重力差异，就将地球处理成一个完美球形，地表 $$g$$ 值就取 $$9.80665m/s$$

{:refdef: style="text-align: center;"}
<img src="/images/2020-07-20-Rocket-Modelling-Python/ECEF.JPG" alt="ECEF" style="zoom:15%;" />
{:refdef}

建立其对应的模型：

$$
\begin{array}{l}
\left\{\begin{array}{l}
\tan \theta=\frac{R d \rho}{d R}\Rightarrow R \frac{\varphi^{\prime}}{R^{\prime}}=\frac{\varphi^{\prime}}{\ln R^{\prime}} \\
d v=\frac{R d\varphi}{\sin \theta} \Rightarrow v^{\prime} \sin \theta=R \varphi^{\prime} \\
v^{\prime}\cos \theta=R^{\prime} \\
m \nu^{\prime}=F=T-D-G \\
D=\rho c_D A v^2 \\
m=-\dot{m}_{f u e l} \\
T=\dot{m}_{f u e l} u_{e} \\
G = mg\frac{R_e^2}{R^2} \\

\end{array}\right. \\
\end{array}
$$

当考虑到轨道火箭喷出的气体速度较大的时候，牛顿第二定理不再适用，应当考虑动量定理，因此修正方程：

$$
\begin{array}{l}
\left\{\begin{array}{l}
\tan \theta=\frac{R d \rho}{d R}\Rightarrow R \frac{\varphi^{\prime}}{R^{\prime}}=\frac{\varphi^{\prime}}{\ln R^{\prime}} \\
d v=\frac{R d\varphi}{\sin \theta} \Rightarrow v^{\prime} \sin \theta=R \varphi^{\prime} \\
v^{\prime}\cos \theta=R^{\prime} \\
\begin{array}{l}
m d v-d m_{f u e l}=(T-D-G) d t  \Rightarrow 
m v^{\prime}-m^{\prime} u_{e}=T-D-G
\end{array} \\
D=\rho c_D A v^2 \\
m=-\dot{m}_{f u e l} \\
T=\dot{m}_{f u e l} u_{e} \\
G = mg\frac{R_e^2}{R^2}
\end{array}\right. \\
\end{array}
$$




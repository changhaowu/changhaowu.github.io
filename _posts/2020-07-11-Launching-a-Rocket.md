---
layout: post
title: "Launch A Rocket"
date: 2020-07-11
image: images/cover/A3.jpg
tags: [Numerical Computation]
toc: true

---

## Launch A Rocket

​	Inspired by my senior Npa, I determined to make a post about "How to launch a rocket ". As we all know, rocket will be sent to a cetatin orbit where it is supposed to serve as a communication satellite. The "Starlink" Project even proposed to launch a chain to provide more stable communication service . 

​	How can we make a computational approxmation to the trajectory of a rocket given some initial setting? This post will build a series of progressive models and use julia to make a computational approximation on the models

#### Notation

| Notation                    | Meaning                       |
| --------------------------- | ----------------------------- |
| $$ t \quad$$                | time                          |
| $$h \quad$$                 | altitude                      |
| $$V \quad$$                 | velocity, positive upwards    |
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

#### The setting of the problem

The setting of Inital Triplet：

- Initial velocity $$V_0$$
- Direction/Angle $$\theta_0$$
- Thrust Force $$T$$

It comes to the problems: how to set them to launch the rocket into expected orbit?

We just throw away the problems such as the type of fuel, the complexity of the atomosphere and so on, we just deal with the problem in an ideal situation：

- Thrust $$T$$ can be predicted
- AeroDynamics drag $$D=\frac{1}{2} \rho V^{2} C_{D} A$$ 
- $$T\;\&D\;$$ have the same direction with $$V$$
- Drag Reference Area $$A$$ is a constant



#### The Model Of Launching A Rocket

​	In this Section, we will make a series of progressive models on the trajectory of rocket

{:refdef: style="text-align: center;"}
<img src="/images/2020-07-11-Launching-a-Rocket/rocket-trajectory.png" alt="rocket-trajectory" style="zoom:90%;" />
{:refdef}

​	No matter how many factors we consider, the target is sending the rocket into the orbit, with the velocity not less than the first cosmic velocity /escape velocity

##### First Model $$<T,G,\theta>$$

​	The first model is very simplified, it only consider:

- Thrust $$T$$
- Gravity $$G$$
- $$\theta$$



---
layout: post
title: "Advances-in-Active-Portfolio-Management-Notes"
date: 2023-12-19
image: images/cover/C_Scenery3.jpg   
tags: [Econmics]
toc: false
published: false

---

{: class="table-of-content"}
* TOC
{:toc}



## Details reading notes of "Introduction to the Recap of Active Portfolio Management" of the "Advances in Active Portfolio Management"

On this blogpost, it is the detailed reading notes the "Introduction to the Recap of Active Portfolio Management" section of "Advances in Active Portfolio Management", section-wise.

<!-- ## Introduction to the Recap of Active Portfolio Management -->

On this chapter, the authors include three articles here:

## “Seven Insights into Active Management” by Ronald N. Kahn, which can be summarized as following key insights:

### Framework

  The framework in  context of active protfolio management is based on a decompostion equation: 

  $$
  r_n(t) = \beta_n \cdot r_B(t) + \theta_n(t)
  $$

  - $r_n(t)$: The return of asset $n$ at time $t$.
  - $\beta_n$: The sensitivity of asset $n$'s returns to the returns of the benchmark. This is often referred to as the asset's "beta."
  - $r_B(t)$: The return of the benchmark at time $t$.
  - $\theta_n(t)$: The residual or idiosyncratic(residual) return of asset $n$ at time $t$, which is independent of the benchmark.
  
  This equation suggests that the return of an asset can be viewed as a combination of the return that is due to the market or benchmark (systematic risk) and the return that is unique to the asset itself (idiosyncratic risk).

  The Capital Asset Pricing Model states that the zero expected idiosyncratic return, while the 'smart' participatant in the market can craft the information $g$, yield non-zero expected idiosyncratic return $\alpha$
  
  - $ E\left\{\theta_n\right\}=0 $
  - $ E\left\{\theta_n \mid g\right\} \equiv \alpha_n $

  The authors specialized denote $\alpha$ as the expected idiosyncratic return. To connect the forecast residual return with optimal portfolios, the Markowitz mean-variance optimization reveal that:

  $$
  \begin{aligned}
  \text { Utility } & =\mathbf{h}^T \cdot \boldsymbol{\alpha}-\lambda \mathbf{h}^T \cdot \mathbf{V} \cdot \mathbf{h} \\
  & =\alpha_P-\lambda \omega_P^2
  \end{aligned}
  $$

  Here the $\lambda$ is the risk-aversion parameter, and the $\omega$ measures risk. The optimal portfolio Q can be deduced from the deritvative of the utility, with $\alpha=2 \lambda \mathbf{V} \cdot \mathbf{h}_{\varrho}$. The risk-aversion vary, the optimal protfolio also vary.


- “A Retrospective Look at the Fundamental Law of Active Management” by Richard C. Grinold and Ronald N. Kahn

  <!-- On this chaper, the fundamental law of active nmanagement is discuessed. Given the insight on the information ratios, the high information ratio products should be prefered. The defintion of information ratio $I R$ in the article can be divided into two factors: -->

  The chapter on the fundamental law of active management provides critical insights into how investment decisions can be optimized for better performance. This optimization is quantified using the Information Ratio $I R$, which is a critical metric in portfolio management. The formula provided in the chapter for the Information Ratio is:

  $$
  I R=I C \cdot \sqrt{B R}
  $$

  Here, $I C$ stands for the Information Coefficient, which measures the skill involved in each investment decision. Essentially, it reflects the accuracy of an investment manager's predictions about the performance of an asset. A higher $I C$ suggests a greater level of skill or accuracy in forecasting asset returns.

  Conversely, $B R$ represents the Breadth, referring to the number of independent investment decisions made each year. Breadth can be seen as the scope or range of investment decisions. A larger breadth means more opportunities to utilize skill, potentially leading to more successful returns.
  <!-- understood as the scope or range of investment decisions. A larger breadth indicates more opportunities to utilize skill, essentially providing more shots at achieving returns. -->

  Moving beyond the theoretical framework, the authors pose two practical questions in investment:

  - Investing in Team Skill vs. Expanding Coverage Universe:
    The decision to invest in enhancing the team's skill (improving $I C$) versus expanding the coverage universe (increasing $B R$) hinges on which option more effectively increases the Information Ratio. This decision may vary based on the current level of team skill, the potential for improvement, and the opportunities available in expanding the investment universe.

  - Viability of Basic Tactical Asset Allocation:
    The discussion on the viability of shifting between stocks, bonds, and cash on a quarterly basis relates to the frequency and diversity of investment decisions, thus impacting $B R$. If such tactical decisions are made accurately, they can contribute to a higher $I C$ as well.

  In addition to the traditional fundamental law, the authors introduce an extended version, incorporating the transfer coefficient $T C$, which measures efficiency — the correlation of the actual portfolio held with the optimal paper portfolio, built without constraints and costs.

  $$
  I R=I C \cdot \sqrt{B R} \cdot T C 
  $$

  This extended law has profound implications for participants in the market, regardless of their investment style, must find a balance between skill, breadth, and efficiency. Quantitative strategies often focus on all three aspects. While fundamental strategies aim in particular for high IC — deep understanding of the relatively few stocks they hold — in combination with lower breadth and efficiency. 
  
  The authors illustrate us the dynamics of the market with the tactical asset allocation strategies, once popular in the 1980s and 1990s, have fallen out of favor, largely due to the insights provided by the fundamental law. Their limited breadth means that even with reasonably high levels of skill, these strategies struggle to deliver high active returns per unit of active risk. This

- “Breadth, Skill, and Time” by Richard C. Grinold and Ronald N. Kahn

  Given the four ellements which comprise the fundeamental law of active management:
  - $I R$ stands for the Information Rate
  - $I C$ stands for the Information Coefficient
  - $B R$ represents the Breadth
  - $T C$ represents the Transfer Coefficient
  <!-- 
  The $I R$, which is the central of the investment , can be interpretes as active return and active risk ratio. And the information coefficient and the transfer coefficient, is usually understood as corrleation. The Breath is defined as the number of independent bets per year, which confuses many readers.
  The authors explain the concept of the breath in the article. -->

  The $I R$, central to investment analysis, can be interpreted as the ratio of active return to active risk. Both the Information Coefficient $I C$ and the Transfer Coefficient $T C$ are typically understood as measures of correlation. While Breadth $B R$, often a source of confusion, is defined as the number of independent bets per year. The authors delves into the concept of the breath in the article.


### Insight 1. Active Management Is Worse Than a Zero-Sum Game

Start with Sharpe's arugement:

- The Sum of All Investment Positions Equals the Market:
  Active + Passive = Market
- Index Management Positions Mimic the Market:
  Passive = k * Market
- Active Management Positions Also Make Up the Market
  Active = (1 - k) * Market

Based on these points, Sharpe concludes that:

- Before Fees and Costs: 
  The average performance of all active managers, when considered as a group, will match the market. This is a simple mathematical truth arising from the fact that the market's performance is just the weighted sum of all individual performances, including both active and passive strategies.

- After Fees and Costs: 
  Once you account for the fees and costs associated with active management (which are typically higher than those for passive index funds), the average active manager will underperform the market. This is because they are, on average, only matching the market before fees and costs.

- Index Funds Performance:
  Consequently, index funds, which have lower costs, tend to be above median performers compared to active funds, since they match market performance before fees and outperform the average active manager after fees.

### Insight 2. Information Ratios Determine Added Value

Individual preferences enter into the utilitu only in how individuals trade off residual return against risk. The information ratio is the manager's ratio of residual return to risk:

$$
I R_P=\frac{\alpha_P}{\omega_P}
$$

which can be asserted, that given a participant, more risk, means more residual return. Vice versa, more residual loss came from more risk.




## The Transaction-Based Measure on Economic Machine

Dalio introduces a foundational concept that diverges from classical economics. He depicts the modern credit monetary-based economy through what he terms the "Transaction-Based Approach." Through this lens, we can view the economy as a cumulative result of individual transactions.

At the core of economics lies the transaction, which involves two primary entities:

- a buyer, who brings both money and credit to the market
- a seller, who provides a particular good or financial assets. 

Moving up in complexity, a specific market represents the aggregated transactions related to a particular good. Broadening this idea, our global economy is essentially the sum of all such markets for every conceivable good. 

Apart from categorizing all the markets on their trading qualities, we can also classify the transactions based on the buyers. The buyers can originate from:

- The private sector: Further broken down into households and businesses, both domestic and foreign.
- The government sector: Illustrated by the US, the federal government allocates money to domestic goods and services, while the central bank has the capability to create money and primarily invest in financial assets.

In the transaction-based perspective, assuming a stable supply of the asset quantity, which shows short-term fluctuation, the asset's price is influenced by both the money and credit expended on it. Given our modern credit-based economy, it's easy to adjust the supply of spending in the market.

{:refdef: style="text-align: center;"}
<img src="/images/2023-10-23-How-the-Economics-Machine-Works-Notes/Transaction_based_economics.png" alt="Transport_Map_Visualization" style="zoom:50%;" />
{:refdef}

## Reference

[1] Ray Dalio 2015 [How the Economic Machine Works](https://www.bridgewater.com/research-and-insights/how-the-economic-machine-works) 

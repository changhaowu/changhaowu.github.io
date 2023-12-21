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



## Introduction to My Reading Notes on "Advances in Active Portfolio Management"

This is my latest blog post where I share my journey into the dynamic and often perplexing world of active portfolio management. As a newcomer to this field, I found myself drawn to a comprehensive book on the subject, eager to unravel its complexities. My approach to understanding this intricate topic began with a simple yet effective strategy: navigating through the book's table of contents. This provided me with a clear roadmap, guiding me through the diverse and interconnected topics covered in the book.

The structure of the book is methodically designed, with each section building upon the last, beginning with foundational concepts and progressively delving into more advanced and practical strategies. Let's embark on a brief tour of these sections:

- **Section 1: Recap of Active Portfolio Management**
  This section revisits the key themes of the original book, including a retrospective look at the fundamental law of active management and a deep dive into the concept of breadth.

- **Section 2: Advances in Active Portfolio Management**
  Delving into more sophisticated strategies and approaches in the dynamic world of portfolio management.

  - **Section 2.1: Dynamic Portfolio Management**
    Focusing on adaptive strategies for investment management, this subsection includes articles on implementation efficiency, dynamic portfolio analysis, signal weighting, and both linear and nonlinear trading rules.

  - **Section 2.2: Portfolio Analysis and Attribution**
    Exploring the analytical side, this subsection converts attributes into portfolios for analysis through covariances and correlations. It covers topics like portfolio attribution and portfolio descriptions.

- **Section 3: Applications of Active Portfolio Management**
  Applying the theories to real-world scenarios, covering a wide range of topics including expected return, smart beta, risk, and portfolio construction.

  - **Section 3.1: Expected Return: The Equity Risk Premium and Market Efficiency**
    Discussing the estimation of the equity risk premium, the uses of beta, and the efficiency of benchmark portfolios.

  - **Section 3.2: Expected Return: Smart Beta**
    A look into smart beta/factor investing, including who should use it, an owner's manual, illustrations, and the asset manager's dilemma.

  - **Section 3.3: Risk**
    Examining alternative definitions of risk and their implications in portfolio management.

  - **Section 3.4: Portfolio Construction**
    Articles covering various aspects of portfolio construction, including optimal gearing, the dangers of diversification, asset growth's impact on expected alpha, scenario-based approaches, and myths about fees.

- **Section4: Extras and Conclusion**
  Concluding with additional material, including award-winning presentations, essays on behavioral finance, and career advice for financial engineering students, followed by a retrospective look at quantitative investing over the past 65 years.

Throughout the book, complex concepts such as the equity risk premium, beta and alpha, benchmark portfolios, smart beta strategies, and risk are unpacked, making them accessible for readers who seek to understand the intricacies of active portfolio management:

- **The Equity Risk Premium**: It's the extra profit we expect from stocks compared to those oh-so-safe government bonds.
  
- **Beta and Alpha**: Beta measures a security's or portfolio's volatility compared to the market. Alpha is all about outperformance; it's the score showing how well a portfolio manager has done against a benchmark.
  
- **Benchmark Portfolios**: These are standard portfolios, like the S&P 500, against which other investment performances are measured.
  
- **Smart Beta and Expected Return**: Smart beta is a strategic approach to index construction that aims for higher returns for a given level of risk.
  
- **Risk**: In the investment world, risk is about the uncertainty of returns and the potential for financial loss.

Interestingly, I learned that in financial modelling research, risk and return are often modeled together, by associating expected returns with risk, quantitative models can seek to maximize returns for a given level of risk, or minimize risk for a given level of expected return. This approach allows for a more nuanced understanding and management of a portfolio's risk profile in relation to its return potential.

And when it comes to achieving 'alpha,' or that extra return over the market, there are many paths to explore:

- **Smart Beta as a Source of Alpha**
  By deviating from traditional market-cap indices and focusing on factors like size or value, smart beta strategies try to outperform the market.

- **Other Methods to Generate Alpha**
  - **Stock Selection**: Choosing stocks believed to be undervalued or with high growth potential.
  - **Market Timing**: Attempting to predict market movements to make gains.
  - **Sector Rotation**: Investing in sectors likely to outperform based on current economic trends.
  - **Arbitrage Strategies**: Taking advantage of price differences in various markets.
  - **Alternative Investments**: Diversifying with assets like real estate or commodities.
  - **Derivatives**: Using financial instruments like options and futures for hedging or speculation.
  - **Quantitative Strategies**: Employing mathematical models to identify profitable investment opportunities.


## Introduction to the Recap of Active Portfolio Management

On this chapter, the authors include three articles here:

- “Seven Insights into Active Management” by Ronald N. Kahn, which can be summarized as following key insights:
  - The arithmetic of active management shows that it is worse than a zero-sum game.
  - Information ratios determine value added.
  - Allocate risk budgets in proportion to information ratios.
  - Alphas must control for skill, volatility, and expectations.
  - The fundamental law of active management shows that information ratios depend on skill, breadth, and efficiency.
  - Data mining is easy.
  - Constraints and costs have a surprisingly large impact.

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

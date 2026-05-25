# Fundamental Review of the Trading Book

This project analyzes the evolution of market risk capital requirements in the transition from Basel II/III to the Fundamental Review of the Trading Book (FRTB). It compares:
1. The Internal Models Approach (IMA) under Basel II/III, based on a 10-day, 99% Value-at-Risk (VaR) and a Stressed VaR, scaled by a regulatory multiplier.
2. The Standardized Approach (SA) under FRTB, which applies the Sensitivities-Based Method (SBM) for a more risk-sensitive and transparent framework.

The objective is to evaluate how each method reflects portfolio’s market risk by comparing their capital charges and identifying when their results converge or diverge. 

# Problem setting

We study the market risk capital requirements for a portfolio composed by three Corporate Bonds, an IRS and a Swaption.

Key challenges addressed in this project include:
1. Bootstrap of Discount Factors, Zero Rates and Zeta Spreads curves
2. Portfolio pricing
3. Sensitivity estimation 
4. VaR and sVaR computation 
7. Comparison of IMA with SA

# Methods and Strategies

We compute the VaR and sVaR with several different approaches:
1. Statistical Bootstrap 
2. Gaussian Monte Carlo 
3. Weighted Historical Simulation 
4. Parametric Gaussian 

# Acknowledgements

I thank my colleague Susanna Gao who worked with me on this project.

"""
Mathematical Engineering - Financial Engineering, FY 2024-2025
Risk Management - Final Project: Fundamental Review Trading Book - Group 2A
Var Computation functions: VaR methods, minimize functions
"""

import numpy as np
import math
import pandas as pd

from scipy.stats import norm
from typing import List, Tuple, Callable

def read_historical_data (file_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Read historical data and stressed historical data from the file Excel.

    Parameters:
        file_path (str): Path of the Excel file.

    Returns:
        histo (np.ndarray): Historical data.
        stress_histo (np.ndarray): Stressed data.
    """

    # Read the data
    histo_rates = pd.read_excel(file_path, sheet_name=1, engine='openpyxl').iloc[30:292, 1:11].to_numpy()
    histo_spread = pd.read_excel(file_path, sheet_name=2, engine='openpyxl').iloc[30:292, 1:6].to_numpy()
    stress_rates = pd.read_excel(file_path, sheet_name=1, engine='openpyxl').iloc[1753:2014, 1:11].to_numpy()
    stress_spread = pd.read_excel(file_path, sheet_name=2, engine='openpyxl').iloc[1753:2014, 1:11].to_numpy()

    # Unique matrix of risk factor for both scenarios
    histo = np.hstack((histo_rates, histo_spread))
    stress_histo = np.hstack((stress_rates, stress_spread))

    return histo, stress_histo

def daily_losses (
    s_ir: List[float],
    s_cs: List[float],
    histo: np.ndarray,
    stress_histo: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes daily portfolio losses using IR/CS sensitivities and historical/stress shock scenarios.
    The loss is (-) a linear combination of the risk factors, weighted by the sensitivities.

    Parameters:
        s_ir (List[float]): Interest rate sensitivities vector.
        s_cs (List[float]): Credit spread sensitivities vector.
        histo (np.ndarray): Historical data scenarios.
        stress_histo (np.ndarray): Historical stress period scenarios.

    Returns:
        daily_loss (np.ndarray): Historical daily losses.
        daily_sloss (np.ndarray): Stress period daily losses.
    """

    # Select only the vertex of interest: ([3, 5] for cs, [0.25, 0.5, 1, 2, 3, 5] for ir)
    daily_sloss = - ( np.dot(stress_histo[:, 12:14], s_cs) + np.dot(stress_histo[:, 0:6], s_ir) )
    daily_loss = - ( np.dot(histo[:, 12:14], s_cs) + np.dot(histo[:,0:6], s_ir) )

    return daily_loss, daily_sloss

def bootstrap_VAR (
    alpha: float,
    Delta: int,
    daily_loss: np.ndarray,
    N_sim: int
) -> float:
    """
    Computes the Value-at-Risk using historical bootstrapping.

    Parameters:
        alpha (float): Confidence level.
        Delta (int): Holding period (in days).
        daily_loss (np.ndarray): Array of daily portfolio losses.
        N_sim (int): Number of bootstrap simulations.

    Returns:
        VaR (float): Value-at-Risk at the given confidence level.
    """

    # Number of historical observations
    m = len(daily_loss)

    # Randomly sample Delta days with replacement
    idx = np.random.randint(0, m, size=(Delta, N_sim))

    # Extract and sum losses for each simulation
    boot_sim = daily_loss[idx]
    loss = np.sum(boot_sim, axis=0).flatten()

    # Sort losses in descending order
    sorted_loss = np.sort(loss)[::-1]

    # Compute index for the VaR threshold
    n = len(sorted_loss)
    index = math.floor(n * (1 - alpha))

    # VaR
    VaR = sorted_loss[index]

    return VaR

def gauss_parametric_VAR(
    alpha: float,
    Delta: int,
    daily_loss: np.ndarray
) -> float:
    """
    Computes the Parametric Gaussian Value-at-Risk.

    Parameters:
        alpha (float): Confidence level.
        Delta (int): Holding period (in days).
        daily_loss (np.ndarray): Array of daily portfolio losses.

    Returns:
        VaR (float): Value-at-Risk at the given confidence level.
    """

    # Gaussian parameters of the loss
    mu = np.mean(daily_loss)
    sigma = np.std(daily_loss, ddof=1)

    # Closed parametric formula
    VaR = Delta * mu + sigma * np.sqrt(Delta) * norm.ppf(alpha)

    return VaR

def gauss_MC_VAR(
    alpha: float,
    Delta: int,
    histo: np.ndarray,
    s_ir: List[float],
    s_cs: List[float],
    Nsim: int
) -> float:
    """
    Computes the Gaussian Monte Carlo Value-at-Risk, simulating portfolio losses,
    assuming multivariate normal risk factors.

    Parameters:
        alpha (float): Confidence level.
        Delta (int): Holding period (in days).
        histo (np.ndarray): Historical matrix of risk factor changes.
        s_ir (List[float]): Interest rate sensitivities vector.
        s_cs (List[float]): Credit spread sensitivities vector.
        Nsim (int): Number of Monte Carlo simulations.

    Returns:
        VaR (float): Value-at-Risk at the given confidence level.
    """

    # Mean and Variance with ALL the risk factors
    mu = np.array(np.mean(histo, axis=0)).flatten()
    sigma = np.cov(histo, rowvar=False)

    # Simulate risk factor changes over Delta days (mean and variance scaled by Delta)
    risk_sim = np.random.multivariate_normal(Delta*mu, Delta*sigma, Nsim)

    # Calculate simulated portfolio losses, using only relevant columns of interest rates and spreads
    loss = -(np.dot(risk_sim[:, 0: 6], s_ir)+ np.dot(risk_sim[:, 12: 14], s_cs))

    # Sort losses in descending order
    sorted_loss = np.sort(loss)[::-1]

    # Compute index for the VaR threshold
    n = len(sorted_loss)
    index = math.floor(n * (1 - alpha))

    # VaR
    VaR = sorted_loss[index]

    return VaR

def regulatory_capital_boot(
        alpha: float,
        Delta: int,
        daily_loss: np.ndarray,
        daily_sloss: np.ndarray,
        Nsim: int,
        Nboot: int = 1000,
        CI_level: float = 0.99
) -> Tuple[float, Tuple[float, float]]:
    """
    Computes the value and a confidence interval for regulatory capital 3*(VaR + sVaR),
    using Bootstrap simulation repeated Nboot times.

    Inputs:
        - alpha (float): Confidence level
        - Delta (int): Holding period in days
        - daily_loss (np.ndarray): Array of daily portfolio losses.
        - daily_sloss (np.ndarray): Array of stressed daily portfolio losses.
        - Nsim (int): Number of Bootstrap simulations per run
        - Nboot (int): Number of Bootstrap repetitions, default = 1000
        - CI_level (float): Confidence level for the CI, default = 0.99

    Returns:
        - mean (float): Mean estimated capital
        - CI (Tuple[float, float]): Confidence interval as lower bound & upper bound
    """

    capital_vals = []

    # Every simulation compute the capital
    for _ in range(Nboot):
        VaR_i = bootstrap_VAR(alpha, Delta, daily_loss, Nsim)
        sVaR_i = bootstrap_VAR(alpha, Delta, daily_sloss, Nsim)
        capital_vals.append(3*(VaR_i + sVaR_i))

    # Confidence Interval
    capital_vals = np.array(capital_vals)

    mean = np.mean(capital_vals)
    std = np.std(capital_vals, ddof=1)
    z = norm.ppf((1 + CI_level) / 2)

    CI = (mean - z * std / np.sqrt(Nboot), mean + z * std / np.sqrt(Nboot))

    return mean, CI

def regulatory_capital_MC(
        alpha: float,
        Delta: int,
        histo: np.ndarray,
        stress_histo: np.ndarray,
        s_ir: List[float],
        s_cs: List[float],
        Nsim: int,
        Nboot: int = 1000,
        CI_level: float = 0.99
) -> Tuple[float, Tuple[float, float]]:
    """
    Computes the value and a confidence interval for regulatory capital 3*(VaR + sVaR),
    using Gaussian Monte-Carlo simulation repeated Nboot times.

    Inputs:
        - alpha (float): Confidence level
        - Delta (int): Holding period in days
        - histo_var (np.ndarray): Historical data for VaR estimation
        - histo_svar (np.ndarray): Historical data for stressed VaR estimation
        - s_ir (List[float]): Interest rate sensitivities vector
        - s_cs (List[float]): Credit spread sensitivities vector
        - Nsim (int): Number of MC simulations per run
        - Nboot (int): Number of bootstrap repetitions, default = 1000
        - CI_level (float): Confidence level for the CI, default = 0.99

    Returns:
        - mean(float): Mean estimated capital
        - CI (Tuple[float, float]): Confidence interval as lower bound & upper bound
    """

    capital_vals = []

    # Every simulation compute the capital
    for _ in range(Nboot):
        VaR_i = gauss_MC_VAR(alpha, Delta, histo, s_ir, s_cs, Nsim)
        sVaR_i = gauss_MC_VAR(alpha, Delta, stress_histo, s_ir, s_cs, Nsim)
        capital_vals.append(3*(VaR_i + sVaR_i))

    # Confidence Interval
    capital_vals = np.array(capital_vals)

    mean = np.mean(capital_vals)
    std = np.std(capital_vals, ddof=1)
    z = norm.ppf((1 + CI_level) / 2)

    CI = (mean - z * std / np.sqrt(Nboot), mean + z * std / np.sqrt(Nboot))

    return mean, CI

def weighted_hs_VAR (
        alpha: float,
        Delta: int,
        daily_loss: np.ndarray,
        l: float
) -> float:
    """
    Computes the Weighted Historical Simulation Value-at-Risk. Past scenarios are weighted
    exponentially to give more importance to recent observations.

    Parameters:
        alpha (float): Confidence level.
        Delta (int): Holding period (in days).
        daily_loss (np.ndarray): Array of daily portfolio losses.
        l (float): Exponential decay

    Returns:
        VaR (float): Value-at-Risk at the given confidence level.
    """

    # Compute scale factor for the holding period
    b = np.sqrt(Delta)

    # Adjust losses to reflect the holding period using mean-scaling
    a = np.mean(daily_loss) * (Delta - b)
    loss = a + b * daily_loss

    n = loss.shape[0]

    # Compute normalization constant for exponential weights
    C = (1 - l) / (1 - l**n)

    # Compute exponentially decaying weights
    weights = C * (l ** np.arange(n))

    # Sort losses in descending order and apply the same order to weights
    idx = np.argsort(loss)[::-1]
    sorted_loss = loss[idx]
    sorted_weights = weights[idx]

    # Compute cumulative sum of sorted weights
    cum_weights = np.cumsum(sorted_weights)

    # Find the smallest index where cumulative weight exceeds 1 - alpha
    VaR_idx = np.searchsorted(cum_weights, 1 - alpha, side='left')

    # Return the corresponding loss as the VaR
    VaR = sorted_loss[VaR_idx]

    return VaR

def search_min(
        deltas: np.ndarray,
        alphas: np.ndarray,
        f: Callable[[float, float], float],
        x_min: float = np.inf
) -> Tuple[float, float, float, np.ndarray]:
    """
    Finds the combination (alpha, delta) that minimizes the function f(alpha, delta), i.e.
    find the pair (alpha, delta) that best matches Kold with Ksa.

    Parameters:
        deltas (np.ndarray): Array of delta values.
        alphas (np.ndarray): Array of alpha values.
        f (function): Function to be minimized, f(alpha, delta), i.e. capitals difference.
        x_min (float): Initial reference value, default = inf.

    Returns:
        diff_min (float): Minimum value found.
        alpha_min (float): Alpha corresponding to the minimum.
        delta_min (float): Delta corresponding to the minimum.
        x (np.ndarray): 2D array of f evaluated at each (alpha, delta) combination.
    """

    # Initialize a matrix to store the evaluated values of f over the alpha-delta grid
    x = np.zeros((len(deltas), len(alphas)))

    alpha_min = None
    delta_min = None

    # Loop over each delta
    for i, delta in enumerate(deltas):

        # Loop over each alpha
        for j, alpha in enumerate(alphas):

            # Evaluate the function f at the current (alpha, delta) combination
            val = f(alpha, delta)
            x[i, j] = val

            # Update the minimum if a lower value is found
            if val < x_min:
                x_min = val
                alpha_min = alpha
                delta_min = delta

    # Final minimum value found
    diff_min = x_min

    return diff_min, alpha_min, delta_min, x

def search_min_delta(
        deltas: np.ndarray,
        alphas: np.ndarray,
        f: Callable[[float, float], float]
) -> np.ndarray:
    """
    For each delta, finds the alpha that minimizes the function f(alpha, delta), i.e. for each
    delta finds the alpha that best matches kold with Ksa.

    Parameters:
        deltas (np.ndarray): Array of delta values.
        alphas (np.ndarray): Array of alpha values.
        f (function): Function to be minimized, f(alpha, delta), i.e. capitals difference.

    Returns:
        alpha_star (np.ndarray): Array containing the optimal alpha for each delta.
    """

    alpha_star = np.zeros(len(deltas))

    # Iterate over each delta
    for i, delta in enumerate(deltas):
        current_min = np.inf
        alphamin = None

        # Find the alpha that minimizes f for this delta
        for alpha in alphas:
            x = f(alpha, delta)
            if x < current_min:
                current_min = x
                alphamin = alpha

        # Store the result
        alpha_star[i] = alphamin

    return alpha_star











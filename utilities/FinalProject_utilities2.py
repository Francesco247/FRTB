"""
Mathematical Engineering - Financial Engineering, FY 2024-2025
Risk Management - Final Project: Fundamental Review Trading Book - Group 2A
Standardised Approach functions: sensitivities, capital allocation
"""

import numpy as np
import datetime as dt
from typing import List, Union

from utilities.FinalProject_utilities1 import (
    swap_par_rate,
    portfolio_price_calculator,
    discounts_from_zero_rates
)

def bump (
    rates: np.ndarray,
    vertex: int,
) -> np.ndarray:
    """
    Bumps a single rate/spread by 1 basis point at the specified index (vertex).

    Parameters:
        rates (np.ndarray): Array of Zero-Rates/Zeta-Spreads to be bumped.
        vertex (int): Index of the curve to bump.

    Returns:
        new_rates (np.ndarray): New array with the specified rate/spread bumped by 1 bp.
    """

    bps = 1e-4
    new_rates = rates.copy() # Create a copy and not modify in the main!!
    new_rates[vertex] += bps

    return new_rates

def sensitivities_ir (
    notionals: List[float],
    maturities: List[int],
    coupons: List[float],
    start_date: Union[dt.date, dt.datetime],
    fwd_start_date: Union[dt.date, dt.datetime],
    ZS: np.ndarray,
    dates: List[Union[dt.date, dt.datetime]],
    sigma: float,
    K: float,
    zero_rates: np.ndarray,
    vertex: List[int],
    p1: float
) -> List[float]:
    """
    Computes the interest rate sensitivities of the portfolio via bump-and-reprice. The discounts are re-computed, as well
    as the forward swap rate (depends on the discounts).

    Parameters:
        notionals (List[float]): Notional amounts of contracts.
        maturities (List[int]): Maturities of the contracts.
        coupons (List[float]): Coupon rates of the contracts.
        start_date (Union[dt.date, dt.datetime]): Valuation date.
        fwd_start_date (Union[dt.date, dt.datetime]): Forward swap start date.
        ZS (np.ndarray): Zeta-Spreads for bond pricing.
        dates (List[Union[dt.date, dt.datetime]]): Dates from the bootstrap.
        sigma (float): Implied black volatility of the swaption.
        K (float): Strike rate of the swaption.
        zero_rates (np.ndarray): Zero-Rates curve.
        vertex (List[int]): Indices of the zero curve to bump.
        p1 (float): Baseline portfolio value.

    Returns:
        s_ir (List[float]): List of sensitivities DV01z.
    """

    s_ir = []

    for j in range (len(vertex)):

        # Extract the index
        idx = vertex[j]

        # Bump the vertex
        new_zerorates = bump(zero_rates, idx)
        new_discounts = discounts_from_zero_rates(dates, new_zerorates) # Discounts change
        new_SIR = swap_par_rate(dates, new_discounts, 3, maturities[4], fwd_start_date) # Also the SIR changes

        # New portfolio price with update data
        p2 = portfolio_price_calculator(notionals, maturities, coupons, start_date, fwd_start_date, ZS, dates, new_discounts, K, sigma, new_SIR)

        # DV01z
        s_ir.append(p2-p1)

    return s_ir

def sensitivities_cs (
    notionals: List[float],
    maturities: List[int],
    coupons: List[float],
    start_date: Union[dt.date, dt.datetime],
    fwd_start_date: Union[dt.date, dt.datetime],
    ZS: np.ndarray,
    dates: List[Union[dt.date, dt.datetime]],
    sigma: float,
    K: float,
    SIR: float,
    discounts: np.ndarray,
    vertex: List[int],
    p1: float
) -> List[float]:
    """
    Computes the credit spread sensitivities of the portfolio bump-and-reprice. In this case only the Zeta-Spreads curve is recomputed.

    Parameters:
        notionals (List[float]): Notional amounts of the contracts.
        maturities (List[int]): Maturities of the contracts.
        coupons (List[float]): Coupon rates of the contracts.
        start_date (Union[dt.date, dt.datetime]): Valuation date.
        fwd_start_date (Union[dt.date, dt.datetime]): Forward swap start date.
        ZS (np.ndarray): Zeta-Spreads for bond pricing.
        dates (List[Union[dt.date, dt.datetime]]): Dates from the bootstrap.
        sigma (float): Implied black volatility of the swaption.
        K (float): Strike rate of the swaption.
        SIR (float): Forward swap rate.
        discounts (np.ndarray): Discount factors from the bootstrap.
        vertex (List[int]): Indices of the ZS curve to bump.
        p1 (float): Baseline portfolio value.

    Returns:
        s_cs (List[float]): List of sensitivities CS01.
    """

    s_cs = []

    for j in range (len(vertex)):

        # Extract the index
        idx = vertex[j]

        # Bump the vertex
        new_ZS = bump(ZS, idx) # Zeta-Spreads change

        # New portfolio price with update data
        p2 = portfolio_price_calculator(notionals, maturities, coupons, start_date, fwd_start_date, new_ZS, dates, discounts, K, sigma, SIR)

        # CS01
        s_cs.append(p2 - p1)

    return s_cs

def capital (
    weights: List[float],
    rho: np.ndarray,
    s_k: List[float]
) -> float:
    """
    Computes the capital requirement using the sensitivities and the correlation matrix.

    Parameters:
        weights (List[float]): Risk weights for each risk factor.
        rho (np.ndarray): Correlation matrix between risk factors.
        s_k (List[float]): Sensitivities to each risk factor.

    Returns:
        K_ir (float): Capital requirement.
    """

    # Standardised approach delta formula
    ws_k = np.array(weights) * np.array(s_k)
    K_ir = np.sqrt(max(0, ws_k @ rho @ ws_k.T))

    return K_ir



















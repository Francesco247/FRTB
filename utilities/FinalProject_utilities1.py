"""
Mathematical Engineering - Financial Engineering, FY 2024-2025
Risk Management - Final Project: Fundamental Review Trading Book - Group 2A
Basic functions: handle market data, price portfolio instruments
"""

import numpy as np
import pandas as pd
import datetime as dt
import calendar
import holidays

from datetime import timedelta, datetime
from scipy.optimize import root_scalar
from typing import List, Union, Dict
from enum import Enum
from scipy.stats import norm

class DayCount(Enum):
    """
    Types of conventions.
    """

    EU_30_360 = 2  # EU 30/360
    ACT_360 = 0    # ACT/360
    ACT_365 = 1    # ACT/365

class SwapType(Enum):
    """
    Types of swaptions.
    """

    RECEIVER = "receiver"
    PAYER = "payer"

def adjust_to_busday(dates: List[dt.date]) -> List[dt.date]:
    """
    Adjust a list of dates to the next business day if they fall on a weekend or holiday.

    Parameters:
        dates (list[dt.date]): List of dates to be adjusted.

    Returns:
        list[dt.date]: Adjusted business dates.
    """

    # Collect all unique years from the input dates to define the holiday calendar
    dates = pd.to_datetime(dates)
    years = dates.year.unique()

    # Initialize the US holiday calendar for the relevant years
    hol = holidays.US(years=years)

    # Manually add additional holidays
    hol[datetime(2023, 4, 7)] = "Good Friday"

    def next_bd(d):
        while d.weekday() >= 5 or d in hol:
            # If it's Saturday (5), Sunday (6), or a holiday, move to the next day
            d += timedelta(days=1)
        return d.date()

    # For each date, return the date itself if it's a business day,
    # otherwise move it to the next business day
    return [d.date() if d.weekday() < 5 and d not in hol else next_bd(d) for d in dates]

def business_date_offset(
    base_date: Union[dt.date, pd.Timestamp],
    year_offset: int = 0,
    month_offset: int = 0,
    day_offset: int = 0,
) -> Union[dt.date, pd.Timestamp]:
    """
    Return the closest following business date to a reference date after applying the specified offset.

    Parameters:
        base_date (Union[dt.date, pd.Timestamp]): Reference date.
        year_offset (int): Number of years to add.
        month_offset (int): Number of months to add.
        day_offset (int): Number of days to add.

    Returns:
        Union[dt.date, pd.Timestamp]: Closest following business date to ref_date once the specified
            offset is applied.
    """

    # Adjust the year and month
    total_months = base_date.month + month_offset - 1
    year, month = divmod(total_months, 12)
    year += base_date.year + year_offset
    month += 1

    # Adjust the day and handle invalid days
    day = base_date.day
    try:
        adjusted_date = base_date.replace(
            year=year, month=month, day=day
        ) + dt.timedelta(days=day_offset)
    except ValueError:
        # Set to the last valid day of the adjusted month
        last_day_of_month = calendar.monthrange(year, month)[1]
        adjusted_date = base_date.replace(
            year=year, month=month, day=last_day_of_month
        ) + dt.timedelta(days=day_offset)

    # Adjust to the closest business day
    if adjusted_date.weekday() == 5:  # Saturday
        adjusted_date += dt.timedelta(days=2)
    elif adjusted_date.weekday() == 6:  # Sunday
        adjusted_date += dt.timedelta(days=1)

    return adjusted_date

def discounts_from_zero_rates (
        dates: List[Union[dt.date, pd.Timestamp]],
        zero_rates: np.ndarray,
) -> np.ndarray:
    """
    Return the discount factors from a set of Zero-Rates and their corresponding dates.

    Parameters:
        dates (List[Union[dt.date, pd.Timestamp]]): Set of dates corresponding to the Zero-Rates.
        zero_rates (np.ndarray): Set of Zero-Rates.

    Returns:
        discounts (np.ndarray): Set of discounts.
    """

    yf = [yearfrac(dates[0], dates[i], DayCount.ACT_365.value) for i in range(len(zero_rates))]
    discounts = np.exp(-(zero_rates * yf))

    return discounts

def date_series(
    t0: Union[dt.date, pd.Timestamp],
    t1: Union[dt.date, pd.Timestamp],
    freq: int
) -> Union[List[dt.date], List[pd.Timestamp]]:
    """
    Return a list of dates from t0 to t1 inclusive with frequency freq, where freq is specified as
    the number of dates per year.

    Parameters:
        t0 (Union[dt.date, pd.Timestamp]): Initial date.
        t1 (Union[dt.date, pd.Timestamp]): Final date.
        freq (int): Number of dates per year.

    Returns:
        dates (Union[List[dt.date], List[pd.Timestamp]]): Set of required dates.
    """

    dates = [t0]
    while dates[-1] < t1:
        dates.append(business_date_offset(t0, month_offset=len(dates) * 12 // freq))
    if dates[-1] > t1:
        dates.pop()
    if dates[-1] != t1:
        dates.append(t1)

    return dates

def extract_data(file_path: str) -> Dict[str, np.ndarray]:
    """
    Reads the rates and the Yield To Maturities from the Excel file. Saves the data into a dictionary.

    Parameters:
        file_path (str): Path to the Excel file.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing:
            - 'depos_rates': Deposit rates as a NumPy array.
            - 'swaps_rates': Swap rates as a NumPy array.
            - 'ytm': Yield To Maturities as a NumPy array.
    """

    df_cleaned = pd.read_excel(file_path, sheet_name="Market Data")

    # Extraction of depos rates
    depos_rates = df_cleaned.iloc[15:17, 1:3].values
    depos_rates = np.array(depos_rates, dtype=np.float64) # Conversion to float

    # Extraction of swaps rates
    swaps_rates = df_cleaned.iloc[1:11, 1:3].values
    swaps_rates = np.array(swaps_rates, dtype=np.float64) # Conversion to float

    # Extraction of Yield To Maturities
    ytm = df_cleaned.iloc[3:6,5].values
    ytm = np.array(ytm, dtype=np.float64) # Conversion to float

    return {
        "depos_rates": depos_rates,
        "swaps_rates": swaps_rates,
        "ytm" : ytm
    }

def yearfrac(
        t1: dt.datetime,
        t2: dt.datetime,
        x: int
) -> float:
    """
    Compute the year fraction between two dates using the ACT/x convention.

    Parameters:
        t1 (dt.datetime): First date.
        t2 (dt.datetime): Second date.
        x (int): Number of days in a year.

    Returns:
        float: Year fraction between the two dates.
    """

    if x == 0:
        return ((t2-t1).days/360)
    elif x == 1:
        return ((t2-t1).days/365)
    elif x == 2:
        return ((t2.day-t1.day)+30*(t2.month-t1.month)+360*(t2.year-t1.year))/360

def bootstrap(
        datesSet: list[Union[dt.date, dt.datetime]],
        ratesSet: dict[str, np.ndarray]
) -> tuple[list[Union[dt.date, dt.datetime]], np.ndarray]:
    """
    This function computes the discount factors using the bootstrap method. The process involves extracting market rates
    for deposits and swaps, then using them to calculate discount factors iteratively.

    Parameters:
        datesSet (list[Union[dt.date, dt.datetime]]): List of dates corresponding to the discount factors to be calculated.
        ratesSet (dict[str, np.ndarray]): List of rates extracted from the Excel file.

    Returns:
        dates (list[Union[dt.date, dt.datetime]]): List of dates corresponding to the calculated discount factors.
        discounts (np.ndarray): List of discount factors computed using bootstrap.
    """

    n_total = 13   # Total number of dates in the curve (from today to ten years, with 3 months and 6 months, so 13 elements)
    dates = datesSet  # Initialize dates list
    discounts = np.zeros(n_total)  # Initialize Discounts array
    discounts[0] = 1  # Discount equal to 1 for settlement date

    # Compute mid-rates for depos and swaps by averaging bid and ask rates
    rate_mid_depos = ratesSet['depos'].mean(axis=1)
    rate_mid_swaps = ratesSet['swaps'].mean(axis=1)

    # Compute Discount Factors for depos
    for i in range(1, 3):
     discounts[i] = 1 / (1 + (yearfrac(dates[0], dates[i], DayCount.ACT_360.value) * rate_mid_depos[i - 1]))

    # Initial Discount Factor using 1Y swap
    discounts[3] = 1 / (1 + yearfrac(dates[0], dates[3], DayCount.EU_30_360.value) * rate_mid_swaps[0])

    yf = np.zeros(len(rate_mid_swaps))

    # Year fractions for subsequent years
    yf[0] = yearfrac(dates[0], dates[3], DayCount.EU_30_360.value)
    yf[1:] = [yearfrac(dates[i - 1], dates[i], DayCount.EU_30_360.value) for i in range(4, len(dates))]

    # Recursive Discounts calculation
    for i in range(9):
        bpv = np.dot(yf[:i + 1], discounts[3:4 + i])
        discounts[4 + i] = (1 - rate_mid_swaps[i + 1] * bpv) / (1 + yf[i + 1] * rate_mid_swaps[i + 1])

    return dates, discounts

def zero_rates(
        dates: list[Union[dt.date, dt.datetime]],
        discounts: np.ndarray
) -> np.ndarray:
    """
    Computes the Zero-Rates from the discount factors.

    Parameters:
        dates (list[Union[dt.date, dt.datetime]]): List of dates corresponding to the discount factors.
        discounts (np.ndarray): List of discount factors.

    Returns:
        ZR (np.ndarray): Zero-Rates curve.
    """

    # Compute year fractions between the starting date and each other date
    delta = np.array([yearfrac(dates[0], date, 1) for date in dates[1:len(discounts)]])

    # Compute the zero-rates
    ZR = (-np.log(discounts[1:]) / delta)

    # Zero-rate today is 0
    ZR = np.insert(ZR, 0, 0)

    return ZR

def bond_price_ytm(
        start_date: Union[dt.date, dt.datetime],
        expiry: int,
        coupon: float,
        ytm: float,
        notional: float
) -> float:
    """
    Computes the market bond price, using the YTM, for bonds with annual payments.

    Parameters:
        start_date (Union[dt.date, dt.datetime]): Valuation date.
        expiry (int): Tenor of the bond in years.
        coupon (float): Coupon rate.
        ytm (float): YTM of the bond.
        notional (float): Face value of the bond.

    Returns:
        price (float): Price of the bond.
    """

    # Payment dates (annuals) with business day adjustment
    payment_dates = [business_date_offset(start_date, i) for i in range(1, expiry + 1)]
    payment_dates = [start_date] + payment_dates
    payment_dates = adjust_to_busday(payment_dates)

    # Year fractions
    yf_1 = np.array([yearfrac(payment_dates[i], payment_dates[i + 1], DayCount.EU_30_360.value) for i in range(len(payment_dates) - 1)]) # For coupons
    yf_2 = np.array([yearfrac(start_date, payment_dates[i + 1], DayCount.ACT_365.value) for i in range(len(payment_dates) - 1)]) # For discounts

    # Continuous compounding discount factors
    discounts = np.exp(-ytm * yf_2)

    # Price
    price = notional * (coupon * np.sum(discounts * yf_1) + discounts[-1])

    return price

def bond_price(
        start_date: Union[dt.date, dt.datetime],
        expiry: int,
        coupon: float,
        ZS: float,
        dates: List[Union[dt.date, dt.datetime]],
        discounts: np.ndarray,
        notional: float
) -> float:
    """
    Computes the Zeta-Spread bond price (equal to the market price), for bonds with annual payments.

    Parameters:
        start_date ([Union[dt.date, dt.datetime]): Valuation date.
        expiry (int): Tenor of the bond in years.
        coupon (float): Coupon rate.
        ZS (float): Zeta-Spread of the bond.
        dates (list[Union[dt.date, dt.datetime]]): List of dates from the bootstrap.
        discounts (np.ndarray): Discount factors from the bootstrap.
        notional (float): Face value of the bond.

    Returns:
        price (float): Price of the bond.
    """

    # Payment dates (annuals) with business day adjustment
    payment_dates = [business_date_offset(start_date, i) for i in range(1, expiry + 1)]
    payment_dates = [start_date] + payment_dates
    payment_dates = adjust_to_busday(payment_dates)

    # Year fractions
    yf_1 = [yearfrac(payment_dates[i], payment_dates[i + 1], DayCount.EU_30_360.value) for i in range(len(payment_dates) - 1)] # For coupons
    yf_2 = [yearfrac(start_date, payment_dates[i + 1], DayCount.ACT_365.value) for i in range(len(payment_dates) - 1)] # For discounts

    # Get discounts on the payment dates (could skip this computation and directly use discounts from the bootstrap,
    # but to write a more general and re-usable function add this calculation).
    B = new_discounts(dates, discounts, payment_dates[1:])

    # Discounts with Zeta-Spreads
    B_hat = B * np.exp([-ZS*y for y in yf_2])

    # Price
    price = notional * (coupon * np.sum(B_hat * yf_1) + B_hat[-1])

    return price

def zeta_spread (
    start_date: Union[dt.date, dt.datetime],
    expiries: List[int],
    coupons: List[float],
    notionals: List[float],
    ytm: List[float],
    dates: List[Union[dt.date, dt.datetime]],
    discounts: np.ndarray
) -> np.ndarray:
    """
    Calculate Zeta-Spreads from a list of bonds. At each step, computes the ZS as the value which make the bond price
    computed by zero-rates equal to the market price computed by yield to maturity.

    Parameters:
        start_date (Union[dt.date, dt.datetime]): Valuation date.
        expiries (List[int]): Tenors of bonds in years.
        coupons (List[float]): List of coupon rates.
        notionals (List[float]): List of face values.
        ytm (List[float]): List of Yield To Maturities.
        dates (List[Union[dt.date, dt.datetime]]): Dates from the bootstrap.
        discounts (np.ndarray): Discount factors from the bootstrap.

    Returns:
        zs (np.ndarray): Zeta-Spreads values.
    """

    zs = np.zeros(len(expiries))

    for j in range(len(expiries)):

        # Market price with ytm
        p_market = bond_price_ytm(start_date, expiries[j], coupons[j], ytm[j], notionals[j])

        # Function handle: bond price as function of the Zeta-Spread
        def f(zs_):
            return bond_price(start_date, expiries[j], coupons[j], zs_, dates, discounts, notionals[j]) - p_market

        # Find the root using "root_scalar"
        result = root_scalar(f, bracket=[-0.05, 0.10], method='brentq')

        # Save the result
        if result.converged:
            zs[j] = result.root
        else:
            zs[j] = np.nan

    return zs

def new_discounts(
    dates: List[Union[dt.date, dt.datetime]],
    discounts: np.ndarray,
    new_dates: List[Union[dt.date, dt.datetime]]
) -> np.ndarray:
    """
    Interpolate discount factors on the Zero-Rates curve for a new set of dates.

    Parameters:
        dates (List[Union[dt.date, dt.datetime]): Original discounts curve dates.
        discounts (np.ndarray): Discount factors corresponding to the dates.
        new_dates (List[Union[dt.date, dt.datetime]): New dates for which to compute interpolated discounts.

    Returns:
        new_disc (np.ndarray): Interpolated discount factors corresponding to the new dates.
    """

    # Get the Zero-Rates curve
    zRates = zero_rates(dates, discounts)

    # Compute the yearfracs between today and each other date in "new_dates"
    yf = [yearfrac(dates[0], d, DayCount.ACT_365.value) for d in new_dates]

    # Convert dates to ordinal format for interpolation
    dates = [d.toordinal() for d in dates]
    new_date = [d.toordinal() for d in new_dates]

    # Linear interpolation of Zero-Rates at new dates
    new_zrates = np.interp(new_date, dates, zRates)

    # Compute new discount factors
    new_disc = np.exp(-(yf * new_zrates))

    return new_disc

def swap_par_rate(
    dates: List[Union[dt.date, dt.datetime]],
    discounts: np.ndarray,
    expiry: int,
    maturity: int,
    fwd_start_date: Union[dt.date, dt.datetime],
) -> float:
    """
    Return the forward swap rate.

    Parameters:
        dates (List[Union[dt.date, dt.datetime]]): Dates from the bootstrap.
        discounts (np.ndarray): Discount from the bootstrap.
        expiry (int): Forward start in years.
        maturity (int): Forward end in years.
        fwd_start_date (Union[dt.date, dt.datetime]): Forward start date of the swap.

    Returns:
        float: Rate of the swap.
    """

    # Fixed leg payment dates: count from the forward start date
    fixed_leg_schedule = [business_date_offset(fwd_start_date, i) for i in range(1, maturity-expiry+1)]
    fixed_leg_schedule = [fwd_start_date] + fixed_leg_schedule
    fixed_leg_schedule = adjust_to_busday(fixed_leg_schedule)

    # BPV
    df = new_discounts(dates, discounts, fixed_leg_schedule)
    yf = [yearfrac(fixed_leg_schedule[j], fixed_leg_schedule[j+1], DayCount.EU_30_360.value) for j in range(len(fixed_leg_schedule)-1)]
    bpv = np.dot(yf, df[1:])

    return (df[0] - df[-1]) / bpv

def swap_mtm(
        swap_rate: float,
        start_date: Union[dt.date, dt.datetime],
        dates: List[Union[dt.date, dt.datetime]],
        discount_factors: np.ndarray,
        expiry: int,
        notional: float,
        swap_type: SwapType = SwapType.PAYER,
) -> float:
    """
    Return the swap mark-to-market.

    Parameters:
        swap_rate (float): Fixed rate of the swap.
        start_date (Union[dt.date, dt.datetime]): Valuation date.
        dates (List[Union[dt.date, dt.datetime]]): Dates from the bootstrap.
        discount_factors (np.ndarray): Discount factors from the bootstrap.
        expiry (int): Swap maturity in years.
        notional (float): Notional amount of the swap.
        swap_type (SwapType): Swap direction (PAYER or RECEIVER), default is PAYER.

    Returns:
        float: Mark-to-market value of the swap.
    """

    # Fixed leg payment dates: count from the forward start date
    fixed_leg_schedule = [business_date_offset(start_date, i) for i in range(1, expiry + 1)]
    fixed_leg_schedule = [start_date] + fixed_leg_schedule
    fixed_leg_schedule = adjust_to_busday(fixed_leg_schedule)

    # Fixed yearfracs and discount factors
    fixed_yf = [yearfrac(fixed_leg_schedule[i], fixed_leg_schedule[i + 1], DayCount.EU_30_360.value) for i in range(len(fixed_leg_schedule) - 1)]
    fixed_df = new_discounts(dates, discount_factors, fixed_leg_schedule[1:])

    # Fixed and floating leg NPVS
    bpv = np.dot(fixed_yf, fixed_df)

    NPV_fixed = swap_rate*bpv
    NPV_floating = 1 - fixed_df[-1]

    if swap_type == SwapType.RECEIVER:
        multiplier = -1
    elif swap_type == SwapType.PAYER:
        multiplier = 1
    else:
        raise ValueError("Unknown swap type.")

    return notional * multiplier * (NPV_floating - NPV_fixed)

def swaption_price_calculator(
    S0: float,
    strike: float,
    start_date: Union[dt.date, pd.Timestamp],
    fwd_start_date: Union[dt.date, pd.Timestamp],
    expiry: int,
    maturity: int,
    sigma_black: float,
    dates: List[Union[dt.date, pd.Timestamp]],
    discount_factors: np.ndarray,
    notional: float,
    swaption_type: SwapType = SwapType.RECEIVER,
) -> float:
    """
    Return the swaption price.

    Parameters:
        S0 (float): Forward swap rate.
        strike (float): Swaption strike rate.
        start_date (Union[dt.date, pd.Timestamp]): Valuation date.
        fwd_start_date (Union[dt.date, pd.Timestamp]): Forward swap start date.
        expiry (int): Forward start of the swap in years.
        maturity (int): Forward end of the swap in years.
        sigma_black (float): Implied Black volatility.
        dates (List[Union[dt.date, pd.Timestamp]]): Dates from the bootstrap.
        discount_factors (np.ndarray): Discount from the bootstrap.
        notional (float): Notional amount of the swaption.
        swaption_type (SwapType): (PAYER or RECEIVER), default is PAYER.

    Returns:
        float: Swaption price.
    """

    # Dates (fixed leg): count from the forward start date
    fixed_leg_schedule = [business_date_offset(fwd_start_date, i) for i in range(1, maturity - expiry + 1)]
    fixed_leg_schedule = [fwd_start_date] + fixed_leg_schedule
    fixed_leg_schedule = adjust_to_busday(fixed_leg_schedule)

    # BPV
    yf = [yearfrac(fixed_leg_schedule[i], fixed_leg_schedule[i + 1], DayCount.EU_30_360.value) for i in range(len(fixed_leg_schedule) - 1)]
    df = new_discounts(dates, discount_factors, fixed_leg_schedule[1:])
    bpv = np.dot(yf, df)

    # d1 & d2 terms
    DeltaT = yearfrac(start_date, fixed_leg_schedule[0], DayCount.ACT_365.value) ** (1/2)
    d1 = 1/(sigma_black * DeltaT)*np.log(S0/strike)+1/2*sigma_black*DeltaT
    d2 = 1/(sigma_black * DeltaT)*np.log(S0/strike)-1/2*sigma_black*DeltaT

    # Black Formula for both type
    if swaption_type == SwapType.PAYER:
        price = notional * bpv * (S0 * norm.cdf(d1)-strike * norm.cdf(d2))
    elif swaption_type == SwapType.RECEIVER:
        price = notional * bpv * (strike * norm.cdf(-d2)-S0 * norm.cdf(-d1))

    return price

def portfolio_price_calculator(
        notionals: List[float],
        maturities: List[int],
        coupons: List[float],
        start_date: Union[dt.date, pd.Timestamp],
        fwd_start_date: Union[dt.date, pd.Timestamp],
        ZS: List[float],
        dates: List[Union[dt.date, pd.Timestamp]],
        discounts: np.ndarray,
        K: float,
        sigma: float,
        SIR: float,
) -> float:
    """
    Computes the total price of a portfolio consisting of:
        - 3 bonds with annual fixed coupons
        - 1 IRS (payer)
        - 1 swaption (payer)

    Parameters:
        notionals (List[float]): Notional of the contracts.
        maturities (List[int]): Maturities of the contracts.
        coupons (List[float]): Coupons of the contracts.
        start_date (Union[dt.date, pd.Timestamp]): Valuation date.
        fwd_start_date (Union[dt.date, pd.Timestamp]): Forward swap start date.
        ZS (List[float]): Zeta-Spreads of the bonds.
        dates (List[Union[dt.date, pd.Timestamp]]): Dates from the bootstrap.
        discounts (np.ndarray): Discount from the bootstrap.
        K (float): Strike of the swaption.
        sigma (float): Implied black volatility of the swaption.
        SIR (float): Forward swap rate.

    Returns:
        price_portfolio (float): Total portfolio value.
    """

    bond2y = bond_price(start_date, maturities[0], coupons[0], ZS[0], dates, discounts, notionals[0])
    #print(bond2y)
    bond3y = bond_price(start_date, maturities[1], coupons[1], ZS[1], dates, discounts, notionals[1])
    #print(bond3y)
    bond5y = bond_price(start_date, maturities[2], coupons[2], ZS[2], dates, discounts, notionals[2])
    #print(bond5y)
    irs = swap_mtm(coupons[3], start_date, dates, discounts, maturities[3], notionals[3], SwapType.PAYER)
    #print(irs)
    swaption = swaption_price_calculator(SIR, K, start_date, fwd_start_date, 3, maturities[4], sigma, dates, discounts, notionals[4], SwapType.PAYER)
    #print(swaption)

    # Total portfolio price
    price_portfolio = float(bond2y+bond3y+bond5y+irs+swaption)

    return price_portfolio

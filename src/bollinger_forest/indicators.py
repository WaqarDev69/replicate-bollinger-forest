"""
Technical Indicators and Performance Metrics Module.

This module contains functions to calculate various technical indicators used in
quantitative trading strategies, such as Weighted Moving Averages (WMA),
Bollinger Bands, and Average True Range (ATR). It also includes functions
to evaluate portfolio performance metrics like Max Drawdown and Sharpe Ratio.
"""

import pandas as pd
import numpy as np


def calculate_wma_3(series: pd.Series) -> pd.Series:
    """
    Calculates the 3-day Weighted Moving Average (WMA).

    The calculation follows the specific formula described in the research paper:
    WMA(3)_t = (3 * Close_t + 2 * Close_{t-1} + 1 * Close_{t-2}) / 6

    Args:
        series: A pandas Series representing the price time series (typically Close prices).

    Returns:
        A pandas Series containing the calculated 3-day WMA values.
    """
    weights: np.ndarray = np.array([1, 2, 3])
    sum_weights: float = np.sum(weights)
    return series.rolling(window=3).apply(lambda x: np.sum(weights * x) / sum_weights, raw=True)


def calculate_bollinger_bands(df: pd.DataFrame, d: int = 20, k: float = 3.0) -> pd.DataFrame:
    """
    Calculates Bollinger Bands and adds them to the DataFrame.

    This function computes the Moving Average (MA), Standard Deviation (STD),
    Upper Track, and Lower Track based on the specified window size and
    standard deviation multiplier.

    Args:
        df: A pandas DataFrame containing a 'Close' column.
        d: The window size for the moving average (default is 20).
        k: The number of standard deviations for the upper and lower bands (default is 3.0).

    Returns:
        The input pandas DataFrame with added columns: 'MA', 'STD', 'Upper_Track', and 'Lower_Track'.
    """
    close: pd.Series = df['Close']
    df['MA'] = close.rolling(window=d).mean()
    df['STD'] = close.rolling(window=d).std()
    df['Upper_Track'] = df['MA'] + (k * df['STD'])
    df['Lower_Track'] = df['MA'] - (k * df['STD'])
    return df


def calculate_atr(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """
    Calculates the Average True Range (ATR).

    ATR is a measure of volatility. It considers the maximum of:
    1. High - Low
    2. |High - Previous Close|
    3. |Low - Previous Close|

    Args:
        df: A pandas DataFrame containing 'High', 'Low', and 'Close' columns.
        n: The rolling window size for the average (default is 20).

    Returns:
        A pandas Series containing the calculated ATR values.
    """
    high: pd.Series = df['High']
    low: pd.Series = df['Low']
    close: pd.Series = df['Close']

    high_low: pd.Series = high - low
    high_close: pd.Series = np.abs(high - close.shift())
    low_close: pd.Series = np.abs(low - close.shift())

    ranges: pd.DataFrame = pd.concat([high_low, high_close, low_close], axis=1)
    true_range: pd.Series = ranges.max(axis=1)

    return true_range.rolling(window=n).mean()


def calculate_max_drawdown(portfolio_values: pd.Series) -> float:
    """
    Calculates the Maximum Drawdown percentage of a portfolio.

    Maximum Drawdown is the maximum observed loss from a peak to a trough
    of a portfolio, before a new peak is attained.

    Args:
        portfolio_values: A pandas Series representing the portfolio value over time.

    Returns:
        The maximum drawdown as a percentage (e.g., -25.5 for a 25.5% loss).
    """
    rolling_max: pd.Series = portfolio_values.cummax()
    drawdown: pd.Series = portfolio_values / rolling_max - 1.0
    return drawdown.min() * 100.0


def calculate_sharpe_ratio(portfolio_values: pd.Series) -> float:
    """
    Calculates the Annualized Sharpe Ratio.

    The Sharpe Ratio measures the performance of an investment compared to a
    risk-free asset, after adjusting for its risk. This implementation assumes
    a risk-free rate of 0.0 and annualizes the daily returns using a factor of 252.

    Args:
        portfolio_values: A pandas Series representing the portfolio value over time.

    Returns:
        The annualized Sharpe Ratio as a float. Returns 0.0 if the standard deviation of returns is 0.
    """
    daily_returns: pd.Series = portfolio_values.pct_change().dropna()
    if daily_returns.std() == 0:
        return 0.0

    return daily_returns.mean() / daily_returns.std() * np.sqrt(252)

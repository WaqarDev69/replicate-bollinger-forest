"""
Classical Bollinger Band Strategy Module.

This module implements the traditional Bollinger Band trading strategy as described
in Algorithm 1 of the research paper. It serves as a baseline for comparison
against the Enhanced strategy.
"""

from typing import List
import pandas as pd
from ..indicators import calculate_bollinger_bands


class ClassicalBollingerStrategy:
    """
    Implements the Traditional Bollinger Band Strategy (Algorithm 1).

    The strategy follows a simple mean-reversion logic:
    1. Buy (Long) when the closing price falls below the Lower Track.
    2. Sell (Close) when the closing price rises above the Upper Track.

    Attributes:
        initial_capital: The starting cash amount for the simulation.
        d: The window size for the moving average (period).
        k: The number of standard deviations for the bands.
    """

    def __init__(self, initial_capital: float = 100000.0, d: int = 20, k: float = 3.0) -> None:
        """
        Initializes the strategy with capital and indicator parameters.

        Args:
            initial_capital: The starting capital in base currency (default: 100,000).
            d: The moving average window size (default: 20).
            k: The standard deviation multiplier (default: 3.0).
        """
        self.initial_capital: float = initial_capital
        self.d: int = d
        self.k: float = k

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the trading simulation on the provided historical data.

        This method calculates the Bollinger Bands, iterates through the dataframe
        to simulate trading decisions based on Algorithm 1, and tracks the
        portfolio value over time.

        Args:
            df: A pandas DataFrame containing historical stock data with a 'Close' column.

        Returns:
            The input DataFrame with an added 'Portfolio_Value' column representing
            the equity curve of the strategy.
        """
        df = df.copy()
        df = calculate_bollinger_bands(df, d=self.d, k=self.k)

        cash: float = self.initial_capital
        position: int = 0
        shares: float = 0.0
        portfolio_values: List[float] = []

        for i in range(len(df)):
            row = df.iloc[i]
            close: float = float(row['Close'])
            upper: float = float(row['Upper_Track'])
            lower: float = float(row['Lower_Track'])

            if position == 0 and close <= lower:
                position = 1
                shares = cash / close
                cash = 0.0
            elif position == 1 and close >= upper:
                position = 0
                cash = shares * close
                shares = 0.0

            curr_val: float = cash + (shares * close)
            portfolio_values.append(curr_val)

        df['Portfolio_Value'] = portfolio_values
        return df

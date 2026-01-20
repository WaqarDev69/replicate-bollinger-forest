"""
Enhanced Bollinger Band Strategy Module.

This module implements the Enhanced Bollinger Band trading strategy based on Random Forest,
as described in Algorithm 2 of the research paper. It combines machine learning for
trend prediction with technical indicators for signal generation and risk management.
"""

from typing import List, Tuple
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from ..indicators import calculate_wma_3, calculate_bollinger_bands, calculate_atr


class EnhancedBollingerStrategy:
    """
    Implements the Enhanced Bollinger Band Strategy (Algorithm 2).

    This strategy uses a Random Forest Regressor to predict the future trend of the
    Weighted Moving Average (WMA). It combines this prediction with Bollinger Bands
    for entry signals and uses the Average True Range (ATR) for dynamic stop-loss
    and take-profit levels.

    Attributes:
        initial_capital: The starting cash amount for the simulation.
        model: The Random Forest Regressor instance used for prediction.
    """

    def __init__(self, initial_capital: float = 100000.0) -> None:
        """
        Initializes the strategy with capital and the machine learning model.

        Args:
            initial_capital: The starting capital in base currency (default: 100,000).
        """
        self.initial_capital: float = initial_capital
        self.model: RandomForestRegressor = RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        )

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Generates technical indicators and features for the machine learning model.

        Constructs the 3-day WMA, Bollinger Bands, and ATR. It also creates
        lagged features of the WMA to serve as inputs for the Random Forest
        and defines the target variable as the difference between the next day's
        WMA and the current day's WMA.

        Args:
            df: A pandas DataFrame containing historical stock data.

        Returns:
            A tuple containing:
            1. The processed DataFrame with features and target, with NaN values removed.
            2. A list of column names used as input features for the model.
        """
        df = df.copy()
        df['WMA_3'] = calculate_wma_3(df['Close'])
        df = calculate_bollinger_bands(df, d=20, k=3)
        df['ATR'] = calculate_atr(df, n=20)

        feature_cols: List[str] = ['Open', 'High', 'Low', 'Close', 'Volume']
        for i in range(6):
            col = f'WMA_3_lag_{i}'
            df[col] = df['WMA_3'].shift(i)
            feature_cols.append(col)

        df['Target_Diff'] = df['WMA_3'].shift(-1) - df['WMA_3']
        return df.dropna(), feature_cols

    def run(self, df: pd.DataFrame, split_date: str) -> pd.DataFrame:
        """
        Executes the training and trading simulation.

        This method performs the following steps:
        1. Prepares features and splits data into training and testing sets.
        2. Trains the Random Forest model on pre-split data.
        3. Predicts the WMA trend for the test period.
        4. Simulates trading based on Algorithm 2:
           - Long Entry: Predicted WMA <= Lower Track.
           - Short Entry: Predicted WMA >= Upper Track.
           - Exit: Based on ATR stop-loss or Bollinger Band take-profit levels.

        Args:
            df: A pandas DataFrame containing historical stock data.
            split_date: The date string (YYYY-MM-DD) to split train and test data.

        Returns:
            The test set DataFrame with added 'Predicted_Diff', 'Predicted_WMA_Next',
            and 'Portfolio_Value' columns.

        Raises:
            ValueError: If the test set is empty after splitting.
        """
        df_processed, feature_cols = self.prepare_features(df)

        train_data: pd.DataFrame = df_processed[df_processed.index < split_date]
        test_data: pd.DataFrame = df_processed[df_processed.index >= split_date].copy()

        if len(test_data) == 0:
            raise ValueError("No test data available.")

        X_train: pd.DataFrame = train_data[feature_cols]
        y_train: pd.Series = train_data['Target_Diff']
        self.model.fit(X_train, y_train)

        X_test: pd.DataFrame = test_data[feature_cols]
        test_data['Predicted_Diff'] = self.model.predict(X_test)
        test_data['Predicted_WMA_Next'] = test_data['WMA_3'] + test_data['Predicted_Diff']

        cash: float = self.initial_capital
        position: int = 0
        entry_price: float = 0.0
        shares: float = 0.0
        portfolio_values: List[float] = []

        for i in range(len(test_data)):
            row = test_data.iloc[i]

            pred_wma: float = float(row['Predicted_WMA_Next'])
            upper: float = float(row['Upper_Track'])
            lower: float = float(row['Lower_Track'])
            atr: float = float(row['ATR'])
            close: float = float(row['Close'])

            if position == 0 and pred_wma <= lower:
                position = 1
                entry_price = close
                shares = cash / close
                cash = 0.0
            elif position == 0 and pred_wma >= upper:
                position = -1
                entry_price = close
                shares = self.initial_capital / close
                cash = self.initial_capital + (shares * close)
            elif position == 1:
                stop_loss = entry_price - (3 * atr)
                if pred_wma < stop_loss or pred_wma > upper:
                    position = 0
                    cash = shares * close
                    shares = 0.0
            elif position == -1:
                stop_loss = entry_price + (3 * atr)
                if pred_wma > stop_loss or pred_wma < lower:
                    position = 0
                    cost = shares * close
                    cash = cash - cost
                    shares = 0.0

            if position == 0:
                curr_val = cash
            elif position == 1:
                curr_val = shares * close
            elif position == -1:
                curr_val = cash - (shares * close)

            portfolio_values.append(curr_val)

        test_data['Portfolio_Value'] = portfolio_values
        return test_data

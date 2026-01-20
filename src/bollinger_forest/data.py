"""
Data management module for the Bollinger Forest package.

This module handles the retrieval, caching, and loading of historical stock market data.
It interfaces with Yahoo Finance to download data and stores it locally to minimize
network requests and improve performance on subsequent runs.
"""

import pandas as pd
import yfinance as yf
from pathlib import Path

DATA_DIR: Path = Path("data")


def ensure_data_dir() -> None:
    """
    Ensures that the local data directory exists.
    Creates the directory and any necessary parent directories if they do not exist.
    """
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True)


def get_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Retrieves historical stock data for a given ticker within a specified date range.

    This function first checks a local cache (CSV files in the 'data/' directory).
    If the data exists locally, it is loaded and filtered by the requested dates.
    If not, the data is downloaded from Yahoo Finance, cleaned, saved to the cache,
    and then returned.

    Args:
        ticker: The stock symbol (e.g., 'AAPL', '2888.HK').
        start_date: The start date for the data in 'YYYY-MM-DD' format.
        end_date: The end date for the data in 'YYYY-MM-DD' format.

    Returns:
        A pandas DataFrame containing the historical stock data with columns
        typically including Open, High, Low, Close, and Volume.

    Raises:
        ValueError: If no data is found for the ticker or if the download fails.
    """
    ensure_data_dir()

    safe_ticker: str = ticker.replace("^", "").replace(".", "_")
    file_path: Path = DATA_DIR / f"{safe_ticker}.csv"

    if file_path.exists():
        print(f"Loading {ticker} from cache...")
        df: pd.DataFrame = pd.read_csv(file_path, index_col=0, parse_dates=True)

        df = df[(df.index >= start_date) & (df.index <= end_date)]
        return df

    print(f"Downloading {ticker} from Yahoo Finance...")
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty:
        raise ValueError(f"No data found for {ticker}")

    df.to_csv(file_path)
    return df

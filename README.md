# Bollinger Forest

A Python implementation of the "[Enhanced Bollinger Band Stock Quantitative Trading Strategy Based on Random Forest](https://www.wiserpub.com/uploads/1/20230112/e9480fe01ecc8c34944e57bde7f4f9f4.pdf)".

## Structure

*   `src/bollinger_forest`: Source code.
*   `data/`: Local cache for stock data (CSV).
*   `evaluation/`: Output plots and summary CSVs.

## Installation

1.  Clone the repository.
2.  Install in editable mode:

```bash
pip install -e .
```

## Usage

Run the command line tool:

```bash
bollinger-forest --tickers 2888.HK 0005.HK 0939.HK --start 2011-01-01 --end 2021-12-31
```

### Arguments

*   `--tickers`: List of Yahoo Finance tickers (default: 2888.HK 0005.HK).
*   `--start`: Start date for data (YYYY-MM-DD).
*   `--end`: End date for data.
*   `--split`: Date to split Training and Testing data (default: 2019-01-01).

## Methodology

1.  **Classical Strategy**: Buys when price < Lower Band, Sells when price > Upper Band.
2.  **Enhanced Strategy**:
    *   Uses Random Forest to predict the 3-day Weighted Moving Average (WMA) change.
    *   Uses Predicted WMA to trigger Bollinger Band signals.
    *   Uses ATR (Average True Range) for Stop Loss logic.

### 4. How to Run

1.  Create the folder structure as shown above.
2.  Paste the code into the respective files.
3.  Open your terminal in the root folder.
4.  Install: `pip install -e .`
5.  Run: `bollinger-forest --tickers 2888.HK 0005.HK`

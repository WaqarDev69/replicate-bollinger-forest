"""
Command Line Interface (CLI) for the Bollinger Forest package.

This module serves as the entry point for running the trading strategy simulations.
It handles data fetching, model execution, performance evaluation, and result visualization.
"""

import argparse
import traceback
from pathlib import Path
from typing import List, Dict, Tuple, Any

import pandas as pd
import matplotlib.pyplot as plt

from .data import get_stock_data
from .models.classical import ClassicalBollingerStrategy
from .models.enhanced import EnhancedBollingerStrategy
from .indicators import calculate_max_drawdown, calculate_sharpe_ratio, calculate_bollinger_bands

EVAL_DIR: Path = Path("evaluation")


def ensure_eval_dir() -> None:
    """
    Creates the evaluation directory if it does not exist.
    """
    if not EVAL_DIR.exists():
        EVAL_DIR.mkdir(parents=True)


def plot_results(ticker: str, classical_res: pd.DataFrame, enhanced_res: pd.DataFrame) -> None:
    """
    Generates and saves a comparison plot between Enhanced, Classical, and Buy & Hold strategies.

    The plot aligns the timeframes of all strategies to the test period of the Enhanced strategy.
    It saves the resulting figure as a PNG file in the evaluation directory.

    Args:
        ticker: The stock symbol associated with the results.
        classical_res: DataFrame containing simulation results from the Classical strategy.
        enhanced_res: DataFrame containing simulation results from the Enhanced strategy.
    """
    plt.figure(figsize=(12, 6))

    common_idx: pd.Index = enhanced_res.index

    plt.plot(common_idx, enhanced_res['Portfolio_Value'], label='Enhanced Strategy', color='blue')

    cls_filtered: pd.DataFrame = classical_res.reindex(common_idx, method='ffill')
    plt.plot(cls_filtered.index, cls_filtered['Portfolio_Value'], label='Classical Strategy', color='orange', linestyle='--')

    initial_capital: float = 100000.0
    start_price: float = enhanced_res['Close'].iloc[0]
    buy_hold: pd.Series = (enhanced_res['Close'] / start_price) * initial_capital
    plt.plot(common_idx, buy_hold, label='Buy & Hold', color='gray', alpha=0.4)

    plt.title(f"Strategy Comparison: {ticker}")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    safe_ticker: str = ticker.replace(".", "_")
    plt.savefig(EVAL_DIR / f"{safe_ticker}_comparison.png")
    plt.close()


def main() -> None:
    """
    Main execution function.

    Parses command line arguments, iterates through requested tickers,
    runs both trading strategies, calculates performance metrics,
    and saves the results to CSV and PNG files.
    """
    parser = argparse.ArgumentParser(description="Run Bollinger Forest Strategies")
    parser.add_argument("--tickers", nargs="+", default=["2888.HK", "0005.HK"], help="List of stock tickers")
    parser.add_argument("--start", default="2011-01-01", help="Start Date")
    parser.add_argument("--end", default="2021-12-31", help="End Date")
    parser.add_argument("--split", default="2019-01-01", help="Train/Test Split Date")

    args: argparse.Namespace = parser.parse_args()
    ensure_eval_dir()

    results: List[Dict[str, Any]] = []

    for ticker in args.tickers:
        print(f"\nProcessing {ticker}...")
        try:
            df: pd.DataFrame = get_stock_data(ticker, args.start, args.end)

            df = calculate_bollinger_bands(df, d=20, k=3)

            df_test_classical: pd.DataFrame = df[df.index >= args.split].copy()

            if df_test_classical.empty:
                print(f"No test data for {ticker}")
                continue

            c_strat = ClassicalBollingerStrategy(initial_capital=100000)
            c_res: pd.DataFrame = c_strat.run(df_test_classical)

            e_strat = EnhancedBollingerStrategy(initial_capital=100000)
            e_res: pd.DataFrame = e_strat.run(df, args.split)

            def get_metrics(res_df: pd.DataFrame) -> Tuple[float, float, float]:
                """
                Calculates Return, Max Drawdown, and Sharpe Ratio for a given simulation result.

                Args:
                    res_df: DataFrame containing the 'Portfolio_Value' column.

                Returns:
                    A tuple containing (Total Return %, Max Drawdown %, Sharpe Ratio).
                """
                if res_df.empty:
                    return 0.0, 0.0, 0.0

                initial: float = 100000.0
                final: float = res_df['Portfolio_Value'].iloc[-1]

                ret: float = ((final - initial) / initial) * 100.0
                dd: float = calculate_max_drawdown(res_df['Portfolio_Value'])
                sharpe: float = calculate_sharpe_ratio(res_df['Portfolio_Value'])
                return ret, dd, sharpe

            c_ret, c_dd, c_sharpe = get_metrics(c_res)
            e_ret, e_dd, e_sharpe = get_metrics(e_res)

            results.append({
                "Ticker": ticker,
                "Classical Return %": round(c_ret, 2),
                "Enhanced Return %": round(e_ret, 2),
                "Classical DD %": round(c_dd, 2),
                "Enhanced DD %": round(e_dd, 2),
                "Classical Sharpe": round(c_sharpe, 2),
                "Enhanced Sharpe": round(e_sharpe, 2)
            })

            plot_results(ticker, c_res, e_res)
            print(f"  -> Enhanced: {e_ret:.2f}% | Classical: {c_ret:.2f}%")

        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            traceback.print_exc()

    if results:
        res_df: pd.DataFrame = pd.DataFrame(results)
        print("\n=== Final Results ===")
        print(res_df.to_string(index=False))
        res_df.to_csv(EVAL_DIR / "results_summary.csv", index=False)
        print(f"\nResults saved to {EVAL_DIR}")


if __name__ == "__main__":
    main()

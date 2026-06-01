"""
Print the full per-vegetable validation log to stdout — screenshot this for the report.

Usage (from veggieezee/):
  python predict/ml/validate_next_day.py
  python predict/ml/print_validation_log.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ML_DIR = Path(__file__).resolve().parent
CSV = ML_DIR / 'validation_per_vegetable_summary.csv'


def main() -> None:
    if not CSV.is_file():
        print('Run validate_next_day.py first.', file=sys.stderr)
        sys.exit(1)

    import numpy as np

    df = pd.read_csv(CSV, encoding='utf-8-sig')
    pairs = int(df['forecast_days'].sum())
    mae = rmse = mape = 0.0
    results_path = ML_DIR / 'validation_next_day_results.csv'
    if results_path.is_file():
        r = pd.read_csv(results_path, encoding='utf-8-sig')
        mae = float(r['abs_error_npr'].mean())
        rmse = float(np.sqrt((r['error_npr'] ** 2).mean()))
        mape = float(r['pct_error'].mean())

    print(f'=== {len(df)} vegetables (Nepali name | MAPE % | MAE NPR | days) ===')
    print(f'Overall MAE Rs.{mae:.2f} | RMSE Rs.{rmse:.2f} | MAPE {mape:.2f}%')
    print(f'Forecast pairs: {pairs:,}')
    print()
    for i, row in df.iterrows():
        print(
            f'{i + 1:2d}. {row["vegetable_nepali"]}  |  '
            f'MAPE {row["mean_pct_error"]:.2f}%  |  '
            f'MAE Rs.{row["mean_abs_error"]:.2f}  |  '
            f'n={int(row["forecast_days"])}'
        )


if __name__ == '__main__':
    main()

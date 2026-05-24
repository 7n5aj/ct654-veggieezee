"""
Sir's validation flow (next-day forecast):
  1. Use existing trained XGBoost model (no daily retrain).
  2. For each day in the validation window: today's data -> predict tomorrow.
  3. Compare tomorrow's prediction with tomorrow's actual Kalimati price.

Usage (from veggieezee/):
  python predict/ml/validate_next_day.py

Outputs:
  predict/ml/validation_next_day_results.csv
  predict/ml/validation_predicted_vs_actual.png
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'veggieezee.settings')

import django  # noqa: E402

django.setup()

from veggieezee.predict_service import (  # noqa: E402
    _ensure_kalimati_maps,
    _load_xgboost_model,
    build_xgboost_features,
    get_training_vegetable_name,
)

ML_DIR = Path(__file__).resolve().parent
DATA_PATH = ML_DIR / 'kalimati_vegetable_prices_last_30_days.csv'
NEPALI_MAP_PATH = ML_DIR / 'kalimati_nepali_to_english.csv'
RESULTS_CSV = ML_DIR / 'validation_next_day_results.csv'
CHART_PATH = ML_DIR / 'validation_predicted_vs_actual.png'
TRAIN_DAYS = 15


def _price_key(min_p: float, max_p: float, avg_p: float) -> tuple:
    return (round(float(min_p), 2), round(float(max_p), 2), round(float(avg_p), 2))


def build_nepali_to_english_map(save: bool = True) -> dict[str, str]:
    """
    Kalimati Nepali export uses Devanagari names; the model maps English API names.
    Pair EN/NP rows from the live API by (min, max, avg) — names are stable day to day.
    """
    import cloudscraper

    scraper = cloudscraper.create_scraper()
    en = scraper.get(
        'https://kalimatimarket.gov.np/api/daily-prices/en', timeout=30
    ).json()['prices']
    np = scraper.get(
        'https://kalimatimarket.gov.np/api/daily-prices/np', timeout=30
    ).json()['prices']

    mapping: dict[str, str] = {}
    if len(en) == len(np) and all(
        abs(float(en[i]['avgprice']) - float(np[i]['avgprice'])) < 0.01
        for i in range(len(en))
    ):
        for erow, nrow in zip(en, np):
            mapping[str(nrow['commodityname']).strip()] = str(erow['commodityname']).strip()
    else:
        en_by_price = {}
        for row in en:
            key = _price_key(row['minprice'], row['maxprice'], row['avgprice'])
            en_by_price.setdefault(key, []).append(row['commodityname'])
        for row in np:
            key = _price_key(row['minprice'], row['maxprice'], row['avgprice'])
            candidates = en_by_price.get(key, [])
            if len(candidates) == 1:
                mapping[str(row['commodityname']).strip()] = candidates[0]

    if save and mapping:
        pd.DataFrame(
            [{'nepali': k, 'english': v} for k, v in sorted(mapping.items())]
        ).to_csv(NEPALI_MAP_PATH, index=False, encoding='utf-8-sig')
        print(f'Nepali->English map: {len(mapping)} commodities -> {NEPALI_MAP_PATH}')

    return mapping


def _load_nepali_to_english() -> dict[str, str]:
    if NEPALI_MAP_PATH.is_file():
        df = pd.read_csv(NEPALI_MAP_PATH, encoding='utf-8-sig')
        return dict(zip(df['nepali'].astype(str).str.strip(), df['english'].astype(str).str.strip()))
    return build_nepali_to_english_map(save=True)


def _resolve_training_name(commodity: str, nepali_map: dict[str, str]) -> str | None:
    english = nepali_map.get(str(commodity).strip())
    if english:
        return get_training_vegetable_name(english)
    return get_training_vegetable_name(commodity)


def _load_kalimati_30d() -> pd.DataFrame:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f'Missing {DATA_PATH}. Copy kalimati_vegetable_prices_last_30_days.csv here.'
        )
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df = df.rename(
        columns={
            'commodity': 'commodity',
            'avg_price_npr': 'average',
            'min_price_npr': 'minimum',
            'max_price_npr': 'maximum',
        }
    )
    df = df.sort_values(['commodity', 'date']).reset_index(drop=True)
    return df


def _historical_for_veg(
    veg_df: pd.DataFrame, before_date: pd.Timestamp
) -> list[dict]:
    past = veg_df[veg_df['date'] < before_date]
    return [
        {'date': r['date'].strftime('%Y-%m-%d'), 'price_npr': float(r['average'])}
        for _, r in past.iterrows()
    ]


def _live_from_today(veg_df: pd.DataFrame, today: pd.Timestamp) -> dict | None:
    row = veg_df[veg_df['date'] == today]
    if row.empty:
        return None
    r = row.iloc[-1]
    return {
        'avg_price': float(r['average']),
        'min_price': float(r['minimum']),
        'max_price': float(r['maximum']),
    }


def run_validation() -> pd.DataFrame:
    _ensure_kalimati_maps()
    model, label_encoder = _load_xgboost_model()
    if model is None or label_encoder is None:
        raise RuntimeError('XGBoost model or label encoder not found in models/.')

    raw = _load_kalimati_30d()
    all_dates = sorted(raw['date'].unique())
    if len(all_dates) < TRAIN_DAYS + 1:
        raise ValueError(f'Need at least {TRAIN_DAYS + 1} days; got {len(all_dates)}.')

    split_date = all_dates[TRAIN_DAYS - 1]
    validation_dates = [d for d in all_dates if d > split_date]
    print(f'Dataset: {all_dates[0].date()} -> {all_dates[-1].date()} ({len(all_dates)} days)')
    print(f'Train context: first {TRAIN_DAYS} days (through {split_date.date()})')
    print(f'Validation: {len(validation_dates)} next-day forecasts ({validation_dates[0].date()} onward)')
    print('Using existing model — no retrain.\n')

    nepali_map = _load_nepali_to_english()
    rows = []
    by_commodity = {c: raw[raw['commodity'] == c] for c in raw['commodity'].unique()}
    mapped_count = sum(1 for c in by_commodity if _resolve_training_name(c, nepali_map))
    print(f'Commodities mapped to model classes: {mapped_count}/{len(by_commodity)}\n')

    for tomorrow in validation_dates:
        today = tomorrow - pd.Timedelta(days=1)
        for commodity, veg_df in by_commodity.items():
            training_name = _resolve_training_name(commodity, nepali_map)
            if training_name is None:
                continue

            actual_row = veg_df[veg_df['date'] == tomorrow]
            if actual_row.empty:
                continue
            actual = float(actual_row.iloc[0]['average'])

            historical = _historical_for_veg(veg_df, tomorrow)
            live = _live_from_today(veg_df, today)
            if live is None and not historical:
                continue

            try:
                X = build_xgboost_features(
                    tomorrow, training_name, historical, live, label_encoder
                )
                log_pred = float(model.predict(X)[0])
                predicted = float(np.expm1(log_pred))  # model trained on log1p(price)
            except Exception:
                continue

            err = predicted - actual
            rows.append(
                {
                    'today': today.date().isoformat(),
                    'predict_date': tomorrow.date().isoformat(),
                    'commodity': commodity,
                    'training_name': training_name,
                    'predicted_npr': round(predicted, 2),
                    'actual_npr': round(actual, 2),
                    'error_npr': round(err, 2),
                    'abs_error_npr': round(abs(err), 2),
                    'pct_error': round(abs(err) / actual * 100, 2) if actual else None,
                }
            )

    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError('No predictions generated — check commodity name mapping.')

    # Sir wants Nepali commodity names in the report (English is model-internal only).
    results = results.rename(columns={'commodity': 'nepali_name'})
    report_cols = [
        'nepali_name',
        'today',
        'predict_date',
        'predicted_npr',
        'actual_npr',
        'error_npr',
        'abs_error_npr',
        'pct_error',
    ]
    results[report_cols].to_csv(RESULTS_CSV, index=False, encoding='utf-8-sig')
    mae = results['abs_error_npr'].mean()
    rmse = np.sqrt((results['error_npr'] ** 2).mean())
    mape = results['pct_error'].mean()
    print(f'Predictions: {len(results):,} (Nepali commodity-day pairs)')
    print(f'Commodity labels: Nepali names from Kalimati dataset')
    print(f'MAE:  Rs.{mae:.2f}')
    print(f'RMSE: Rs.{rmse:.2f}')
    print(f'MAPE: {mape:.2f}%')
    print(f'Saved: {RESULTS_CSV}')
    return results


def _setup_nepali_plot_font():
    """Use a Devanagari-capable font (Colab/Linux often lack Nirmala/Mangal)."""
    import warnings
    from matplotlib import font_manager
    import matplotlib.pyplot as plt

    plt.rcParams['axes.unicode_minus'] = False
    font_path = ML_DIR / 'NotoSansDevanagari-Regular.ttf'
    if not font_path.is_file():
        try:
            import urllib.request

            url = (
                'https://raw.githubusercontent.com/googlefonts/noto-fonts/main/'
                'hinted/ttf/NotoSansDevanagari/NotoSansDevanagari-Regular.ttf'
            )
            urllib.request.urlretrieve(url, font_path)
        except Exception as exc:
            warnings.warn(f'Could not download Nepali font ({exc}); legend may show boxes.')

    if font_path.is_file():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams['font.family'] = 'Noto Sans Devanagari'
    else:
        plt.rcParams['font.family'] = ['Nirmala UI', 'Mangal', 'DejaVu Sans']


def _plot_results(results: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('Install matplotlib to generate chart: pip install matplotlib')
        return

    import warnings

    warnings.filterwarnings('ignore', message='.*categorical units.*', category=UserWarning)

    _setup_nepali_plot_font()

    name_col = 'nepali_name' if 'nepali_name' in results.columns else 'commodity'
    plot_df = results.copy()
    plot_df['predict_date'] = pd.to_datetime(plot_df['predict_date'])

    daily = (
        plot_df.groupby('predict_date')[['predicted_npr', 'actual_npr']]
        .mean()
        .reset_index()
        .sort_values('predict_date')
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(daily['predict_date'], daily['actual_npr'], 'o-', label='Actual', color='#2e7d32')
    axes[0].plot(daily['predict_date'], daily['predicted_npr'], 's--', label='Predicted', color='#ef6c00')
    axes[0].set_title('Next-day forecast: market average (all vegetables)')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Price (NPR)')
    axes[0].legend()
    axes[0].tick_params(axis='x', rotation=45)

    top_vegs = (
        results.groupby(name_col)['abs_error_npr']
        .count()
        .sort_values(ascending=False)
        .head(3)
        .index.tolist()
    )
    for nepali_name in top_vegs:
        sub = plot_df[plot_df[name_col] == nepali_name].sort_values('predict_date')
        axes[1].plot(sub['predict_date'], sub['actual_npr'], 'o-', label=f'{nepali_name} actual')
        axes[1].plot(sub['predict_date'], sub['predicted_npr'], 's--', label=f'{nepali_name} pred')

    axes[1].set_title('Sample vegetables: predicted vs actual')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Price (NPR)')
    axes[1].legend(fontsize=8)
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=120)
    plt.close()
    print(f'Saved: {CHART_PATH}')


if __name__ == '__main__':
    out = run_validation()
    _plot_results(out)

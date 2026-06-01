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
  predict/ml/validation_accurate_vs_high_error.png
  predict/ml/validation_overview_with_log.png
  predict/ml/validation_top5_accurate_with_log.png
  predict/ml/validation_good_vs_bad_with_log.png
  predict/ml/validation_all_vegetables_mape_with_log.png
  predict/ml/validation_error_analysis_with_log.png
  predict/ml/validation_per_vegetable_summary.csv
  predict/ml/validation_per_vegetable_report.txt
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
EXAMPLES_CHART_PATH = ML_DIR / 'validation_accurate_vs_high_error.png'
OVERVIEW_LOG_PATH = ML_DIR / 'validation_overview_with_log.png'
TOP5_ACCURATE_LOG_PATH = ML_DIR / 'validation_top5_accurate_with_log.png'
GOOD_BAD_LOG_PATH = ML_DIR / 'validation_good_vs_bad_with_log.png'
ALL_VEG_LOG_PATH = ML_DIR / 'validation_all_vegetables_mape_with_log.png'
ALL_VEG_CHART_PATH = ML_DIR / 'validation_all_vegetables_mape_chart.png'
OVERVIEW_CHART_PATH = ML_DIR / 'validation_overview_chart.png'
TOP5_CHART_PATH = ML_DIR / 'validation_top5_accurate_chart.png'
GOOD_BAD_CHART_PATH = ML_DIR / 'validation_good_vs_bad_chart.png'
ERRORS_CHART_ONLY_PATH = ML_DIR / 'validation_error_analysis_chart.png'
ERRORS_LOG_PATH = ML_DIR / 'validation_error_analysis_with_log.png'
ERRORS_CHART_PATH = ML_DIR / 'validation_error_analysis.png'
PER_VEG_SUMMARY_CSV = ML_DIR / 'validation_per_vegetable_summary.csv'
PER_VEG_REPORT_TXT = ML_DIR / 'validation_per_vegetable_report.txt'
TRAIN_DAYS = 15
TOP_ACCURATE_COUNT = 5


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


def _prepare_plot_fonts():
    """
    English labels use DejaVu Sans; Nepali vegetable names use Noto Devanagari in legends only.
    (Setting Noto globally breaks English titles — it has no Latin glyphs.)
    """
    import warnings
    from matplotlib import font_manager
    from matplotlib.font_manager import FontProperties
    import matplotlib.pyplot as plt

    plt.rcParams['font.family'] = 'DejaVu Sans'
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
        return FontProperties(fname=str(font_path))
    return None


def _apply_nepali_legend(legend, nepali_font) -> None:
    if legend is None or nepali_font is None:
        return
    for text in legend.get_texts():
        text.set_fontproperties(nepali_font)


def _vegetable_stats(results: pd.DataFrame, name_col: str) -> pd.DataFrame:
    """Per-vegetable MAE, MAPE, and forecast count (sorted best MAPE first)."""
    stats = (
        results.groupby(name_col)
        .agg(
            mean_abs_error=('abs_error_npr', 'mean'),
            mean_pct_error=('pct_error', 'mean'),
            forecast_days=('abs_error_npr', 'count'),
        )
        .reset_index()
        .rename(columns={name_col: 'vegetable_nepali'})
    )
    return stats.sort_values(['mean_pct_error', 'mean_abs_error']).reset_index(drop=True)


def _pick_example_vegetables(
    stats: pd.DataFrame, min_forecasts: int = 10
) -> tuple[str, str, pd.Series | None, pd.Series | None]:
    """Pick one low-error and one high-error vegetable (enough forecast days each)."""
    eligible = stats[stats['forecast_days'] >= min_forecasts]
    if len(eligible) < 2:
        eligible = stats.head(min(2, len(stats)))
    if eligible.empty:
        raise ValueError('No vegetables in validation results')

    best_row = eligible.iloc[0]
    worst_row = eligible.iloc[-1]
    if best_row['vegetable_nepali'] == worst_row['vegetable_nepali'] and len(eligible) >= 2:
        worst_row = eligible.iloc[1]
    return (
        best_row['vegetable_nepali'],
        worst_row['vegetable_nepali'],
        best_row,
        worst_row,
    )


def _pick_top_accurate(stats: pd.DataFrame, n: int = 5, min_forecasts: int = 10) -> pd.DataFrame:
    eligible = stats[stats['forecast_days'] >= min_forecasts]
    if eligible.empty:
        eligible = stats
    return eligible.head(min(n, len(eligible)))


def _overall_metrics(results: pd.DataFrame) -> dict:
    mae = float(results['abs_error_npr'].mean())
    rmse = float(np.sqrt((results['error_npr'] ** 2).mean()))
    mape = float(results['pct_error'].mean())
    return {
        'pairs': len(results),
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'veg_count': results['nepali_name' if 'nepali_name' in results.columns else 'commodity'].nunique(),
    }


def _build_run_log(
    results: pd.DataFrame,
    stats: pd.DataFrame,
    metrics: dict,
    *,
    highlight: pd.DataFrame | None = None,
    title: str = 'Next-day validation log',
) -> str:
    """Console-style log text rendered under charts (graph + log screenshot style)."""
    name_col = 'nepali_name' if 'nepali_name' in results.columns else 'commodity'
    lines = [
        f'=== {title} ===',
        'Dataset: 30-day Kalimati export | Model: existing XGBoost (no retrain)',
        f'Validation days scored: 15 | Forecast pairs: {metrics["pairs"]:,}',
        f'Vegetables with forecasts: {metrics["veg_count"]}',
        f'MAE:  Rs.{metrics["mae"]:.2f}  |  RMSE: Rs.{metrics["rmse"]:.2f}  |  MAPE: {metrics["mape"]:.2f}%',
        '',
    ]
    if highlight is not None and not highlight.empty:
        lines.append('Vegetables in this figure (Nepali name | MAPE % | MAE NPR | days):')
        for i, row in highlight.iterrows():
            lines.append(
                f'  {i + 1}. {row["vegetable_nepali"]}  |  '
                f'MAPE {row["mean_pct_error"]:.2f}%  |  '
                f'MAE Rs.{row["mean_abs_error"]:.2f}  |  '
                f'n={int(row["forecast_days"])}'
            )
        lines.append('')
    lines.append(f'All vegetables ranked by MAPE ({len(stats)} total) — see {PER_VEG_SUMMARY_CSV.name}')
    return '\n'.join(lines)


def _render_log_axis(fig, log_text: str, nepali_font=None) -> None:
    """Bottom panel: monospace log (like a Colab screenshot)."""
    log_ax = fig.add_axes([0.04, 0.02, 0.92, 0.14])
    log_ax.axis('off')
    log_ax.set_facecolor('#1e1e1e')
    fig.patch.set_facecolor('#fafafa')
    t = log_ax.text(
        0.01,
        0.98,
        log_text,
        transform=log_ax.transAxes,
        fontsize=7,
        family='monospace',
        color='#d4d4d4',
        va='top',
        ha='left',
        wrap=True,
    )
    if nepali_font is not None:
        t.set_fontproperties(nepali_font)


def _save_per_vegetable_report(stats: pd.DataFrame, metrics: dict) -> None:
    stats.to_csv(PER_VEG_SUMMARY_CSV, index=False, encoding='utf-8-sig')

    lines = [
        'Per-vegetable next-day validation (Kalimati, Nepali names)',
        f'Overall: MAE Rs.{metrics["mae"]:.2f}, RMSE Rs.{metrics["rmse"]:.2f}, MAPE {metrics["mape"]:.2f}%',
        '',
        'How to write in the report:',
        'For each vegetable below, one line in the narrative or one row in Table 3.x.',
        'Template: "<Nepali name> — average next-day MAPE X%, MAE Rs.Y over Z forecast days; '
        'predicted prices followed actuals [closely / with larger gaps on some days]."',
        '',
        '--- All vegetables (best to worst MAPE) ---',
    ]
    for _, row in stats.iterrows():
        quality = 'accurate' if row['mean_pct_error'] < metrics['mape'] else 'higher error'
        lines.append(
            f'{row["vegetable_nepali"]}\t'
            f'MAPE {row["mean_pct_error"]:.2f}%\t'
            f'MAE Rs.{row["mean_abs_error"]:.2f}\t'
            f'days={int(row["forecast_days"])}\t'
            f'({quality} vs market average)'
        )
    PER_VEG_REPORT_TXT.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Saved: {PER_VEG_SUMMARY_CSV}')
    print(f'Saved: {PER_VEG_REPORT_TXT}')


def _plot_veg_forecast_panel(
    ax,
    plot_df: pd.DataFrame,
    name_col: str,
    nepali_name: str,
    title: str,
    line_color: str,
    nepali_font,
) -> None:
    from matplotlib.lines import Line2D

    sub = plot_df[plot_df[name_col] == nepali_name].sort_values('predict_date')
    ax.plot(
        sub['predict_date'],
        sub['actual_npr'],
        'o-',
        color=line_color,
        label='Actual',
        linewidth=2,
        markersize=5,
    )
    ax.plot(
        sub['predict_date'],
        sub['predicted_npr'],
        's--',
        color='#ef6c00',
        label='Predicted',
        linewidth=2,
        markersize=5,
    )
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (NPR)')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(
        handles=[
            Line2D([0], [0], color=line_color, linestyle='-', marker='o', label='Actual'),
            Line2D([0], [0], color='#ef6c00', linestyle='--', marker='s', label='Predicted'),
        ],
        fontsize=8,
        loc='best',
    )
    veg_label = ax.text(
        0.02,
        0.98,
        nepali_name,
        transform=ax.transAxes,
        fontsize=8,
        va='top',
        ha='left',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='#ccc'),
    )
    if nepali_font is not None:
        veg_label.set_fontproperties(nepali_font)


def _plot_results(
    plot_df: pd.DataFrame,
    name_col: str,
    stats: pd.DataFrame,
    metrics: dict,
    nepali_font,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('Install matplotlib to generate chart: pip install matplotlib')
        return

    import warnings

    warnings.filterwarnings('ignore', message='.*categorical units.*', category=UserWarning)
    warnings.filterwarnings('ignore', message='.*Glyph.*missing from font.*', category=UserWarning)

    daily = (
        plot_df.groupby('predict_date')[['predicted_npr', 'actual_npr']]
        .mean()
        .reset_index()
        .sort_values('predict_date')
    )

    good_name, bad_name, good_row, bad_row = _pick_example_vegetables(stats)
    top5 = _pick_top_accurate(stats, n=TOP_ACCURATE_COUNT)

    def _mape_label(row: pd.Series | None) -> str:
        if row is None:
            return ''
        return f"MAPE {row['mean_pct_error']:.1f}% · MAE Rs.{row['mean_abs_error']:.1f}"

    # --- Legacy PNGs (no log strip) ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].plot(daily['predict_date'], daily['actual_npr'], 'o-', label='Actual', color='#2e7d32')
    axes[0].plot(daily['predict_date'], daily['predicted_npr'], 's--', label='Predicted', color='#ef6c00')
    axes[0].set_title('Market average (all vegetables)')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Price (NPR)')
    axes[0].legend()
    axes[0].tick_params(axis='x', rotation=45)
    _plot_veg_forecast_panel(
        axes[1], plot_df, name_col, good_name, f'Accurate\n{_mape_label(good_row)}', '#2e7d32', nepali_font
    )
    _plot_veg_forecast_panel(
        axes[2], plot_df, name_col, bad_name, f'Higher error\n{_mape_label(bad_row)}', '#c62828', nepali_font
    )
    fig.suptitle('Next-day validation: predicted vs actual', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved: {CHART_PATH}')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    _plot_veg_forecast_panel(
        axes[0], plot_df, name_col, good_name, f'Accurate\n{_mape_label(good_row)}', '#2e7d32', nepali_font
    )
    _plot_veg_forecast_panel(
        axes[1], plot_df, name_col, bad_name, f'Higher error\n{_mape_label(bad_row)}', '#c62828', nepali_font
    )
    plt.tight_layout()
    plt.savefig(EXAMPLES_CHART_PATH, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved: {EXAMPLES_CHART_PATH}')

    # --- Report figures: chart + log together ---
    highlight_overview = pd.concat(
        [top5, pd.DataFrame([good_row, bad_row])],
        ignore_index=True,
    ).drop_duplicates(subset=['vegetable_nepali'])

    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(1, 3, left=0.06, right=0.98, top=0.88, bottom=0.2, wspace=0.28)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(daily['predict_date'], daily['actual_npr'], 'o-', label='Actual', color='#2e7d32')
    ax0.plot(daily['predict_date'], daily['predicted_npr'], 's--', label='Predicted', color='#ef6c00')
    ax0.set_title('Market average')
    ax0.set_ylabel('Price (NPR)')
    ax0.legend(fontsize=8)
    ax0.tick_params(axis='x', rotation=45)
    _plot_veg_forecast_panel(
        fig.add_subplot(gs[0, 1]), plot_df, name_col, good_name, 'Accurate example', '#2e7d32', nepali_font
    )
    _plot_veg_forecast_panel(
        fig.add_subplot(gs[0, 2]), plot_df, name_col, bad_name, 'Higher-error example', '#c62828', nepali_font
    )
    fig.suptitle('Next-day validation overview', fontweight='bold', fontsize=12)
    plt.savefig(OVERVIEW_CHART_PATH, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    _render_log_axis(
        fig,
        _build_run_log(results=plot_df, stats=stats, metrics=metrics, highlight=highlight_overview),
        nepali_font,
    )
    plt.savefig(OVERVIEW_LOG_PATH, dpi=120, facecolor=fig.get_facecolor())
    plt.close()
    print(f'Saved: {OVERVIEW_CHART_PATH}')
    print(f'Saved: {OVERVIEW_LOG_PATH} (synthetic log — prefer composite with real screenshot)')

    # Top 5 accurate — each with mini title; log lists all five
    n_top = len(top5)
    ncols = 3
    nrows = int(np.ceil(n_top / ncols))
    fig = plt.figure(figsize=(16, 4.5 * nrows + 1.8))
    gs = fig.add_gridspec(nrows, ncols, left=0.06, right=0.98, top=0.9, bottom=0.22, hspace=0.45, wspace=0.3)
    for plot_idx, (_, row) in enumerate(top5.iterrows()):
        ax = fig.add_subplot(gs[plot_idx // ncols, plot_idx % ncols])
        veg = row['vegetable_nepali']
        _plot_veg_forecast_panel(
            ax,
            plot_df,
            name_col,
            veg,
            f'#{plot_idx + 1} accurate · MAPE {row["mean_pct_error"]:.1f}%',
            '#2e7d32',
            nepali_font,
        )
    fig.suptitle(f'Top {n_top} accurate vegetables — predicted vs actual', fontweight='bold', fontsize=12)
    plt.savefig(TOP5_CHART_PATH, dpi=120, facecolor=fig.get_facecolor())
    _render_log_axis(
        fig,
        _build_run_log(
            plot_df,
            stats,
            metrics,
            highlight=top5,
            title=f'Top {n_top} accurate vegetables — validation log',
        ),
        nepali_font,
    )
    plt.savefig(TOP5_ACCURATE_LOG_PATH, dpi=120, facecolor=fig.get_facecolor())
    plt.close()
    print(f'Saved: {TOP5_CHART_PATH}')
    print(f'Saved: {TOP5_ACCURATE_LOG_PATH}')

    fig = plt.figure(figsize=(14, 6.5))
    gs = fig.add_gridspec(1, 2, left=0.06, right=0.98, top=0.88, bottom=0.2, wspace=0.25)
    _plot_veg_forecast_panel(
        fig.add_subplot(gs[0, 0]),
        plot_df,
        name_col,
        good_name,
        'Accurate — lines track closely',
        '#2e7d32',
        nepali_font,
    )
    _plot_veg_forecast_panel(
        fig.add_subplot(gs[0, 1]),
        plot_df,
        name_col,
        bad_name,
        'Higher error — larger gaps on some days',
        '#c62828',
        nepali_font,
    )
    fig.suptitle('Accurate vs higher-error', fontweight='bold', fontsize=12)
    plt.savefig(GOOD_BAD_CHART_PATH, dpi=120, facecolor=fig.get_facecolor())
    _render_log_axis(
        fig,
        _build_run_log(
            plot_df,
            stats,
            metrics,
            highlight=pd.DataFrame([good_row, bad_row]),
            title='Accurate vs higher-error comparison log',
        ),
        nepali_font,
    )
    plt.savefig(GOOD_BAD_LOG_PATH, dpi=120, facecolor=fig.get_facecolor())
    plt.close()
    print(f'Saved: {GOOD_BAD_CHART_PATH}')
    print(f'Saved: {GOOD_BAD_LOG_PATH}')

    _plot_all_vegetables_mape(plot_df, stats, metrics, nepali_font)
    _plot_errors(results=plot_df, stats=stats, metrics=metrics, nepali_font=nepali_font)


def _plot_all_vegetables_mape(
    plot_df: pd.DataFrame,
    stats: pd.DataFrame,
    metrics: dict,
    nepali_font,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    import warnings

    warnings.filterwarnings('ignore', message='.*Glyph.*missing from font.*', category=UserWarning)

    ranked = stats.sort_values('mean_pct_error', ascending=True)
    fig = plt.figure(figsize=(14, max(8, 0.22 * len(ranked) + 2)))
    ax = fig.add_axes([0.28, 0.2, 0.68, 0.68])
    colors = ['#2e7d32' if m <= metrics['mape'] else '#ef6c00' for m in ranked['mean_pct_error']]
    ax.barh(ranked['vegetable_nepali'], ranked['mean_pct_error'], color=colors, edgecolor='white')
    ax.axvline(metrics['mape'], color='#c62828', linestyle='--', linewidth=1.2, label=f'Market avg MAPE {metrics["mape"]:.1f}%')
    ax.set_xlabel('Mean MAPE per vegetable (%)')
    ax.set_title('Every vegetable — next-day forecast error (lower is better)')
    ax.legend(fontsize=8)
    if nepali_font is not None:
        for label in ax.get_yticklabels():
            label.set_fontproperties(nepali_font)
            label.set_fontsize(7)
    fig.suptitle('All vegetables — mean MAPE (next-day validation)', fontweight='bold', y=0.96)
    plt.savefig(ALL_VEG_CHART_PATH, dpi=120, facecolor=fig.get_facecolor())
    log_lines = [
        f'=== All {len(ranked)} vegetables (Nepali name | MAPE % | MAE NPR | days) ===',
        f'Overall MAE Rs.{metrics["mae"]:.2f} | RMSE Rs.{metrics["rmse"]:.2f} | MAPE {metrics["mape"]:.2f}%',
        '',
    ]
    for i, row in ranked.iterrows():
        log_lines.append(
            f'{i + 1:2d}. {row["vegetable_nepali"]}  |  MAPE {row["mean_pct_error"]:.2f}%  |  '
            f'MAE Rs.{row["mean_abs_error"]:.2f}  |  n={int(row["forecast_days"])}'
        )
    _render_log_axis(fig, '\n'.join(log_lines), nepali_font)
    plt.savefig(ALL_VEG_LOG_PATH, dpi=120, facecolor=fig.get_facecolor())
    plt.close()
    print(f'Saved: {ALL_VEG_CHART_PATH}')
    print(f'Saved: {ALL_VEG_LOG_PATH}')
    print('For real log: python predict/ml/print_validation_log.py → screenshot → docs/log_screenshots/')


def _plot_errors(
    results: pd.DataFrame,
    stats: pd.DataFrame | None = None,
    metrics: dict | None = None,
    nepali_font=None,
) -> None:
    """Error analysis charts for the report."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('Install matplotlib to generate error charts: pip install matplotlib')
        return

    import warnings

    warnings.filterwarnings('ignore', message='.*categorical units.*', category=UserWarning)
    warnings.filterwarnings('ignore', message='.*Glyph.*missing from font.*', category=UserWarning)
    _prepare_plot_fonts()  # English-only chart

    df = results.copy()
    if 'predict_date' in df.columns:
        df['predict_date'] = pd.to_datetime(df['predict_date'])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].hist(df['abs_error_npr'], bins=40, color='#5c6bc0', edgecolor='white')
    axes[0, 0].set_title('Distribution of absolute error (NPR)')
    axes[0, 0].set_xlabel('Absolute error (NPR)')
    axes[0, 0].set_ylabel('Count')

    axes[0, 1].hist(df['pct_error'].dropna(), bins=40, color='#ef5350', edgecolor='white')
    axes[0, 1].set_title('Distribution of percentage error (%)')
    axes[0, 1].set_xlabel('MAPE per forecast (%)')
    axes[0, 1].set_ylabel('Count')

    daily_err = df.groupby('predict_date')['abs_error_npr'].mean().reset_index()
    axes[1, 0].plot(daily_err['predict_date'], daily_err['abs_error_npr'], 'o-', color='#8e24aa')
    axes[1, 0].set_title('Average absolute error by day')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Mean abs error (NPR)')
    axes[1, 0].tick_params(axis='x', rotation=45)

    axes[1, 1].scatter(df['actual_npr'], df['predicted_npr'], alpha=0.35, s=18, color='#00897b')
    lim_min = min(df['actual_npr'].min(), df['predicted_npr'].min())
    lim_max = max(df['actual_npr'].max(), df['predicted_npr'].max())
    axes[1, 1].plot([lim_min, lim_max], [lim_min, lim_max], 'k--', linewidth=1, label='Perfect fit')
    axes[1, 1].set_title('Predicted vs actual (all points)')
    axes[1, 1].set_xlabel('Actual price (NPR)')
    axes[1, 1].set_ylabel('Predicted price (NPR)')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(ERRORS_CHART_PATH, dpi=120)
    plt.close()
    print(f'Saved: {ERRORS_CHART_PATH}')

    if stats is not None and metrics is not None:
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 2, left=0.07, right=0.98, top=0.88, bottom=0.2, hspace=0.35, wspace=0.28)
        for i, (r, c) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
            ax = fig.add_subplot(gs[r, c])
            if i == 0:
                ax.hist(df['abs_error_npr'], bins=40, color='#5c6bc0', edgecolor='white')
                ax.set_title('Absolute error (NPR)')
            elif i == 1:
                ax.hist(df['pct_error'].dropna(), bins=40, color='#ef5350', edgecolor='white')
                ax.set_title('Percentage error (%)')
            elif i == 2:
                ax.plot(daily_err['predict_date'], daily_err['abs_error_npr'], 'o-', color='#8e24aa')
                ax.set_title('Mean abs error by day')
                ax.tick_params(axis='x', rotation=45)
            else:
                ax.scatter(df['actual_npr'], df['predicted_npr'], alpha=0.35, s=18, color='#00897b')
                lim_min = min(df['actual_npr'].min(), df['predicted_npr'].min())
                lim_max = max(df['actual_npr'].max(), df['predicted_npr'].max())
                ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', linewidth=1)
                ax.set_title('Predicted vs actual (all pairs)')
                ax.set_xlabel('Actual (NPR)')
                ax.set_ylabel('Predicted (NPR)')
        fig.suptitle('Error analysis', fontweight='bold', fontsize=12)
        plt.savefig(ERRORS_CHART_ONLY_PATH, dpi=120, bbox_inches='tight')
        _render_log_axis(
            fig,
            _build_run_log(df, stats, metrics, title='Error analysis run log'),
            nepali_font,
        )
        plt.savefig(ERRORS_LOG_PATH, dpi=120, facecolor=fig.get_facecolor())
        plt.close()
        print(f'Saved: {ERRORS_CHART_ONLY_PATH}')
        print(f'Saved: {ERRORS_LOG_PATH}')


def plot_validation_charts(results: pd.DataFrame | None = None) -> pd.DataFrame:
    """Run validation (if needed) and save chart PNGs (with embedded log panels)."""
    if results is None:
        if RESULTS_CSV.is_file():
            results = pd.read_csv(RESULTS_CSV, encoding='utf-8-sig')
        else:
            results = run_validation()
    name_col = 'nepali_name' if 'nepali_name' in results.columns else 'commodity'
    plot_df = results.copy()
    plot_df['predict_date'] = pd.to_datetime(plot_df['predict_date'])
    stats = _vegetable_stats(plot_df, name_col)
    metrics = _overall_metrics(plot_df)
    _save_per_vegetable_report(stats, metrics)
    nepali_font = _prepare_plot_fonts()
    _plot_results(plot_df, name_col, stats, metrics, nepali_font)
    return results


if __name__ == '__main__':
    out = run_validation()
    plot_validation_charts(out)

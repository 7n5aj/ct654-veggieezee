"""
Stack chart PNG + real notebook log screenshot → one report figure.

1. Save screenshots in docs/log_screenshots/ (see README there).
2. Run: python docs/composite_figure_with_log.py
"""
from __future__ import annotations

from pathlib import Path

try:
    from PIL import Image
except ImportError:
    raise SystemExit('Install Pillow: pip install pillow')

ROOT = Path(__file__).resolve().parents[1]
ML = ROOT / 'predict' / 'ml'
ASSETS = ROOT / 'docs' / 'assets'
LOGS = Path(__file__).resolve().parent / 'log_screenshots'

# (chart file in predict/ml, log screenshot name, output file)
COMPOSITES = [
    ('validation_overview_chart.png', 'validation_run_log.png', 'validation_overview_with_real_log.png'),
    ('validation_top5_accurate_chart.png', 'validation_run_log.png', 'validation_top5_accurate_with_real_log.png'),
    ('validation_good_vs_bad_chart.png', 'validation_run_log.png', 'validation_good_vs_bad_with_real_log.png'),
    ('validation_all_vegetables_mape_chart.png', 'validation_all_vegetables_log.png', 'validation_all_vegetables_with_real_log.png'),
    ('validation_error_analysis_chart.png', 'validation_run_log.png', 'validation_error_analysis_with_real_log.png'),
]

TRAINING = (
    ASSETS / 'fig_3_9_feature_importance.png',
    LOGS / 'training_eval_log.png',
    ML / 'fig_3_9_feature_importance_with_real_log.png',
)


def stack(chart_path: Path, log_path: Path, out_path: Path, log_max_height: int = 450) -> None:
    chart = Image.open(chart_path).convert('RGB')
    log = Image.open(log_path).convert('RGB')

    target_w = chart.width
    log_ratio = log.height / max(log.width, 1)
    log_h = min(int(target_w * log_ratio), log_max_height)
    log = log.resize((target_w, log_h), Image.Resampling.LANCZOS)

    combined = Image.new('RGB', (target_w, chart.height + log_h + 10), (255, 255, 255))
    combined.paste(chart, (0, 0))
    combined.paste(log, (0, chart.height + 10))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(out_path)
    print(f'OK  {out_path.relative_to(ROOT)}')


def main() -> None:
    made = 0
    for chart_name, log_name, out_name in COMPOSITES:
        chart = ML / chart_name
        log = LOGS / log_name
        out = ML / out_name
        if not chart.is_file():
            print(f'SKIP chart missing: {chart_name}  (run validate_next_day.py)')
            continue
        if not log.is_file():
            print(f'SKIP screenshot missing: log_screenshots/{log_name}')
            continue
        stack(chart, log, out)
        made += 1

    chart, log, out = TRAINING
    if chart.is_file() and log.is_file():
        stack(chart, log, out, log_max_height=200)
        made += 1
    else:
        if not log.is_file():
            print('SKIP training: log_screenshots/training_eval_log.png')

    print(f'\nDone: {made} figure(s) with REAL logs.')
    if made:
        print('Next: python docs/build_report_guide.py')


if __name__ == '__main__':
    main()

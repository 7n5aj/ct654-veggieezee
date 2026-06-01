# Real notebook log screenshots (for report figures)

The report figures need **real Colab/Jupyter output**, not fake text drawn under the chart.

## Step 1 — Run the notebooks

| What | Where |
|------|--------|
| Training log + feature importance | `nepal_xgboost_training.ipynb` — run through **Step 8** |
| Validation log | `predict/ml/Validation_Next_Day_Report.ipynb` — run **Section 6** and **Section 8** |
| Full vegetable list log | From project root: `python predict/ml/validate_next_day.py` (prints MAE/RMSE) then `python predict/ml/print_validation_log.py` |

## Step 2 — Take screenshots (only the output / log area)

**Google Colab:** run the cell → select the **text output** below the cell → right-click → copy is not enough; use **Snipping Tool** (Win+Shift+S) and capture the grey/white output box.

**Jupyter locally:** same — capture only the printed output block.

Save files **exactly** with these names in this folder (`docs/log_screenshots/`):

| Filename | What to capture |
|----------|-----------------|
| `training_eval_log.png` | Step 7 — MAE / RMSE / MAPE block |
| `training_feature_log.png` | Step 8 — "Top 10 features" text under the chart (optional) |
| `validation_run_log.png` | Validation notebook Section 6 — MAE, RMSE, MAPE, Saved csv |
| `validation_all_vegetables_log.png` | Output of `print_validation_log.py` (full 68-veg list) |

## Step 3 — Build chart + real log figures

```powershell
cd d:\ct654-veggieezee\veggieezee
python predict/ml/validate_next_day.py
python predict/ml/print_validation_log.py
# (take screenshot → save as validation_all_vegetables_log.png)
python docs/composite_figure_with_log.py
python docs/build_report_guide.py
```

Output PNGs go to `predict/ml/*_with_real_log.png` and the HTML guide will use them.

## Step 4 — Word report

Insert `*_with_real_log.png` as Figure 3.9, 3.10a–3.10d, 3.11.

If a screenshot is missing, the compositor skips that figure and prints a warning.

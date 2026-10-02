# Freight Rate Prediction: Solution

This model predicts truckload `posted_rate` with LightGBM on log(rate). Models were compared with a rolling-origin backtest (test months Aug, Sep, Oct; each fit only on earlier months). On October it cuts MAE from $231 (distance × median RPM baseline) to $102, and MAPE from 10.9% to 4.8%. See [REPORT.md](REPORT.md) / `REPORT.docx`.

## Run (Windows PowerShell, Python 3.10+)

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train.py
python score.py --predictions validation_predictions.csv --december-predictions december-chart-inputs.csv
python make_report.py
```

`python train.py` (about 1 minute on CPU, seed fixed, output is byte-for-byte reproducible) does all of the following in one run:
- cleans the data, runs the Aug/Sep/Oct rolling backtest and the October holdout, and prints baseline vs. model metrics
- refits on all labeled data (Jan–Oct)
- writes `validation_predictions.csv` (12,000 rows, `load_id,predicted_rate`)
- fills `predicted_rate` in `december-chart-inputs.csv`. The untouched original is saved once as `december-chart-inputs.original.csv`, and predictions are always regenerated from that copy.
- writes `artifacts/backtest.csv`, `artifacts/metrics.json` and `artifacts/holdout_report.csv`

`score.py` is the official scorer (a copy of the provided `solve.py.txt`). It validates both files and writes `scorer_results/candidate_december.png`.

## Files

| File | Purpose |
|---|---|
| `train.py` | cleaning, features, time-based validation, final training, both prediction files |
| `score.py` | provided scorer (unchanged) |
| `make_report.py` | renders `REPORT.md` and the chart to `REPORT.docx` |
| `REPORT.md` / `REPORT.docx` | write-up: data issues, split, model, metrics, December method, limitations |
| `train-test.csv`, `validation.csv`, `validation-predictions-template.csv`, `december-chart-inputs.original.csv` | original inputs (not modified) |

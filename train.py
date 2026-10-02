"""Freight rate model: clean -> time-based holdout -> refit -> predict validation + December chart.

Run:  python train.py
Writes: validation_predictions.csv, december-chart-inputs.csv (filled in place),
        artifacts/metrics.json, artifacts/holdout_report.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

SEED = 42
ROOT = Path(__file__).resolve().parent
TRAIN_CSV = ROOT / "train-test.csv"
VALID_CSV = ROOT / "validation.csv"
TEMPLATE_CSV = ROOT / "validation-predictions-template.csv"
DECEMBER_CSV = ROOT / "december-chart-inputs.csv"
DECEMBER_ORIG = ROOT / "december-chart-inputs.original.csv"  # pristine copy, made on first run
OUT_DIR = ROOT / "artifacts"

HOLDOUT_START = pd.Timestamp("2025-10-01")  # train Jan-Sep, validate October
WEIGHT_MIN, WEIGHT_MAX = 5_000.0, 47_500.0  # observed legal range in the data
MARKET_LOOKBACK_DAYS = 56  # history window for imputing future market_index / quote_signal

LGB_PARAMS = dict(
    objective="regression_l1",  # L1 on log(rate) ~ median relative error, robust to outlier rates
    learning_rate=0.03,
    num_leaves=63,
    min_child_samples=30,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    n_estimators=3000,
    random_state=SEED,
    deterministic=True,
    force_row_wise=True,
    verbose=-1,
)
EARLY_STOP_ROUNDS = 150
ANOMALY_RATIO = 2.5  # rate this many x off the fitted value => label anomaly


# --------------------------------------------------------------------------- cleaning
def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_CSV, parse_dates=["date"])
    valid = pd.read_csv(VALID_CSV, parse_dates=["date"])
    return train, valid


def clean(df: pd.DataFrame, weight_fill: dict[str, float]) -> pd.DataFrame:
    """Data-quality rules (documented in REPORT.md).

    * Negative weight: sign error. The negative values mirror the positive distribution
      (median ~ -31.8k vs +31.4k), so take abs(), then clip to the observed [5k, 47.5k] range.
    * Missing weight: impute equipment-level median from the training data.
    * Missing market_index: it is effectively a daily series (within-day std 0.025 vs 0.17
      overall), so impute with the same-date mean of the same file, falling back to a
      day-of-week profile. Uses features only, never labels.
    """
    df = df.copy()
    df["weight_was_missing"] = df["weight"].isna().astype(int)
    df["weight_was_negative"] = (df["weight"] < 0).astype(int)
    df["weight"] = df["weight"].abs().clip(WEIGHT_MIN, WEIGHT_MAX)
    df["weight"] = df["weight"].fillna(df["equipment"].map(weight_fill))

    if "market_index" in df:
        day_mean = df.groupby("date")["market_index"].transform("mean")
        df["market_index"] = df["market_index"].fillna(day_mean)
        dow_mean = df.groupby(df["date"].dt.dayofweek)["market_index"].transform("mean")
        df["market_index"] = df["market_index"].fillna(dow_mean)
    return df


# --------------------------------------------------------------------------- features
def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


EQUIP_CODES = {"Dry Van": 0, "Reefer": 1, "Flatbed": 2}
BASE_FEATURES = [
    "log_distance", "distance", "equipment_code", "weight", "weight_was_missing", "weight_was_negative",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "d_lat", "d_lon", "circuity", "bearing_sin", "bearing_cos",
    "dow", "is_weekend",
]
MARKET_FEATURES = ["market_index", "quote_signal", "mi_x_logdist"]
# market level relative to its own trailing 28-day mean (removes month-to-month regime drift)
REL_MARKET_FEATURES = ["mi_rel", "qs_rel", "qs_load_rel"]
FEATURE_SETS = {
    "lgbm_with_market": BASE_FEATURES + MARKET_FEATURES,
    "lgbm_rel_market": BASE_FEATURES + REL_MARKET_FEATURES,
    "lgbm_no_market": BASE_FEATURES,
}
NEEDS_MARKET = {"lgbm_with_market", "lgbm_rel_market"}  # unusable as-is for December
BACKTEST_MONTHS = ["2025-08-01", "2025-09-01", "2025-10-01"]  # rolling origin: fit on all prior months


def featurize(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    f["distance"] = df["distance"]
    f["log_distance"] = np.log(df["distance"])
    f["equipment_code"] = df["equipment"].map(EQUIP_CODES).astype(int)
    f["weight"] = df["weight"]
    f["weight_was_missing"] = df["weight_was_missing"]
    f["weight_was_negative"] = df["weight_was_negative"]
    for c in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]:
        f[c] = df[c]
    f["d_lat"] = df["delivery_lat"] - df["pickup_lat"]
    f["d_lon"] = df["delivery_lon"] - df["pickup_lon"]
    gc = haversine_miles(df["pickup_lat"], df["pickup_lon"], df["delivery_lat"], df["delivery_lon"])
    f["circuity"] = df["distance"] / np.maximum(gc, 1.0)
    bearing = np.arctan2(f["d_lon"], f["d_lat"])
    f["bearing_sin"], f["bearing_cos"] = np.sin(bearing), np.cos(bearing)
    f["dow"] = df["date"].dt.dayofweek
    f["is_weekend"] = (f["dow"] >= 5).astype(int)
    # Lat/lon (not city IDs) carry geography, so cities unseen in training (8 in validation) still work.
    if "market_index" in df:
        f["market_index"] = df["market_index"]
        f["quote_signal"] = df["quote_signal"]
        f["mi_x_logdist"] = df["market_index"] * f["log_distance"]
    if "mi_rel" in df:
        for c in REL_MARKET_FEATURES:
            f[c] = df[c]
    return f


def add_relative_market(df: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Daily market level / trailing 28-day mean of earlier days (past-only, features only)."""
    trail = daily.shift(1).rolling(28, min_periods=7).mean()
    rel = (daily / trail).fillna(1.0)
    df = df.copy()
    df["mi_rel"] = df["date"].map(rel["market_index"]).fillna(1.0).to_numpy()
    df["qs_rel"] = df["date"].map(rel["quote_signal"]).fillna(1.0).to_numpy()
    df["qs_load_rel"] = (df["quote_signal"] / df["date"].map(daily["quote_signal"])).to_numpy()
    return df


# --------------------------------------------------------------------------- market imputation for dates without market data
def market_profile(history: pd.DataFrame) -> dict:
    """Recent level x day-of-week shape, estimated from the last MARKET_LOOKBACK_DAYS of history."""
    daily = history.groupby("date").agg(mi=("market_index", "mean"), qs=("quote_signal", "mean"))
    recent = daily[daily.index > daily.index.max() - pd.Timedelta(days=MARKET_LOOKBACK_DAYS)]
    dow = recent.groupby(recent.index.dayofweek)["mi"].mean()
    return {"mi_dow": (dow / dow.mean() * recent["mi"].mean()).to_dict(), "qs": float(recent["qs"].mean())}


def apply_market_profile(df: pd.DataFrame, profile: dict) -> pd.DataFrame:
    df = df.copy()
    df["market_index"] = df["date"].dt.dayofweek.map(profile["mi_dow"]).astype(float)
    df["quote_signal"] = profile["qs"]
    return df


# --------------------------------------------------------------------------- modeling
def fit(X: pd.DataFrame, y_log: pd.Series, n_estimators: int | None = None, eval_set=None):
    params = dict(LGB_PARAMS)
    if n_estimators is not None:
        params["n_estimators"] = n_estimators
    model = lgb.LGBMRegressor(**params)
    callbacks = [lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False)] if eval_set else None
    model.fit(X, y_log, eval_set=eval_set, callbacks=callbacks,
              categorical_feature=["equipment_code"])
    return model


def metrics(y: np.ndarray, p: np.ndarray, distance: np.ndarray) -> dict:
    err = p - y
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE_%": float(np.mean(np.abs(err) / y) * 100),
        "MedAPE_%": float(np.median(np.abs(err) / y) * 100),
        "RPM_RMSE": float(np.sqrt(np.mean((err / distance) ** 2))),
    }


def baseline_predict(train: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    """Naive baseline: distance x median rate-per-mile by equipment."""
    rpm = (train["posted_rate"] / train["distance"]).groupby(train["equipment"]).median()
    return target["distance"].to_numpy() * target["equipment"].map(rpm).to_numpy()


def anomaly_mask(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Flag label anomalies: rows whose rate is > ANOMALY_RATIO x off a robust in-sample L1 fit.

    ~1.4% of rates are ~5x too high or too low for their distance/equipment (e.g. 421 mi Reefer
    at $184). They are unlearnable label noise, so they are dropped from *training* only.
    """
    y = np.log(df["posted_rate"])
    m = fit(featurize(df)[cols], y, n_estimators=300)
    return (np.abs(y - m.predict(featurize(df)[cols])) > np.log(ANOMALY_RATIO)).to_numpy()


def train_and_predict(fit_df: pd.DataFrame, target: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, int]:
    """Early-stop on the last 30 days of fit_df (time-based), refit on all of it, predict target."""
    y = np.log(fit_df["posted_rate"])
    is_es = (fit_df["date"] > fit_df["date"].max() - pd.Timedelta(days=30)).to_numpy()
    X = featurize(fit_df)[cols]
    m = fit(X[~is_es], y[~is_es], eval_set=[(X[is_es], y[is_es])])
    n_iter = int(m.best_iteration_ or LGB_PARAMS["n_estimators"])
    m = fit(X, y, n_estimators=n_iter)
    return np.exp(m.predict(featurize(target)[cols])), n_iter


def rolling_backtest(train: pd.DataFrame) -> pd.DataFrame:
    """Rolling-origin backtest: for each month M in BACKTEST_MONTHS, fit on all months < M, test on M."""
    rows = []
    for start in BACKTEST_MONTHS:
        s = pd.Timestamp(start)
        e = s + pd.offsets.MonthEnd(0)
        fit_df = train[train["date"] < s]
        test = train[(train["date"] >= s) & (train["date"] <= e)]
        fit_df = fit_df[~anomaly_mask(fit_df, BASE_FEATURES)]
        y, d = test["posted_rate"].to_numpy(), test["distance"].to_numpy()
        preds = {"baseline": baseline_predict(fit_df, test)}
        for name, cols in FEATURE_SETS.items():
            preds[name] = train_and_predict(fit_df, test, cols)[0]
        for name, p in preds.items():
            rows.append({"test_month": start[:7], "model": name, **metrics(y, p, d)})
    return pd.DataFrame(rows)


def main() -> None:
    np.random.seed(SEED)
    OUT_DIR.mkdir(exist_ok=True)
    train_raw, valid_raw = load()
    weight_fill = train_raw.assign(weight=train_raw["weight"].abs()).groupby("equipment")["weight"].median().to_dict()
    train = clean(train_raw, weight_fill)
    valid = clean(valid_raw, weight_fill)
    daily_market = pd.concat([train, valid]).groupby("date")[["market_index", "quote_signal"]].mean()
    train = add_relative_market(train, daily_market)
    valid = add_relative_market(valid, daily_market)

    # ---------------- rolling-origin backtest (Aug, Sep, Oct) -> model selection
    bt = rolling_backtest(train)
    bt.to_csv(OUT_DIR / "backtest.csv", index=False)
    bt_mean = bt.groupby("model")[["MAE", "RMSE", "MAPE_%", "MedAPE_%", "RPM_RMSE"]].mean()
    print("Rolling backtest MAE by test month:")
    print(bt.pivot(index="model", columns="test_month", values="MAE").assign(mean=bt_mean["MAE"]).round(1))

    # ---------------- time-based holdout: fit Jan-Sep, evaluate October (no shuffling)
    is_hold = (train["date"] >= HOLDOUT_START).to_numpy()
    fit_df, hold_df = train[~is_hold], train[is_hold]
    fit_df = fit_df[~anomaly_mask(fit_df, BASE_FEATURES)]  # anomaly filter sees Jan-Sep only
    y_fit = np.log(fit_df["posted_rate"])
    # early-stopping split is also time-based: last 30 days of the fit window
    is_es = (fit_df["date"] > fit_df["date"].max() - pd.Timedelta(days=30)).to_numpy()
    y_hold = hold_df["posted_rate"].to_numpy()
    dist_hold = hold_df["distance"].to_numpy()
    # diagnostic only: October rows whose label is >2.5x off -> reported as a separate "clean" metric
    hold_anom = anomaly_mask(hold_df, BASE_FEATURES)

    results, results_clean, best_iters, holdout_preds, importance = {}, {}, {}, {}, {}

    def record(name, p):
        holdout_preds[name] = p
        results[name] = metrics(y_hold, p, dist_hold)
        results_clean[name] = metrics(y_hold[~hold_anom], p[~hold_anom], dist_hold[~hold_anom])

    record("baseline", baseline_predict(fit_df, hold_df))
    for name, cols in FEATURE_SETS.items():
        X_fit, X_hold = featurize(fit_df)[cols], featurize(hold_df)[cols]
        m = fit(X_fit[~is_es], y_fit[~is_es], eval_set=[(X_fit[is_es], y_fit[is_es])])
        best_iters[name] = int(m.best_iteration_ or LGB_PARAMS["n_estimators"])
        m = fit(X_fit, y_fit, n_estimators=best_iters[name])  # refit on full Jan-Sep
        record(name, np.exp(m.predict(X_hold)))
        importance[name] = pd.Series(m.booster_.feature_importance("gain"), index=cols).sort_values(ascending=False)
        if "market_index" in cols:
            # December-style path: market unknown -> impute from Jan-Sep history only
            prof = market_profile(train[~is_hold])
            record(name + "+imputed_market",
                   np.exp(m.predict(featurize(apply_market_profile(hold_df, prof))[cols])))

    for k in results:
        print(f"{k:32s} all  : " + "  ".join(f"{m}={x:,.2f}" for m, x in results[k].items()))
        print(f"{'':32s} clean: " + "  ".join(f"{m}={x:,.2f}" for m, x in results_clean[k].items()))

    # model selection: mean MAE over the 3 backtest months (all rows)
    best_val = min(FEATURE_SETS, key=lambda k: bt_mean.loc[k, "MAE"])
    # December has no market data: best market-free model, unless a market model with
    # history-imputed inputs beats it on October (the only fold where imputation was simulated)
    best_dec = min((k for k in FEATURE_SETS if k not in NEEDS_MARKET), key=lambda k: bt_mean.loc[k, "MAE"])
    for k in [k for k in results if k.endswith("+imputed_market")]:
        if results[k]["MAE"] < results[best_dec]["MAE"]:
            best_dec = k
    print("selected for validation:", best_val, "| selected for December:", best_dec, "| iters:", best_iters)

    pd.DataFrame({"load_id": hold_df["load_id"], "date": hold_df["date"], "equipment": hold_df["equipment"],
                  "distance": dist_hold, "actual": y_hold, "label_anomaly": hold_anom.astype(int),
                  **{f"pred_{k}": v for k, v in holdout_preds.items()}}).to_csv(OUT_DIR / "holdout_report.csv", index=False)

    # ---------------- refit on all labeled data (Jan-Oct) with the same recipe
    all_df = train[~anomaly_mask(train, BASE_FEATURES)]
    y_all = np.log(all_df["posted_rate"])
    models = {}

    def final_model(name):
        if name not in models:
            # ~11% more data than the holdout fit -> scale the early-stopped iteration count
            models[name] = fit(featurize(all_df)[FEATURE_SETS[name]], y_all, n_estimators=int(best_iters[name] * 1.1))
        return models[name], FEATURE_SETS[name]

    model, cols = final_model(best_val)
    pred = np.exp(model.predict(featurize(valid)[cols]))
    template = pd.read_csv(TEMPLATE_CSV)
    out = template[["load_id"]].merge(pd.DataFrame({"load_id": valid["load_id"], "predicted_rate": pred}),
                                      on="load_id", how="left", validate="one_to_one")
    assert len(out) == 12_000 and out["predicted_rate"].notna().all() and (out["predicted_rate"] > 0).all()
    out["predicted_rate"] = out["predicted_rate"].round(2)
    out.to_csv(ROOT / "validation_predictions.csv", index=False)

    # ---------------- December fixed lane (no market_index / quote_signal given)
    if not DECEMBER_ORIG.exists():
        DECEMBER_ORIG.write_bytes(DECEMBER_CSV.read_bytes())
    dec = pd.read_csv(DECEMBER_ORIG)
    cols_order = list(dec.columns)
    both = pd.concat([train, valid])
    lex = both[both["pickup"] == "Lexington"].iloc[0]
    fw = both[both["delivery"] == "Fort Wayne"].iloc[0]
    dec_feat = pd.DataFrame({
        "date": pd.to_datetime(dec["date"]), "distance": dec["distance"].astype(float),
        "equipment": dec["equipment"], "weight": dec["weight"].astype(float),
        "weight_was_missing": 0, "weight_was_negative": 0,
        "pickup_lat": lex["pickup_lat"], "pickup_lon": lex["pickup_lon"],
        "delivery_lat": fw["delivery_lat"], "delivery_lon": fw["delivery_lon"],
    })
    if best_dec.endswith("+imputed_market"):
        # only labeled-period (Jan-Oct) market history, mirroring the holdout test
        dec_feat = apply_market_profile(dec_feat, market_profile(train))
    model, cols = final_model(best_dec.replace("+imputed_market", ""))
    dec["predicted_rate"] = np.exp(model.predict(featurize(dec_feat)[cols])).round(2)
    assert (dec["predicted_rate"] > 0).all() and list(dec.columns) == cols_order
    dec.to_csv(DECEMBER_CSV, index=False)

    lane_hist = train[(train["pickup"] == "Lexington") & (train["delivery"] == "Fort Wayne")
                      & (train["equipment"] == "Dry Van")]
    with open(OUT_DIR / "metrics.json", "w") as fh:
        json.dump({
            "holdout": "fit 2025-01-01..2025-09-30, evaluate 2025-10-01..2025-10-31",
            "n_fit_after_anomaly_filter": int(len(fit_df)), "n_holdout": int(is_hold.sum()),
            "n_holdout_label_anomalies": int(hold_anom.sum()),
            "n_final_train_after_anomaly_filter": int(len(all_df)),
            "backtest_mean_by_model": bt_mean.round(3).to_dict(orient="index"),
            "october_holdout_all_rows": results, "october_holdout_excluding_label_anomalies": results_clean,
            "best_iterations": best_iters, "selected_validation_model": best_val, "selected_december_model": best_dec,
            "december_min": float(dec["predicted_rate"].min()), "december_max": float(dec["predicted_rate"].max()),
            "lex_fw_dryvan_history_rate_median": float(lane_hist["posted_rate"].median()),
            "lex_fw_dryvan_history_n": int(len(lane_hist)),
            "feature_importance_gain": {k: v.head(10).round(0).to_dict() for k, v in importance.items()},
        }, fh, indent=2)
    print("Wrote validation_predictions.csv, december-chart-inputs.csv, artifacts/")
    print(dec[["date", "predicted_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    error = y_true.to_numpy() - y_pred.to_numpy()
    return float(np.sqrt(np.mean(error**2)))


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.mean(np.abs(y_true.to_numpy() - y_pred.to_numpy())))


def mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    true_values = y_true.to_numpy()
    pred_values = y_pred.to_numpy()
    denominator = np.where(true_values == 0, np.nan, true_values)
    value = np.nanmean(np.abs((true_values - pred_values) / denominator)) * 100.0
    return float(value)


def evaluate_forecast(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    aligned_true, aligned_pred = y_true.align(y_pred, join="inner")
    return {
        "RMSE": rmse(aligned_true, aligned_pred),
        "MAE": mae(aligned_true, aligned_pred),
        "MAPE": mape(aligned_true, aligned_pred),
    }


def build_metrics_table(
    y_true: pd.Series,
    predictions: Mapping[str, pd.Series],
    run_times: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for model_name, model_pred in predictions.items():
        row = {"Model": model_name, **evaluate_forecast(y_true, model_pred)}
        if run_times and model_name in run_times:
            row["Train+Forecast Seconds"] = run_times[model_name]
        rows.append(row)

    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values("RMSE").reset_index(drop=True)
    return table

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from .data import load_monthly_sales
from .dl import recursive_lstm_forecast, train_lstm
from .metrics import build_metrics_table
from .ml import recursive_xgboost_forecast, tune_xgboost_with_tscv
from .statistical import forecast_sarima


def _train_test_split(
    series: pd.Series, test_horizon: int
) -> tuple[pd.Series, pd.Series]:
    if test_horizon <= 0 or test_horizon >= len(series):
        raise ValueError(
            "test_horizon must be positive and smaller than the series length."
        )
    train = series.iloc[:-test_horizon]
    test = series.iloc[-test_horizon:]
    return train, test


def run_benchmark(
    data_path: str | Path | None = None,
    test_horizon: int = 24,
    seasonal_period: int = 12,
    lstm_epochs: int = 250,
) -> dict[str, Any]:
    series = load_monthly_sales(path=data_path)
    train, test = _train_test_split(series, test_horizon=test_horizon)

    predictions: dict[str, pd.Series] = {}
    run_times: dict[str, float] = {}
    metadata: dict[str, dict[str, object]] = {}

    sarima_start = perf_counter()
    sarima_forecast, sarima_meta = forecast_sarima(
        train=train,
        horizon=len(test),
        seasonal_period=seasonal_period,
    )
    run_times["SARIMA (Auto Search)"] = perf_counter() - sarima_start
    predictions["SARIMA (Auto Search)"] = sarima_forecast
    metadata["SARIMA (Auto Search)"] = sarima_meta

    xgb_start = perf_counter()
    xgb_model, xgb_features, xgb_meta = tune_xgboost_with_tscv(train_series=train)
    xgb_forecast = recursive_xgboost_forecast(
        model=xgb_model,
        train_series=train,
        forecast_index=test.index,
        feature_columns=xgb_features,
    )
    run_times["XGBoost (Feature Based)"] = perf_counter() - xgb_start
    predictions["XGBoost (Feature Based)"] = xgb_forecast
    metadata["XGBoost (Feature Based)"] = xgb_meta

    lstm_start = perf_counter()
    lstm_artifacts = train_lstm(train_series=train, max_epochs=lstm_epochs)
    lstm_values = recursive_lstm_forecast(
        artifacts=lstm_artifacts,
        train_series=train,
        forecast_horizon=len(test),
    )
    lstm_forecast = pd.Series(lstm_values, index=test.index, name="lstm")
    run_times["PyTorch LSTM"] = perf_counter() - lstm_start
    predictions["PyTorch LSTM"] = lstm_forecast
    metadata["PyTorch LSTM"] = lstm_artifacts.metadata

    metrics = build_metrics_table(
        y_true=test, predictions=predictions, run_times=run_times
    )

    forecast_frame = pd.concat([test.rename("Actual"), *predictions.values()], axis=1)
    forecast_frame.columns = ["Actual", *predictions.keys()]

    return {
        "series": series,
        "train": train,
        "test": test,
        "predictions": predictions,
        "forecast_frame": forecast_frame,
        "metrics": metrics,
        "metadata": metadata,
    }


def plot_forecasts(result: dict[str, Any]) -> None:
    full_series = result["series"]
    forecast_frame = result["forecast_frame"]

    plt.figure(figsize=(14, 6))
    plt.plot(
        full_series.index,
        full_series.values,
        label="Actual History",
        color="black",
        linewidth=1.8,
    )
    for column in forecast_frame.columns:
        if column == "Actual":
            continue
        plt.plot(
            forecast_frame.index,
            forecast_frame[column].values,
            label=column,
            linewidth=2,
        )

    plt.title("Champagne Sales Forecast: Statistical vs ML vs Deep Learning")
    plt.xlabel("Month")
    plt.ylabel("Sales")
    plt.legend()
    plt.tight_layout()
    plt.show()

from __future__ import annotations

from collections.abc import Sequence
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from .features import build_supervised_frame, make_recursive_feature_row


def tune_xgboost_with_tscv(
    train_series: pd.Series,
    lags: Sequence[int] = (1, 2, 3, 6, 12),
    rolling_windows: Sequence[int] = (3, 6, 12),
    n_splits: int = 4,
) -> tuple[XGBRegressor, list[str], dict[str, float]]:
    frame = build_supervised_frame(
        train_series, lags=lags, rolling_windows=rolling_windows
    )
    if len(frame) < 24:
        raise ValueError("Not enough rows to run robust TimeSeriesSplit for XGBoost.")

    features = [column for column in frame.columns if column != "target"]
    x_data = frame[features]
    y_data = frame["target"]

    n_splits = min(n_splits, max(2, len(frame) // 12))
    splitter = TimeSeriesSplit(n_splits=n_splits)

    search_grid = {
        "n_estimators": [200, 400],
        "max_depth": [2, 3],
        "learning_rate": [0.03, 0.08],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }

    best_params: dict[str, float] | None = None
    best_score = float("inf")

    for values in product(*search_grid.values()):
        params = dict(zip(search_grid.keys(), values))
        fold_scores: list[float] = []

        for train_idx, valid_idx in splitter.split(x_data):
            x_train = x_data.iloc[train_idx]
            y_train = y_data.iloc[train_idx]
            x_valid = x_data.iloc[valid_idx]
            y_valid = y_data.iloc[valid_idx]

            model = XGBRegressor(
                objective="reg:squarederror",
                random_state=42,
                **params,
            )
            model.fit(x_train, y_train)
            valid_pred = model.predict(x_valid)
            fold_rmse = np.sqrt(mean_squared_error(y_valid, valid_pred))
            fold_scores.append(float(fold_rmse))

        avg_score = float(np.mean(fold_scores))
        if avg_score < best_score:
            best_score = avg_score
            best_params = params

    if best_params is None:
        raise RuntimeError("Failed to tune XGBoost parameters.")

    final_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        **best_params,
    )
    final_model.fit(x_data, y_data)

    metadata = {
        "cv_rmse": round(best_score, 3),
        **{key: float(value) for key, value in best_params.items()},
    }
    return final_model, features, metadata


def recursive_xgboost_forecast(
    model: XGBRegressor,
    train_series: pd.Series,
    forecast_index: pd.DatetimeIndex,
    feature_columns: Sequence[str],
    lags: Sequence[int] = (1, 2, 3, 6, 12),
    rolling_windows: Sequence[int] = (3, 6, 12),
) -> pd.Series:
    history_values = [float(value) for value in train_series.tolist()]
    predictions: list[float] = []

    for step_offset, timestamp in enumerate(forecast_index):
        feature_row = make_recursive_feature_row(
            history_values=history_values,
            timestamp=timestamp,
            step_index=len(train_series) + step_offset,
            lags=lags,
            rolling_windows=rolling_windows,
        )
        x_next = pd.DataFrame([feature_row], columns=feature_columns)
        next_value = float(model.predict(x_next)[0])
        predictions.append(next_value)
        history_values.append(next_value)

    return pd.Series(predictions, index=forecast_index, name="xgboost")

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def _seasonality_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    month = index.month
    return pd.DataFrame(
        {
            "month": month,
            "quarter": index.quarter,
            "year": index.year,
            "trend": np.arange(len(index), dtype=float),
            "month_sin": np.sin(2 * np.pi * month / 12.0),
            "month_cos": np.cos(2 * np.pi * month / 12.0),
        },
        index=index,
    )


def build_supervised_frame(
    series: pd.Series,
    lags: Sequence[int] = (1, 2, 3, 6, 12),
    rolling_windows: Sequence[int] = (3, 6, 12),
) -> pd.DataFrame:
    frame = pd.DataFrame(index=series.index)
    frame["target"] = series

    for lag in lags:
        frame[f"lag_{lag}"] = series.shift(lag)

    for window in rolling_windows:
        # Shift by one so rolling stats only use past data.
        shifted = series.shift(1)
        frame[f"roll_mean_{window}"] = shifted.rolling(window=window).mean()
        frame[f"roll_std_{window}"] = shifted.rolling(window=window).std()

    frame = frame.join(_seasonality_features(frame.index))
    frame = frame.dropna()
    return frame


def make_recursive_feature_row(
    history_values: list[float],
    timestamp: pd.Timestamp,
    step_index: int,
    lags: Sequence[int] = (1, 2, 3, 6, 12),
    rolling_windows: Sequence[int] = (3, 6, 12),
) -> dict[str, float]:
    row: dict[str, float] = {}
    for lag in lags:
        row[f"lag_{lag}"] = float(history_values[-lag])

    for window in rolling_windows:
        window_values = np.array(history_values[-window:], dtype=float)
        row[f"roll_mean_{window}"] = float(window_values.mean())
        row[f"roll_std_{window}"] = float(window_values.std(ddof=0))

    month = timestamp.month
    row["month"] = float(month)
    row["quarter"] = float(timestamp.quarter)
    row["year"] = float(timestamp.year)
    row["trend"] = float(step_index)
    row["month_sin"] = float(np.sin(2 * np.pi * month / 12.0))
    row["month_cos"] = float(np.cos(2 * np.pi * month / 12.0))
    return row

from __future__ import annotations

import warnings
from dataclasses import dataclass
from itertools import product
from typing import Iterable

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


@dataclass(frozen=True)
class SarimaConfig:
    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]
    aic: float


def _sarima_candidates(
    seasonal_period: int,
    p_values: Iterable[int],
    d_values: Iterable[int],
    q_values: Iterable[int],
    p_seasonal_values: Iterable[int],
    d_seasonal_values: Iterable[int],
    q_seasonal_values: Iterable[int],
) -> list[tuple[tuple[int, int, int], tuple[int, int, int, int]]]:
    candidates: list[tuple[tuple[int, int, int], tuple[int, int, int, int]]] = []
    for p, d, q, ps, ds, qs in product(
        p_values,
        d_values,
        q_values,
        p_seasonal_values,
        d_seasonal_values,
        q_seasonal_values,
    ):
        order = (p, d, q)
        seasonal_order = (ps, ds, qs, seasonal_period)
        candidates.append((order, seasonal_order))
    return candidates


def fit_best_sarima(
    train: pd.Series,
    seasonal_period: int = 12,
) -> tuple[object, SarimaConfig, int]:
    """Fit the best SARIMA configuration by AIC over a compact grid."""
    candidates = _sarima_candidates(
        seasonal_period=seasonal_period,
        p_values=range(0, 3),
        d_values=(0, 1),
        q_values=range(0, 3),
        p_seasonal_values=range(0, 2),
        d_seasonal_values=(0, 1),
        q_seasonal_values=range(0, 2),
    )

    best_fit = None
    best_config: SarimaConfig | None = None
    tried = 0

    for order, seasonal_order in candidates:
        tried += 1
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = SARIMAX(
                    train,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fitted = model.fit(disp=False)
            aic = float(fitted.aic)
            if best_config is None or aic < best_config.aic:
                best_fit = fitted
                best_config = SarimaConfig(
                    order=order, seasonal_order=seasonal_order, aic=aic
                )
        except Exception:
            continue

    if best_fit is None or best_config is None:
        raise RuntimeError(
            "No SARIMA model converged. Try reducing grid size or differencing."
        )

    return best_fit, best_config, tried


def forecast_sarima(
    train: pd.Series, horizon: int, seasonal_period: int = 12
) -> tuple[pd.Series, dict[str, object]]:
    model_fit, best_config, tried = fit_best_sarima(
        train=train, seasonal_period=seasonal_period
    )
    forecast = model_fit.get_forecast(steps=horizon).predicted_mean
    forecast.name = "sarima_auto"
    metadata = {
        "order": best_config.order,
        "seasonal_order": best_config.seasonal_order,
        "aic": round(best_config.aic, 3),
        "candidates_tried": tried,
    }
    return forecast, metadata

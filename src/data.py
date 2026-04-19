from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "perrin-freres-monthly-champagne.csv"
)


def load_monthly_sales(path: str | Path | None = None) -> pd.Series:
    """Load monthly champagne sales as a clean time-indexed series."""
    source = Path(path) if path else DEFAULT_DATA_PATH
    frame = pd.read_csv(source)

    first_col, second_col = frame.columns[:2]
    frame = frame.rename(columns={first_col: "month", second_col: "sales"})
    frame = frame.dropna(subset=["month", "sales"])

    frame["month"] = pd.to_datetime(frame["month"], format="%Y-%m", errors="coerce")
    frame["sales"] = pd.to_numeric(frame["sales"], errors="coerce")
    frame = frame.dropna(subset=["month", "sales"]).sort_values("month")

    series = frame.set_index("month")["sales"]
    series = series.asfreq("MS")
    if series.isna().any():
        series = series.interpolate(method="linear")

    series.name = "sales"
    return series

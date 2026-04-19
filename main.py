from __future__ import annotations

import argparse

from src.benchmark import plot_forecasts, run_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a multi-model champagne sales forecasting benchmark."
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=24,
        help="Number of months in the hold-out test horizon.",
    )
    parser.add_argument(
        "--lstm-epochs",
        type=int,
        default=250,
        help="Maximum epochs for LSTM training (early stopping is applied).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable forecast comparison plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_benchmark(test_horizon=args.horizon, lstm_epochs=args.lstm_epochs)

    print("\nForecasting Benchmark Results")
    print(result["metrics"].to_string(index=False, float_format=lambda x: f"{x:,.3f}"))

    print("\nModel Notes")
    for model_name, model_meta in result["metadata"].items():
        print(f"- {model_name}: {model_meta}")

    if not args.no_plot:
        plot_forecasts(result)


if __name__ == "__main__":
    main()

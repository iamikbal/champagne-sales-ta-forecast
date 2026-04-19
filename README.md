# Champagne Sales Forecasting Benchmark

This project is structured as a comparative benchmark to answer one practical question:

"How much additional forecasting quality do we gain as we move from classical statistical modeling to feature-based machine learning and then deep learning?"

The benchmark keeps the same dataset split and evaluation metrics across all models so the comparison is fair and not decorative.

## Why Multiple Models (Without Overengineering)

- SARIMA (Auto Search): strong statistical baseline, low feature engineering cost, interpretable seasonal structure.
- XGBoost: tests whether handcrafted lag and calendar features improve accuracy.
- PyTorch LSTM: tests whether sequence learning offers further gains that justify extra training complexity.

Each model has a clear role in the decision story.

## Project Structure

```text
.
|-- data/
|   `-- perrin-freres-monthly-champagne.csv
|-- src/
|   |-- benchmark.py
|   |-- data.py
|   |-- dl.py
|   |-- features.py
|   |-- metrics.py
|   |-- ml.py
|   `-- statistical.py
|-- main.ipynb
|-- main.py
|-- pyproject.toml
`-- README.md
```

## Methods Implemented

1. Auto-SARIMA style grid search (AIC-driven SARIMA tuning).
2. Time-series cross-validation for XGBoost hyperparameter selection.
3. Recursive multi-step forecasting for ML and DL models to avoid leakage.
4. Native PyTorch LSTM with early stopping.
5. Unified metrics: RMSE, MAE, MAPE.

## Evaluation Protocol

- Chronological split with a fixed hold-out horizon (default 24 months).
- Identical horizon used for all models.
- Rolling-origin logic in model design (especially XGBoost CV).
- No random train/test shuffling.

## Quick Start

* Install dependencies:

```bash
pip install -e .
```

* Run benchmark:

```bash
python main.py
```

* Run without plot:

```bash
python main.py --no-plot
```

* Adjust LSTM training budget:

```bash
python main.py --lstm-epochs 400
```
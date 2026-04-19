restructure as modular package with multi-model forecasting benchmark

- Convert from notebook-centric to proper Python package structure
- Implement three forecasting models: auto-SARIMA, XGBoost with time-series CV, PyTorch LSTM
- Add modular source code with clear separation: data loading, feature engineering, metrics, modeling
- Replace scattered notebooks with unified CLI entry point (main.py) and Jupyter interface (main.ipynb)
- Establish Python 3.13 minimum version with pyproject.toml-based dependency management
- Include comprehensive project documentation with benchmark motivation and quick-start guide
- Remove obsolete notebook-based workflows and orphaned PDF reports

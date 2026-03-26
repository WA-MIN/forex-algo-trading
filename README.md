# FX Experiment Platform

A production-ready, modular research platform for foreign exchange (FX) time-series analysis, strategy development, and minute-level backtesting. Designed for academic research and practical experimentation.

## Overview

Supports end-to-end FX workflows:
- Raw minute-level data processing
- Reproducible feature engineering and labeling  
- Deterministic train/validation/test splits
- Global vs. session-conditional modeling (Asia/London/NY/Overlap)
- Cost-aware backtesting baselines (momentum, mean reversion)

Key Design: Immutable data stages. Stable artifacts. Reproducible research.

## Project Structure

```
fx-experiment-platform/
├── data/
│ └── examples/                 # Example of AUDUSD dataset 2020 Csv data and Parquet (10 years) [Tracked]
│ └── extracted/
│ └── parquet/
│ └── processed/
│    └── cleaned/              # Cleaned dataset
|    └── reports/
│
├── eda/
│ ├── raw_snapshot/            # Exploratory analysis
│ └── reports/                 # EDA notebooks/figures
│
├── features/                  # Engineered features
├── labels/                    # Trading signals
│
├── datasets/
│ ├── train/                   # Fixed splits
│ ├── val/
│ └── test/
│ └── reports/                 # Split metadata
│
├── scripts/                   # CLI pipeline - [Tracked]
│ ├── clean_fx_data.py
│ ├── build_fx_features.py
│ ├── build_fx_labels.py
│ ├── split_fx_dataset.py
│ └── run_backtests.py
│
├── strategies/               # Momentum, MA, etc. - [Tracked]
│
├── backtest/                 # Execution engine - [Tracked] 
│
├── experiments/              # Results (local only)
│
├── requirements.txt          [Tracked]
│
└── README.md

```

## Quick Setup

```
git clone https://github.com/[username]/fx-experiment-platform.git
cd fx-experiment-platform

conda env create -f environment.yml
conda activate fx-experiments
```

## Pipeline Usage

1. Clean Raw Data:
   ```
   python scripts/clean_fx_data.py
   ```

2. Feature Engineering:  
   ```
   python scripts/build_fx_features.py --drop-warmup
   ```

3. Build Labels:
   ```
   python scripts/build_fx_labels.py
   ```

4. Dataset Splits:
   ```
   python scripts/split_fx_dataset.py
   ```

5. Backtest:
   ```
   python scripts/run_backtests.py --strategy momentum --regime global
   ```

## Research Design

Core Question: Do session-conditional models outperform global models?

Regime | Sessions | Model Count
---|---|---
Global | All hours | 1 model
Session | Asia, London, NY, Overlap | 4 models

## Features and Labels

Features:
- Lagged returns and volatility
- Momentum/mean-reversion signals
- Session/time encodings

Labels (3-class):
Long (1)     -> Future return > TP threshold
Short (-1)   -> Future return < -SL threshold  
No trade (0) -> Dead zone

## Evaluation Metrics

Metric | Purpose
---|---
Total Return | Overall % profit/loss 
Sharpe Ratio | Risk-adjusted returns
Profit Factor | Gross profit / loss
Max Drawdown | Worst peak-to-trough
Win Rate | % Profitable trades
Volatility | Variability of returns

Cost-aware: Realistic spreads + slippage.

## Reproducibility

- Deterministic splits (fixed seeds)
- Immutable stages (no overwrites)
- CLI reproducibility (no notebooks)
- Artifact versioning (Parquet metadata)

## Example Full Workflow

```
python scripts/clean_fx_data.py &&
python scripts/build_fx_features.py &&
python scripts/build_fx_labels.py &&
python scripts/split_fx_dataset.py &&
python scripts/run_backtests.py --strategy moving_average --regime session
```

## Roadmap

- [x] Data pipeline + splits
- [x] Feature/label framework  
- [ ] Backtesting engine (vectorbt)
- [ ] ML models (XGBoost/LSTM)
- [ ] Walk-forward optimization
- [ ] Experiment tracking (MLflow)

## Notes

- Raw Parquet never modified
- Local only: experiments/, results/ (.gitignore'd)
- Extensible: Add pairs, strategies easily
- Academic: Griffith University forex research

Authors : Harsh Singh Kanyal , Haruka Iwami , Istiak Ahmed , Savindi Hansila Weerakoon
WIL Placement 2026 Trimester 1
License: MIT

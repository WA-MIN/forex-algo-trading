# forex-algo-trading

> A reproducible, cost-aware, modular research platform for evaluating minute-level FX trading strategies under strict time-series validation.

---

## Overview

This repository implements a complete end-to-end pipeline for algorithmic FX research, from raw 1-minute bar ingestion through feature engineering, label construction, walk-forward model evaluation, and cost-aware backtesting. The platform is designed for scientific rigour: every result is reproducible from code and configuration alone, every metric is computed by a single shared engine, and the held-out test set is touched exactly once.

The central research question is:

> Under realistic transaction costs and strict time-series cross-validation, which strategy class -- rule-based, linear ML, or deep learning -- delivers the best risk-adjusted net performance on minute-level FX data?

---

## Key Properties

- **No data leakage.** Chronological splits with purge gaps matching the maximum label horizon. Scalers fit on training folds only. Sequence windows never cross split boundaries.
- **One backtesting engine.** All strategies -- rule-based, logistic regression, and LSTM -- are evaluated by a single shared engine with identical metric definitions. No strategy computes its own Sharpe.
- **Pair-aware transaction costs.** Spread is imposed externally per pair, grounded in published broker references. HistData mid-price bars do not contain executable bid/ask; costs cannot be inferred from the raw data.
- **Sharpe-driven model selection.** Walk-forward fold Net Sharpe is the primary selection criterion. Accuracy and F1 are diagnostic only. A model predicting flat 97% of the time scores 0.0 Sharpe regardless of accuracy.
- **Locked test set.** The 2024-2025 window is never loaded during development. It is evaluated once, after all hyperparameter decisions are finalised on walk-forward folds.
- **Idempotent scripts.** Every pipeline stage checks for existing outputs. Nothing is recomputed unless `--force` is passed.

---

## Repository Structure

```
forex-algo-trading/
|
|-- data/
|   |-- extracted/              # Raw per-year CSVs, not tracked by git
|   |-- parquet/                # Combined per-pair parquets, not tracked
|   `-- processed/
|       |-- cleaned/            # Output of clean_fx_data.py
|       `-- reports/
|
|-- features/
|   |-- pair/                   # Feature-engineered parquets, not tracked
|   `-- reports/
|
|-- labels/
|   |-- pair/                   # Labelled datasets, not tracked
|   `-- reports/
|
|-- datasets/
|   |-- train/                  # Fixed 2015-2021 training split
|   |-- val/                    # Fixed 2022-2023 validation split
|   |-- test/                   # LOCKED -- 2024-2025, loaded once only
|   |-- folds/                  # Walk-forward folds 0 through 4
|   `-- reports/
|
|-- scripts/
|   |-- download_fx_data.py     # Stage 0: download from HistData
|   |-- inspect_fx_data.py      # Stage 1: structural diagnostics
|   |-- eda_fx_data.py          # Stage 2: exploratory analysis
|   |-- clean_fx_data.py        # Stage 3: cleaning and gap handling
|   |-- features_fx_data.py     # Stage 4: feature engineering
|   |-- labels_fx_data.py       # Stage 5: forward-return label construction
|   |-- split_fx_data.py        # Stage 6: chronological splitting and folds
|   |-- backtest_engine.py      # Shared evaluation engine (in progress)
|   |-- strategy_ma_crossover.py
|   |-- strategy_momentum.py
|   |-- model_logistic.py
|   |-- model_lstm.py
|   `-- evaluate_final.py
|
|-- backtests/
|   |-- ma_crossover/
|   |-- momentum/
|   |-- logistic/
|   |-- lstm/
|   `-- reports/
|
|-- configs/                    # YAML experiment configs, tracked
|-- results/                    # Output metrics and plots, not tracked
|-- .gitignore
`-- README.md
```

---

## Data

**Source:** HistData.com 1-minute ASCII OHLC bar data  
**Frequency:** 1-minute bars  
**Price type:** Indicative mid-prices. No true bid/ask, no order book.  
**Volume:** Synthetic tick count, not true traded volume.

### Currency Pairs

| Pair     | Coverage    | Status   |
|----------|-------------|----------|
| EURUSD   | 2015-2025   | Complete |
| GBPUSD   | 2015-2025   | Complete |
| USDJPY   | 2015-2025   | Complete |
| USDCHF   | 2015-2025   | Complete |
| USDCAD   | 2015-2025   | Complete |
| AUDUSD   | 2015-2025   | Complete |
| NZDUSD   | 2015-2025   | Complete |

### Data Schema

| Column          | Type          | Description                                              |
|-----------------|---------------|----------------------------------------------------------|
| `timestamp_utc` | datetime (UTC)| Primary key for all operations. All pipeline logic uses UTC. |
| `open`          | float64       | Bar open (mid, indicative)                               |
| `high`          | float64       | Bar high                                                 |
| `low`           | float64       | Bar low                                                  |
| `close`         | float64       | Bar close                                                |
| `volume`        | float64       | Synthetic tick volume                                    |
| `pair`          | string        | e.g. "EURUSD"                                            |
| `session`       | string        | Asia / London / Overlap / New_York                       |

### Session Definitions (UTC)

| Session  | UTC Hours     | Liquidity Profile           |
|----------|---------------|-----------------------------|
| Asia     | 00:00 - 06:59 | Low liquidity, wide spreads |
| London   | 07:00 - 12:59 | Peak liquidity, tight spreads |
| Overlap  | 13:00 - 16:59 | London/NY overlap, very liquid |
| New_York | 17:00 - 23:59 | Moderate liquidity           |

---

## Dataset Splits

The chronological split is hardcoded in `split_fx_data.py` and is permanently frozen. It must not be changed.

| Split   | Date Range                  | Share  | Role                                         |
|---------|-----------------------------|--------|----------------------------------------------|
| Train   | 2015-01-02 to 2021-12-31    | 66.3%  | Model fitting and rule calibration           |
| Val     | 2022-01-01 to 2023-12-31    | 14.8%  | Hyperparameter selection only                |
| Test    | 2024-01-01 to 2025-12-31    | 18.9%  | Final evaluation -- loaded exactly once      |

**Purge gap:** 15 rows are dropped at each boundary. This matches the maximum label horizon (15 minutes), ensuring no forward-return label computed at the boundary of one split can see into the next split. This is the standard purging convention for financial time-series cross-validation.

Walk-forward folds are located at `datasets/folds/fold_0` through `fold_4`. All model selection and hyperparameter tuning is performed on these folds.

---

## Label Construction

Labels are constructed in `labels_fx_data.py` as forward log-returns with a dead-zone threshold to produce a 3-class signal:

| Parameter              | Value          | Description                          |
|------------------------|----------------|--------------------------------------|
| `horizon_primary`      | 5 minutes      | Primary prediction window            |
| `horizon_secondary`    | 15 minutes     | Secondary prediction window          |
| `threshold_primary`    | 0.0005 (5 bp)  | Dead-zone threshold at h=5           |
| `threshold_secondary`  | 0.0010 (10 bp) | Dead-zone threshold at h=15          |

**3-class label space:**

```
+1  (Long)   forward return > threshold
-1  (Short)  forward return < -threshold
 0  (Flat)   |forward return| <= threshold
```

All models and strategies in this project emit predictions in this identical label space.

---

## Feature Engineering

Features are constructed in `features_fx_data.py` using strictly backward-looking windows. No future information enters any feature.

| Family           | Examples                                              |
|------------------|-------------------------------------------------------|
| Returns          | Lagged log-returns, rolling cumulative returns        |
| Volatility       | Rolling realised volatility, log-range (H-L spread)   |
| Trend            | Moving-average ratios, MA crossover distance          |
| Momentum         | N-bar momentum at multiple horizons                   |
| Range/Structure  | Bar range, close-to-high, close-to-low ratios         |
| Session          | Session label (categorical or one-hot encoded)        |

---

## Transaction Cost Model

HistData bars are mid-price OHLC. There is no bid/ask in the dataset. Spread costs are therefore imposed externally by the backtesting engine on a per-pair basis, grounded in OANDA historical spread references and BrokerChooser broker data.

### Base Spread Table

| Pair    | Spread (pips) | Pip Size | Spread Cost (price units) |
|---------|--------------|----------|---------------------------|
| EURUSD  | 0.8          | 0.0001   | 0.00008                   |
| GBPUSD  | 1.4          | 0.0001   | 0.00014                   |
| USDJPY  | 1.8          | 0.01     | 0.018                     |
| USDCHF  | 1.6          | 0.0001   | 0.00016                   |
| USDCAD  | 2.0          | 0.0001   | 0.00020                   |
| AUDUSD  | 1.1          | 0.0001   | 0.00011                   |
| NZDUSD  | 1.5          | 0.0001   | 0.00015                   |

The full round-trip spread cost is deducted on every trade entry and exit. No additional commission is applied in the baseline configuration. No overnight financing (swap) is applied in the baseline intraday experiments.

An optional second-pass refinement applies session-aware spread multipliers: Asia x1.25, London x1.00, Overlap x0.90, New_York x1.00.

---

## Strategy and Model Stack

Stages are implemented and evaluated in the following fixed order. Each stage depends on the backtesting engine and on the results of the previous stage.

```
backtest_engine.py              Shared evaluation engine -- build first
      |
      |-- strategy_ma_crossover.py     Rule-based baseline A
      |-- strategy_momentum.py         Rule-based baseline B
      |         |
      |         v
      |    baseline_sharpe defined
      |
      |-- model_logistic.py            Linear ML model
      |         |
      |         v
      |    proceed only if > baseline_sharpe
      |
      `-- model_lstm.py                Temporal deep learning model
                |
                v
          evaluate_final.py            Single locked test evaluation
```

**LightGBM and XGBoost are excluded from this version.** The architecture is intentionally scoped to one interpretable linear model and one temporal deep-learning model.

### Rule-Based Baselines

**MA Crossover** (`strategy_ma_crossover.py`)

Fast MA / slow MA crossover on past prices. Grid search over `fast_window` in {5, 10, 15, 20}, `slow_window` in {30, 50, 75, 100}, `ma_type` in {ema, sma}. Tuned on walk-forward folds by Net Sharpe.

**Momentum** (`strategy_momentum.py`)

N-bar lookback return vs signed threshold. Grid search over `lookback` in {5, 10, 20, 30, 60}, `threshold` in {0.0005, 0.0010, 0.0020, 0.0030}. Best configuration defines `baseline_sharpe`.

### Logistic Regression (`model_logistic.py`)

Multiclass logistic regression with `class_weight='balanced'`. `StandardScaler` fit on training fold only. Hyperparameters tuned by mean fold Net Sharpe. Grid: `C` in {0.01, 0.1, 1.0, 10.0}, `solver` in {lbfgs, saga}.

Proceed to LSTM only if logistic mean fold Net Sharpe > `baseline_sharpe`.

### LSTM (`model_lstm.py`)

```
Input:   (batch_size, seq_len, n_features)
          |
         LSTM  hidden_size=64, num_layers=1, dropout=0.2
          |
         take last hidden state
          |
         Linear(64 -> 32) + ReLU + Dropout(0.2)
          |
         Linear(32 -> 3)
          |
         Softmax  ->  {-1, 0, +1}
```

- Optimiser: Adam
- Loss: class-weighted cross-entropy (weights from training fold class frequencies)
- Early stopping: patience = 5
- Gradient clipping: max_norm = 1.0
- Max epochs: 30
- Sequence construction: sliding window of `seq_len` rows; no window crosses a split boundary

Hyperparameter grid: `seq_len` in {30, 60, 120}, `hidden_size` in {32, 64, 128}, `dropout` in {0.1, 0.2, 0.3}, `lr` in {1e-4, 5e-4, 1e-3}, `batch_size` in {64, 128, 256}.

---

## Execution Convention

Signals are generated at bar `t` and executed at bar `t+1`. This prevents the signal from having access to the price at which it will execute.

Position state machine:

| Current State | Incoming Signal | Action                     |
|---------------|-----------------|----------------------------|
| Flat          | +1              | Open long                  |
| Flat          | -1              | Open short                 |
| Long          | -1              | Close long, open short     |
| Short         | +1              | Close short, open long     |
| Any           | 0               | Close position, go flat    |

**TP/SL policy:** In the first evaluation pass, `tp_pips=None`, `sl_pips=None`, `max_holding_bars=15`. TP/SL are available as optional engine parameters and are explored only after confirming signal edge.

---

## Evaluation and Metrics

### Canonical Metric Stack

The following metrics are computed by the backtesting engine for every fold, every pair, and the final test run. No other script computes trading metrics.

| Metric              | Role              | Notes                                            |
|---------------------|-------------------|--------------------------------------------------|
| Net Sharpe Ratio    | Primary selector  | Mean fold Net Sharpe drives all model decisions  |
| Maximum Drawdown    | Risk gate         | Secondary tie-breaker                            |
| Sortino Ratio       | Confirmation      | Penalises downside volatility only               |
| Calmar Ratio        | Confirmation      | Annual return / abs(max drawdown)                |
| Profit Factor       | Confirmation      | Gross profit / gross loss                        |
| Turnover            | Cost proxy        | Entries + reversals per trading day              |
| Win Rate            | Diagnostic only   | Does not drive decisions                         |
| Avg Win / Avg Loss  | Diagnostic only   | Payoff asymmetry check                           |
| Signal Count        | Reliability check | Low signal count = statistically fragile result  |
| Net Return          | Absolute P&L      | Final % return after all costs                   |

Classification metrics (ROC-AUC, accuracy, F1, confusion matrix) are computed for logistic regression and LSTM but are diagnostic only. They do not drive model selection.

### Model Selection Decision Hierarchy

1. Highest mean fold Net Sharpe wins.
2. If Sharpe is close, lower Maximum Drawdown wins.
3. If still close, lower Turnover wins.
4. Profit Factor, Sortino, and Calmar used as confirmation.

---

## Reproducibility

| Control                     | Implementation                                           |
|-----------------------------|----------------------------------------------------------|
| Random seeds                | `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)` |
| Idempotent scripts          | Outputs checked before recomputation; `--force` flag to override |
| Frozen split boundaries     | Hardcoded in `split_fx_data.py`; must not be changed    |
| Scaler fit scope            | Fit on training fold only; applied forward to val/test  |
| Metric source               | Single shared `backtest_engine.py`; no independent metric computation elsewhere |
| Parquet throughout          | Preserves dtypes; no CSV float precision loss           |
| UTC timestamps everywhere   | No timezone ambiguity in any downstream script          |
| Purge at boundaries         | 15 rows dropped at each split edge                      |

---

## Leakage Controls

The following rules are non-negotiable. Any violation invalidates results.

- No scaler, encoder, or normalisation statistic is computed on validation or test data.
- `datasets/test/` is never loaded until `evaluate_final.py` is run.
- No hyperparameter decision references test-set metrics at any point.
- Sequence windows for the LSTM never span split or fold boundaries.
- No feature redefinition or label change after model tuning begins without creating a new versioned experiment.

---

## Pipeline Execution

Run stages in order. Each stage is idempotent.

```bash
# Stage 0: download
python scripts/download_fx_data.py

# Stage 1: inspect
python scripts/inspect_fx_data.py

# Stage 2: EDA (internal use, outputs not tracked)
python scripts/eda_fx_data.py

# Stage 3: clean
python scripts/clean_fx_data.py

# Stage 4: feature engineering
python scripts/features_fx_data.py

# Stage 5: label construction
python scripts/labels_fx_data.py

# Stage 6: chronological splitting
python scripts/split_fx_data.py

# Stage 7: backtest engine + baselines (in progress)
python scripts/strategy_ma_crossover.py
python scripts/strategy_momentum.py

# Stage 8: ML models (gated on baseline_sharpe)
python scripts/model_logistic.py
python scripts/model_lstm.py

# Stage 9: final locked evaluation (run once only)
python scripts/evaluate_final.py
```

---

## Environment

Python 3.10+. Key dependencies:

```
pandas
numpy
pyarrow
scikit-learn
torch
matplotlib
seaborn
pyyaml
```

A full `requirements.txt` or `environment.yml` will be added when the modelling stage begins.

---

## What Is Not in This Project

- No live trading or paper trading integration.
- No high-frequency or tick-level strategies.
- No LightGBM or XGBoost (excluded from this version by design).
- No Transformer or reinforcement learning models.
- No true bid/ask data; spreads are externally modelled, not recovered from the dataset.
- No overnight financing in the baseline experiments.

---

## References

- Liao, S., Chen, J., & Ni, H. (2021). Forex trading volatility prediction using neural network models. arXiv:2112.01166.
- Garcke, J., Gerstner, T., & Griebel, M. (2010). Intraday foreign exchange rate forecasting using sparse grids. INS Preprint 1006.
- Kearney, F., Shang, H. L., & Zhao, Y. (2025). Forecasting intraday foreign exchange volatility with functional GARCH approaches. arXiv:2311.18477.
- Hu, Z., Zhao, Y., & Khushi, M. (2021). A survey of forex and stock price prediction using deep learning. Applied System Innovation, 4(1).
- Pricope, T.-V. (2021). Deep reinforcement learning in quantitative algorithmic trading: A review. arXiv:2106.00123.
- OANDA Historical Spreads Tool. https://www.oanda.com/us-en/trading/historical-spreads/
- BrokerChooser OANDA EURUSD Spread Review. https://brokerchooser.com/broker-reviews/oanda-review/oanda-eurusd-spread
- HistData.com 1-minute ASCII FX bar data. https://www.histdata.com

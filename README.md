# forex-algo-trading

Research project comparing rule-based, linear ML, and deep learning strategies on 1-minute FX data. The single selection criterion is mean fold net Sharpe ratio after transaction costs.

---

## What this is

Seven currency pairs, 2015 to 2025, 1-minute bars from HistData. The pipeline runs from raw download through feature engineering, label construction, walk-forward cross-validation, and cost-aware backtesting. Every strategy gets evaluated by the same engine with the same metric definitions.

The question being answered: under realistic spreads and strict time-series cross-validation, which strategy class produces the best risk-adjusted net performance at the 1-minute level?

---

## Repository structure

```
forex-algo-trading/
|
|-- data/
|   |-- extracted/              # Raw per-year CSVs, not tracked
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
|   |-- test/                   # Locked -- 2024-2025, loaded once only
|   |-- folds/                  # Walk-forward folds 0 through 4
|   `-- reports/
|
|-- scripts/
|   |-- download_fx_data.py     # Stage 0: download from HistData
|   |-- inspect_fx_data.py      # Stage 1: structural diagnostics
|   |-- eda_fx_data.py          # Stage 2: exploratory analysis
|   |-- clean_fx_data.py        # Stage 3: cleaning and gap handling
|   |-- features_fx_data.py     # Stage 4: feature engineering
|   |-- labels_fx_data.py       # Stage 5: label construction
|   `-- split_fx_data.py        # Stage 6: chronological splitting and folds
|
|-- backtest/
|   |-- engine.py               # Shared evaluation engine
|   |-- strategies.py           # MACrossover and MomentumStrategy
|   |-- run_backtest.py         # CLI runner
|   |-- report_generator.py     # HTML report output
|   |-- reports/                # Generated HTML reports
|   `-- templates/
|
|-- .gitignore
`-- README.md
```

---

## Data

**Source:** HistData.com 1-minute ASCII OHLC bars  
**Pairs:** EURUSD, GBPUSD, USDJPY, USDCHF, USDCAD, AUDUSD, NZDUSD  
**Coverage:** 2015-01-01 to 2025-12-31  
**Price type:** Indicative mid-prices. No bid/ask, no order book.  
**Volume:** Synthetic tick count.

| Column          | Type           | Description                          |
|-----------------|----------------|--------------------------------------|
| `timestamp_utc` | datetime (UTC) | Primary key for all operations       |
| `open`          | float64        | Bar open (mid, indicative)           |
| `high`          | float64        | Bar high                             |
| `low`           | float64        | Bar low                              |
| `close`         | float64        | Bar close                            |
| `volume`        | float64        | Synthetic tick volume                |
| `pair`          | string         | e.g. "EURUSD"                        |
| `session`       | string         | Asia / London / Overlap / New_York   |

Session windows (UTC):

| Session  | UTC Hours     |
|----------|---------------|
| Asia     | 00:00 - 06:59 |
| London   | 07:00 - 12:59 |
| Overlap  | 13:00 - 16:59 |
| New_York | 17:00 - 23:59 |

---

## Dataset splits

Hardcoded in `split_fx_data.py`. Do not change.

| Split | Date range               | Role                                      |
|-------|--------------------------|-------------------------------------------|
| Train | 2015-01-02 to 2021-12-31 | Model fitting and rule calibration        |
| Val   | 2022-01-01 to 2023-12-31 | Hyperparameter selection only             |
| Test  | 2024-01-01 to 2025-12-31 | Final evaluation -- loaded exactly once   |

15-row purge gap at each boundary. Matches the 15-minute maximum label horizon so no forward-return label computed at a boundary sees into the next split.

Walk-forward folds are in `datasets/folds/fold_0` through `fold_4`.

---

## Labels

Constructed in `labels_fx_data.py` as forward log-returns with a dead-zone threshold giving a 3-class signal:

| Parameter             | Value          |
|-----------------------|----------------|
| `horizon_primary`     | 5 minutes      |
| `horizon_secondary`   | 15 minutes     |
| `threshold_primary`   | 0.0005 (5 bp)  |
| `threshold_secondary` | 0.0010 (10 bp) |

```
+1   forward return >  threshold   (Long)
-1   forward return < -threshold   (Short)
 0   |forward return| <= threshold  (Flat)
```

---

## Features

Built in `features_fx_data.py` using strictly backward-looking windows.

| Family          | What is computed                                               |
|-----------------|----------------------------------------------------------------|
| Returns         | Lagged log-returns, rolling cumulative returns                 |
| Volatility      | Rolling realised volatility, log-range                         |
| Trend           | Moving-average ratios, MA crossover distance                   |
| Momentum        | N-bar momentum at multiple horizons                            |
| Range/Structure | Bar range, close-to-high and close-to-low ratios               |
| Session         | Session label                                                  |

---

## Backtest engine

`backtest/engine.py` is the only place that computes trading metrics. No strategy or model computes its own Sharpe.

### Key constants

```python
PAIRS        = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD"]
TRADING_DAYS = 252
BARS_PER_DAY = 1440   # 1-minute data
```

Annualisation factor: `sqrt(BARS_PER_DAY * TRADING_DAYS)` = 602.

### Spread table

Per-pair half-spread applied on every entry and exit:

| Pair   | Spread (pips) | Pip size |
|--------|--------------|----------|
| EURUSD | 1.0          | 0.0001   |
| GBPUSD | 1.2          | 0.0001   |
| USDJPY | 1.5          | 0.01     |
| USDCHF | 1.5          | 0.0001   |
| USDCAD | 1.8          | 0.0001   |
| AUDUSD | 1.4          | 0.0001   |
| NZDUSD | 1.8          | 0.0001   |

Spread can be overridden per run with `--spread-override`.

### Execution

Signals are generated at bar `t` and executed at bar `t+1`. The engine calls `signals.shift(1).fillna(0)` before computing returns. Positions are held by forward-fill until a new signal fires.

| Current position | Incoming signal | Action                        |
|------------------|-----------------|-------------------------------|
| Flat             | +1              | Open long                     |
| Flat             | -1              | Open short                    |
| Long             | -1              | Close long, open short        |
| Short            | +1              | Close short, open long        |
| Any              | 0               | Close position, go flat       |

TP and SL are optional parameters (`tp_pips`, `sl_pips`). Default for the first evaluation pass is `tp_pips=None`, `sl_pips=None`.

### Metrics computed

| Metric           | Role              |
|------------------|-------------------|
| Net Sharpe       | Primary selector  |
| Max drawdown     | Risk gate         |
| Sortino          | Confirmation      |
| Calmar           | Confirmation      |
| Profit factor    | Confirmation      |
| Turnover         | Cost proxy        |
| Win rate         | Diagnostic        |
| Avg trade bars   | Diagnostic        |
| N trades         | Reliability check |
| Total return     | Absolute P&L      |

The `BacktestResult` dataclass also carries a 21-bar rolling Sharpe series and signal distribution counts.

### `run_backtest`

```python
run_backtest(
    signals,           # pd.Series of 1 / -1 / 0
    prices,            # pd.Series of close prices, same index
    pair,              # str, must be in PAIRS
    strategy,          # str, name label for output
    split="val",       # "val" or "test"
    spread_pips=None,  # overrides SPREAD_TABLE if set
    tp_pips=None,
    sl_pips=None,
    position_size=1.0,
    fold_index=None,
) -> BacktestResult
```

### `run_cv_folds`

Splits signals and prices into `n_folds` equal contiguous slices and runs `run_backtest` on each. Returns one `BacktestResult` per fold.

---

## Strategies

`backtest/strategies.py` defines two strategies and a registry.

### MACrossover

Dual moving average crossover. Fires a signal only on the bar where the fast MA crosses the slow MA. All other bars return 0. The engine holds the resulting position by forward-fill.

```python
MACrossover(fast=20, slow=50, ma_type="ema")
# ma_type: "ema" or "sma"
# fast must be < slow
```

Signal names in the registry:

- `MACrossover_f20_s50_EMA`
- `MACrossover_f10_s30_EMA`
- `MACrossover_f20_s50_SMA`

### MomentumStrategy

Fires +1 when close breaks above the rolling `lookback`-bar high, -1 when it breaks below the rolling low. Returns 0 on all other bars.

```python
MomentumStrategy(lookback=60)
```

Registry entries: `Momentum_lb60`, `Momentum_lb120`.

### Registry

```python
from backtest.strategies import get_strategy, STRATEGY_REGISTRY

strat = get_strategy("MACrossover_f20_s50_EMA")
signals = strat.generate_signals(prices_df)
```

---

## Runner CLI

`backtest/run_backtest.py` loads cleaned data from `data/processed/cleaned/`, runs the requested strategies, prints a colour-coded results table to the terminal, and writes an HTML report to `backtest/reports/`.

```bash
python -m backtest.run_backtest \
  --pairs EURUSD GBPUSD \
  --strategies MACrossover_f20_s50_EMA Momentum_lb60 \
  --split val \
  --folds 5 \
  --spread-override 1.2 \
  --tp-pips 10 \
  --sl-pips 5 \
  --no-browser
```

All arguments are optional. Defaults: all pairs, all strategies, `val` split, no folds (single full-period run), table spreads, no TP/SL.

The `--split test` flag is available but must not be used until all hyperparameter decisions are finalised.

---

## Pipeline execution

Run in order. Each script is idempotent -- skip if outputs already exist, rerun with `--force`.

```bash
python scripts/download_fx_data.py
python scripts/inspect_fx_data.py
python scripts/eda_fx_data.py
python scripts/clean_fx_data.py
python scripts/features_fx_data.py
python scripts/labels_fx_data.py
python scripts/split_fx_data.py
```

After the pipeline, run strategies via the backtest runner above.

---

## Leakage rules

- No scaler or normalisation statistic is computed on val or test data.
- `datasets/test/` is not loaded until the final evaluation run.
- No hyperparameter decision references test-set metrics.
- LSTM sequence windows never span split or fold boundaries.

---

## Reproducibility

| Control              | How it is set                                                  |
|----------------------|----------------------------------------------------------------|
| Random seeds         | `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)` |
| Split boundaries     | Hardcoded in `split_fx_data.py`                               |
| Scaler fit scope     | Training fold only, applied forward                           |
| Metric computation   | Single shared `engine.py`, nowhere else                       |
| File format          | Parquet throughout, preserves dtypes                          |
| Timestamps           | UTC everywhere                                                 |

---

## Dependencies

Python 3.10+.

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

---

## What is not in this project

- No live or paper trading
- No tick-level or order book data
- No LightGBM or XGBoost
- No Transformer or reinforcement learning models
- No overnight financing in the baseline experiments

---

## References

- Liao, S., Chen, J., & Ni, H. (2021). Forex trading volatility prediction using neural network models. arXiv:2112.01166.
- Hu, Z., Zhao, Y., & Khushi, M. (2021). A survey of forex and stock price prediction using deep learning. Applied System Innovation, 4(1).
- Pricope, T.-V. (2021). Deep reinforcement learning in quantitative algorithmic trading. arXiv:2106.00123.
- OANDA Historical Spreads Tool. https://www.oanda.com/us-en/trading/historical-spreads/
- HistData.com 1-minute ASCII FX bar data. https://www.histdata.com

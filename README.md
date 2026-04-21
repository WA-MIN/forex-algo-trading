# forex-algo-trading

Research pipeline comparing rule-based strategies on 1-minute FX data. The selection criterion is mean fold net Sharpe after transaction costs.

## What this is

Seven currency pairs (plus EURGBP), 2015 to 2025, 1-minute bars sourced from HistData.com. The pipeline runs from raw download through feature engineering, label construction, walk-forward cross-validation, and cost-aware backtesting. Every strategy is evaluated by the same engine with the same metric definitions.

The question: under realistic spreads and strict time-series cross-validation, which strategy produces the best risk-adjusted net performance at the 1-minute level?
## Quick Start
### 1. Clone the repository
```bash
git clone https://github.com/Kanyal-HarsH/forex-algo-trading.git
cd forex-algo-trading
```
### 2. Create virtual environment
```bash
python -m venv venv
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
If ```requirements.txt``` is missing:
```bash
pip install pandas numpy pyarrow scikit-learn matplotlib seaborn pyyaml
```

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
|   |-- train/                  # 2015-2021 training split
|   |-- val/                    # 2022-2023 validation split
|   |-- test/                   # 2024-2025, loaded once only
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
|   |-- strategies.py           # 13 strategies across 5 strategy classes
|   |-- run_backtest.py         # CLI runner
|   |-- report_generator.py     # HTML report output
|   |-- reports/                # Generated HTML reports
|   `-- templates/
|
|-- cli/
|   |-- config.py               # SimConfig dataclass
|   `-- run.py                  # CLI entry point
|
|-- core/
|   `-- data_loader.py          # Data loading and filtering logic
|
|-- .gitignore
`-- README.md
```

## Configuration
### Option 1: CLI arguments 
All parameters can be passed via CLI.
### Option 2: YAML Config 
Create `configs/default.yaml`:
```yaml
pairs: ["EURUSD", "GBPUSD"]
strategies: ["MACrossover_f20_s50_EMA"]
split: "val"
folds: 5
spread: 1.2
tp_pips: 10
sl_pips: 5
capital: 10000
```

Run using:

```bash
python -m cli.run --config configs/default.yaml
```
### Option 3: Environment Variables (```.env```)
```bash
DATA_DIR=./data
DEFAULT_CAPITAL=10000
DEFAULT_SPREAD=1.0
```
Used inside `config.py`.

## Data

**Source:** HistData.com 1-minute ASCII OHLC bars. 

**Pairs:** EURUSD, GBPUSD, USDJPY, USDCHF, USDCAD, AUDUSD, NZDUSD, EURGBP. 

**Coverage:** 2015-01-01 to 2025-12-31.

**Price type:** Indicative mid-prices. No bid/ask, no order book.

**Volume:** Synthetic tick count.

| Column          | Type           | Description                        |
|-----------------|----------------|------------------------------------|
| `timestamp_utc` | datetime (UTC) | Primary key for all operations     |
| `open`          | float64        | Bar open (mid, indicative)         |
| `high`          | float64        | Bar high                           |
| `low`           | float64        | Bar low                            |
| `close`         | float64        | Bar close                          |
| `volume`        | float64        | Synthetic tick volume              |
| `pair`          | string         | e.g. "EURUSD"                      |
| `session`       | string         | Asia / London / Overlap / New_York |

Session windows (UTC):

| Session  | UTC Hours     |
|----------|---------------|
| Asia     | 00:00 - 06:59 |
| London   | 07:00 - 12:59 |
| Overlap  | 13:00 - 16:59 |
| New_York | 17:00 - 23:59 |

## Dataset splits

Hardcoded in `split_fx_data.py`.

| Split | Date range               | Role                                    |
|-------|--------------------------|-----------------------------------------|
| Train | 2015-01-02 to 2021-12-31 | Model fitting and rule calibration      |
| Val   | 2022-01-01 to 2023-12-31 | Hyperparameter selection only           |
| Test  | 2024-01-01 to 2025-12-31 | Final evaluation -- loaded exactly once |

15-row purge gap at each boundary. Prevents forward-return labels computed at a boundary from seeing into the next split.

Walk-forward folds are in `datasets/folds/fold_0` through `fold_4`.

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

## Features

Built in `features_fx_data.py` using strictly backward-looking windows.

| Family          | What is computed                                             |
|-----------------|--------------------------------------------------------------|
| Returns         | Lagged log-returns, rolling cumulative returns               |
| Volatility      | Rolling realised volatility, log-range                       |
| Trend           | Moving-average ratios, MA crossover distance                 |
| Momentum        | N-bar momentum at multiple horizons                          |
| Range/Structure | Bar range, close-to-high and close-to-low ratios             |
| Session         | Session label                                                |

## Backtest engine

`backtest/engine.py` is the only place that computes trading metrics.

### Key constants

```python
PAIRS        = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD", "EURGBP"]
TRADING_DAYS = 252
BARS_PER_DAY = 390    # 1-minute data, trading hours only
```

Annualisation factor: `sqrt(BARS_PER_DAY * TRADING_DAYS)` = 313 (approx).

### Spread table

Per-pair half-spread applied on entry and exit:

| Pair   | Spread (pips) | Pip size |
|--------|---------------|----------|
| EURUSD | 0.6           | 0.0001   |
| GBPUSD | 0.8           | 0.0001   |
| USDJPY | 0.7           | 0.01     |
| USDCHF | 1.0           | 0.0001   |
| AUDUSD | 0.8           | 0.0001   |
| USDCAD | 1.0           | 0.0001   |
| NZDUSD | 1.2           | 0.0001   |
| EURGBP | 1.0           | 0.0001   |

Override per run with `--spread`.

### Execution model

Signals are generated at bar `t` and executed at bar `t+1`. The engine calls `signals.shift(1).fillna(0)` before computing returns. Positions are held by forward-fill until a new signal fires or a forced exit condition is met.

| Current position | Incoming signal | Action                       |
|------------------|-----------------|------------------------------|
| Flat             | +1              | Open long                    |
| Flat             | -1              | Open short                   |
| Long             | -1              | Close long, open short       |
| Short            | +1              | Close short, open long       |
| Any              | 0               | Close position, go flat      |

TP and SL are optional (`tp_pips`, `sl_pips`). Default is no TP or SL.

### Core Metrics

| Metric          | Role              |
|-----------------|-------------------|
| Net Sharpe      | Primary selector  |
| Max drawdown    | Risk gate         |
| Sortino         | Confirmation      |
| Calmar          | Confirmation      |
| Profit factor   | Confirmation      |
| Turnover        | Cost proxy        |
| Win rate        | Diagnostic        |
| Avg trade bars  | Diagnostic        |
| N trades        | Reliability check |
| Total return    | Absolute P&L      |

`BacktestResult` also carries a 390-bar rolling Sharpe series and signal distribution counts.

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
    capital_initial=10_000.0,
    fold_index=0,
) -> BacktestResult
```

### `run_wf_folds`

Divides the train split into `n_folds` equal contiguous windows and runs `run_backtest` on each. Returns one `BacktestResult` per fold.

## Strategies

`backtest/strategies.py` defines 13 strategies across 5 classes.

### MACrossover

Dual moving average crossover. Fires a signal only on the bar where the fast MA crosses the slow MA.

```python
MACrossover(fast=20, slow=50, ma_type="ema")  # ma_type: "ema" or "sma"
```

Registry entries: `MACrossover_f20_s50_EMA`, `MACrossover_f10_s30_EMA`, `MACrossover_f20_s50_SMA`

### MomentumStrategy

Fires +1 when close breaks above the rolling high, -1 when below the rolling low.

```python
MomentumStrategy(lookback=60)
```

Registry entries: `Momentum_lb60`, `Momentum_lb120`

### DonchianBreakout

Donchian channel breakout. Long on new N-bar high, short on new N-bar low.

```python
DonchianBreakout(period=20)
```

Registry entries: `Donchian_p20`, `Donchian_p55`

### RSIMeanReversion

Fires long when RSI crosses up through the oversold level, short when it crosses down through overbought. Uses Wilder smoothing.

```python
RSIMeanReversion(period=14, oversold=30, overbought=70)
```

Registry entries: `RSI_p14_os30_ob70`, `RSI_p7_os25_ob75`

### BollingerBreakout

Fires long when close crosses above the upper Bollinger Band, short when below the lower band.

```python
BollingerBreakout(period=20, std_dev=2.0)
```

Registry entries: `BB_p20_std2_0`, `BB_p14_std1_5`

### MACDSignalCross

Fires long when the MACD line crosses above the signal line, short when it crosses below.

```python
MACDSignalCross(fast=12, slow=26, signal_period=9)
```

Registry entries: `MACD_f12_s26_sig9`, `MACD_f8_s21_sig5`

### Using the registry

```python
from backtest.strategies import get_strategy, STRATEGY_REGISTRY

strat = get_strategy("MACrossover_f20_s50_EMA")
signals = strat.generate_signals(prices_df)
```

## Runner CLI

`backtest/run_backtest.py` loads data from `datasets/`, runs the requested strategies, prints a results table to the terminal, and writes an HTML report to `backtest/reports/`.

```bash
python -m backtest.run_backtest \
  --pair EURUSD GBPUSD \
  --strategy MACrossover_f20_s50_EMA Momentum_lb60 \
  --split val \
  --folds 5 \
  --spread 1.2 \
  --tp-pips 10 \
  --sl-pips 5 \
  --no-browser
```

All arguments are optional. Defaults: all pairs, all strategies, `full` split, no folds, table spreads, no TP/SL.

Do not use `--split test` until all hyperparameter decisions are final.

## CLI Arguments

| Flag            | Description                      |
|-----------------|----------------------------------|
| `--pair`        | Currency pairs (space separated) |
| `--strategy`    | Strategy names from registry     |
| `--split`       | `train`, `val`, `test` or `full` |
| `--folds`       | Walk-forward folds               |
| `--spread`      | Override spread                  |
| `--tp-pips`     | Take profit                      |
| `--sl-pips`     | Stop loss                        |
| `--no-browser`  | Disable auto report opening      |

## Pipeline execution

Run in order. Each script is idempotent.

```bash
python scripts/download_fx_data.py
python scripts/inspect_fx_data.py
python scripts/eda_fx_data.py
python scripts/clean_fx_data.py
python scripts/features_fx_data.py
python scripts/labels_fx_data.py
python scripts/split_fx_data.py
```

Then run strategies via the backtest runner above.

## Leakage rules

- No scaler or normalisation statistic is computed on val or test data.
- `datasets/test/` is not loaded until the final evaluation run.
- No hyperparameter decision references test-set metrics.

## Reproducibility

| Control            | How it is set                                              |
|--------------------|------------------------------------------------------------|
| Random seeds       | `random.seed(42)`, `np.random.seed(42)`                   |
| Split boundaries   | Hardcoded in `split_fx_data.py`                           |
| Scaler fit scope   | Training fold only, applied forward                       |
| Metric computation | Single shared `engine.py`, nowhere else                   |
| File format        | Parquet throughout, preserves dtypes                      |
| Timestamps         | UTC everywhere                                            |

## Dependencies

Python 3.10+.

```
pandas
numpy
pyarrow
scikit-learn
matplotlib
seaborn
pyyaml
```

## What is not in this project

- No live or paper trading
- No tick-level or order book data
- No ML or deep learning models
- No LightGBM, XGBoost, or Transformer architectures
- No overnight financing in the baseline experiments

## References

- Liao, S., Chen, J., & Ni, H. (2021). Forex trading volatility prediction using neural network models. arXiv:2112.01166.
- Hu, Z., Zhao, Y., & Khushi, M. (2021). A survey of forex and stock price prediction using deep learning. Applied System Innovation, 4(1).
- Pricope, T.-V. (2021). Deep reinforcement learning in quantitative algorithmic trading. arXiv:2106.00123.
- OANDA Historical Spreads Tool. https://www.oanda.com/us-en/trading/historical-spreads/
- HistData.com 1-minute ASCII FX bar data. https://www.histdata.com

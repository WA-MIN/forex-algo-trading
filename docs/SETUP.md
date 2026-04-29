# Setup and usage

This document covers everything needed to install forex-algo-trading, configure it, and run a first evaluation. The audience is someone who has cloned the repository and now needs to get to a working master evaluation. A short quick-start lives in the README; this document is the long form.

## Prerequisites

| Requirement | Minimum version | Check command | Expected output |
|-------------|-----------------|---------------|-----------------|
| Python | 3.11 | `python --version` | `Python 3.11.x` or higher (3.13 is tested) |
| Git | 2.x | `git --version` | `git version 2.x.x` |
| pip | recent | `pip --version` | `pip XX.Y from ...` |
| Free disk space | 50 GB | `df -h .` (Unix) or `dir` (Windows) | shows free space on the partition |

The disk requirement is dominated by the gitignored data directories (`data/`, `features/`, `labels/`, `datasets/`) which together total roughly 46 GB once the seven-stage pipeline has populated them. If only the trained models and outputs are needed, the working set is much smaller, but a full evaluation run requires the full pipeline outputs.

## Clone the repository

```bash
git clone https://github.com/Kanyal-HarsH/forex-algo-trading.git
cd forex-algo-trading
```

Expected output:

```
Cloning into 'forex-algo-trading'...
remote: Enumerating objects: ...
Receiving objects: 100% ...
Resolving deltas: 100% ...
```

## Environment setup

Two methods are documented. The platform was developed against `venv`; `conda` is supported for users who prefer it. A Docker setup is not provided.

### Method 1: virtual environment (recommended)

```bash
python -m venv venv

# macOS or Linux
source venv/bin/activate

# Windows
venv\Scripts\activate

# Verify activation
which python      # macOS or Linux
where python       # Windows
```

The activation step should print a path inside the project's `venv/` directory. If it prints a system Python path, the environment did not activate.

### Method 2: conda

```bash
conda create -n fxalgo python=3.11
conda activate fxalgo
```

After activating either environment, the Python version check should report 3.11 or higher.

## Install dependencies

```bash
pip install -r requirements.txt
```

Expected packages installed (ten production dependencies):

```
pandas
numpy
pyarrow
scikit-learn
matplotlib
seaborn
pyyaml
python-dotenv
torch
scipy
```

Torch is the largest install (~2.5 GB on first install). The remaining nine packages together are well under 200 MB.

The optional `playwright` dependency is needed only by `scripts/export_report_pdf.py`, which converts HTML backtest reports to PDF. To enable it:

```bash
pip install playwright
python -m playwright install chromium
```

Skip these commands if PDF export is not needed.

## Configuration

All runtime constants are defined in `config/constants.py`. Most have environment-variable overrides (loaded via `python-dotenv`) so that behaviour can be changed without editing source. The full list of overrides is documented in `.env.example` at the repository root. Copy it to a real `.env` to customise:

```bash
cp .env.example .env
```

The most relevant overrides:

| Variable | Type | Default | Effect |
|----------|------|---------|--------|
| `TRADING_DAYS_PER_YEAR` | int | 252 | Annualisation factor numerator |
| `BARS_PER_TRADING_DAY` | int | 390 | Annualisation factor denominator. Use 1440 for FX 24-hour annualisation. |
| `MIN_BARS_PER_DAY` | int | 1200 | Minimum 1-minute bars required to keep a calendar day during cleaning (~83% of a 1440-bar day) |
| `PURGE_ROWS` | int | 15 | Tail rows purged from training windows to prevent label leakage. Must be ≥ HORIZON_SECONDARY. |
| `HORIZON_PRIMARY` | int | 5 | Forward-return horizon for the primary label |
| `HORIZON_SECONDARY` | int | 15 | Secondary forward-return horizon |
| `VOL_REGIME_WINDOW` | int | 30 | Rolling window (bars) for the realised-vol regime feature |
| `VOL_HIGH_REGIME_PERCENTILE` | float | 0.80 | Quantile threshold above which a bar counts as high volatility |
| `RET_WINDOWS` | comma-int | `1,5,15` | Return horizons used in feature computation |
| `VOL_WINDOWS` | comma-int | `10,30,60` | Realised-vol windows |
| `MA_WINDOWS` | comma-int | `10,30,60,120` | Simple-moving-average windows |
| `MOM_WINDOWS` | comma-int | `5,15,30` | Momentum windows |
| `RANGE_MA_WINDOWS` | comma-int | `10,30` | Range moving-average windows |
| `ROLLING_SHARPE_WINDOW` | int | 390 | Window for the rolling Sharpe display in HTML reports |
| `DEFAULT_N_FOLDS` | int | 5 | Walk-forward fold count |
| `LOG_LEVEL` | str | `INFO` | Root log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `DEFAULT_CAPITAL` | float | 10000.0 | Starting capital for backtests |
| `TP_SL_GRID` | tuples | `10,5;15,7;20,10;30,15;50,20;100,40` | TP/SL grid used by T3 of the rule-based path |

The locked split dates (`TRAIN_END=2021-12-31`, `VAL_START=2022-01-01`, `VAL_END=2023-12-31`, `TEST_START=2024-01-01`, `TEST_END=2025-12-31`) are not env-overridable. Changing them invalidates the comparability the platform is built around.

## Verify installation

A quick sanity check that the environment is wired correctly:

```bash
python -m pytest tests/ -q
```

Expected output:

```
............s.........
22 passed, 1 skipped in 1.5s
```

The exact passed count varies as tests are added. The full suite contains around 63 tests across twelve files. Any failures here mean the environment is not configured correctly; do not proceed until tests pass.

A second check that the master evaluation CLI is reachable:

```bash
python scripts/master_eval.py --help
```

Expected output is the master evaluation help banner with the available flags.

## Bootstrap from scratch

If the repository was just cloned, the data directories (`data/`, `features/`, `labels/`, `datasets/`, `scalers/`) are empty (they are gitignored). Bootstrapping from scratch runs the full pipeline. This is a multi-hour operation.

```bash
# Stage 1: download yearly CSVs from histdata.com
python scripts/download_fx_data.py

# Stage 2: clean and write per-pair Parquets
python scripts/clean_fx_data.py

# Stage 3: compute features (six families, ~70 columns per pair)
python scripts/features_fx_data.py

# Stage 4: compute three-class labels per pair
python scripts/labels_fx_data.py

# Stage 5: write train/val/test splits, walk-forward folds, and per-pair scalers
python scripts/split_fx_data.py
```

Expected total runtime on a recent laptop: roughly 90 minutes to 3 hours, dominated by stage 3 (feature computation across 1.5M bars per pair times 7 pairs).

After bootstrap, every subsequent run reads from the on-disk Parquets directly, and the pipeline does not need to be re-run unless features or labels are modified.

## Train missing model cells

The repository ships pre-trained model checkpoints in `models/global/` and `models/session/`. As of this writing, all 28 LR cells are trained and 18 of 28 LSTM cells are trained.

To check which cells are present:

```bash
ls models/global/                 # global-condition checkpoints
ls models/session/london/         # london-condition checkpoints
ls models/session/ny/             # ny-condition checkpoints
ls models/session/asia/           # asia-condition checkpoints
```

To train a single missing cell:

```bash
python scripts/train_model.py --pair NZDUSD --model-type lstm --session global
```

To train every cell in the LR x LSTM grid:

```bash
python scripts/train_all.py
```

Per-cell training runtime: roughly 40 to 90 minutes for an LSTM (smaller for session-conditional cells, larger for global), under five minutes for an LR cell.

---

# Usage guide

## Basic usage

Run a single rule-based backtest on a one-day window:

```bash
python backtest/run_backtest.py \
  --pair EURUSD \
  --strategy RSI_p14_os30_ob70 \
  --split test \
  --from 2024-01-02 --to 2024-01-02 \
  --capital 10000 --spread 0.6 \
  --no-browser
```

Expected output: a per-trade summary printed to stdout, plus an HTML report written to `backtest/reports/`.

Run a single LR-based backtest:

```bash
python backtest/run_backtest.py \
  --pair EURUSD --strategy LR_global \
  --split test --from 2024-01-02 --to 2024-01-02 \
  --capital 10000 --spread 0.6 --no-browser
```

Run the master evaluation on the 2024 calendar year only:

```bash
python scripts/master_eval.py --eval-year 2024 --spreads 1.0
```

Expected runtime for a single-year, single-spread evaluation across all seven pairs: roughly 35 to 50 minutes.

## CLI commands

### `scripts/master_eval.py`

| Flag | Description | Default | Example |
|------|-------------|---------|---------|
| `--pairs` | Restrict the evaluation to specific pairs | all 7 | `--pairs EURUSD GBPUSD` |
| `--eval-year` | Evaluate on a specific calendar year (within the test split) | full test split | `--eval-year 2024` |
| `--from` / `--to` | Custom date window (within the test split) | full test split | `--from 2024-01-01 --to 2024-06-30` |
| `--spreads` | One or more spread multipliers | `1.0` | `--spreads 1.0 1.5 2.0` |
| `--ml-only` | Skip the rule-based path (T1 to T5) | `false` | `--ml-only` |
| `--rule-based-only` | Skip the ML cross-session path | `false` | `--rule-based-only` |
| `--output-dir` | Destination directory for CSVs and the text report | `output/master_eval` | `--output-dir output/run_2024` |

### `backtest/run_backtest.py`

| Flag | Description | Default | Example |
|------|-------------|---------|---------|
| `--pair` | Currency pair (one) | required | `--pair EURUSD` |
| `--strategy` | One or more strategies (rule-based name or `LR_*` / `LSTM_*`) | required | `--strategy RSI_p14_os30_ob70 LR_global LSTM_global` |
| `--split` | Which split to use | `val` | `--split test` |
| `--from` / `--to` | Date range within the split | full split | `--from 2024-01-01 --to 2024-12-31` |
| `--capital` | Starting capital | `10000` | `--capital 50000` |
| `--spread` | Spread multiplier | `1.0` | `--spread 1.0` |
| `--tp-pips` / `--sl-pips` | Take-profit and stop-loss in pips | none | `--tp-pips 20 --sl-pips 10` |
| `--no-browser` | Suppress automatic HTML report opening | `false` | `--no-browser` |

## Step-by-step walkthrough

The canonical end-to-end workflow, in order:

1. **Verify the environment.** Run `python -m pytest tests/ -q` and confirm all tests pass.
2. **Bootstrap the data pipeline** (only if the data directories are empty). Run stages 1 to 5 in order.
3. **Train any missing model cells.** Check `models/global/` and `models/session/` against the seven pairs and four sessions. Run `python scripts/train_model.py` for each missing cell.
4. **Run the master evaluation.** Start with `--eval-year 2024 --spreads 1.0` for the fastest meaningful run. Expect 35 to 50 minutes.
5. **Read the master report.** Open `output/master_eval/master_report.txt`. The headline ranking, the DM test results, and the per-pair transfer matrices are summarised in plain text.
6. **Inspect the structured outputs.** The seven CSVs in `output/master_eval/` (results_all, results_ml, results_rule_based, best_worst_per_pair, dm_test_results, session_generalisability, plus per-pair transfer matrices) are the canonical structured form for downstream analysis.
7. **Run head-to-head backtests for specific cells of interest.** For any (pair, strategy) cell that warrants closer inspection, run `backtest/run_backtest.py` with the `--no-browser` flag for batch use, or without it for an interactive HTML report.

## Advanced usage

<details>
<summary>Re-fitting scalers</summary>

Scalers are fit on the training split only and saved to `scalers/{PAIR}_scaler.pkl`. To re-fit a scaler after changing the feature schema:

```bash
python scripts/split_fx_data.py --pair EURUSD --refit-scaler-only
```

The scaler file contains a dict with two keys: `scaler` (the fitted `StandardScaler`) and `feature_cols` (the column-order list). Both keys are required by every downstream consumer.

</details>

<details>
<summary>Single-pair smoke test</summary>

Run the master evaluation against a single pair on a one-year window for a fast sanity check (~8 to 12 minutes):

```bash
python scripts/master_eval.py --pairs EURUSD --eval-year 2024 --spreads 1.0
```

Useful when iterating on a strategy implementation or after modifying the engine.

</details>

<details>
<summary>Multi-strategy head-to-head</summary>

Compare three strategies on the same window in a single backtest invocation:

```bash
python backtest/run_backtest.py \
  --pair EURUSD \
  --strategy RSI_p14_os30_ob70 LR_global LSTM_global \
  --split test --from 2024-01-01 --to 2024-12-31 \
  --capital 10000 --spread 0.6 --no-browser
```

The HTML report renders all three equity curves on the same axes for visual comparison.

</details>

<details>
<summary>Exporting an HTML report to PDF</summary>

Requires the optional `playwright` dependency.

```bash
python scripts/export_report_pdf.py \
  --input backtest/reports/report_EURUSD_RSI_p14_os30_ob70_*.html \
  --output backtest/reports/report.pdf
```

The exporter renders the HTML through a headless Chromium and prints to a single-page PDF.

</details>

<details>
<summary>Custom evaluation windows</summary>

The default evaluation window is the full locked test split (2024-01-01 to 2025-12-31). To evaluate only a specific sub-window:

```bash
python scripts/master_eval.py --from 2024-01-01 --to 2024-06-30 --spreads 1.0
```

Custom windows must lie inside the locked test split. The script validates the bounds and refuses windows outside it.

</details>

<details>
<summary>Re-running a single fold for debugging</summary>

To re-run a specific walk-forward fold without re-running the full T4 stability check:

```bash
python scripts/master_eval.py --pairs EURUSD --rule-based-only --debug-fold 2
```

Useful when investigating fold-level instability for a specific (pair, strategy) cell.

</details>

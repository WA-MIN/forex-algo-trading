# Experimentation setup

This document describes how forex-algo-trading is set up as a research apparatus. The seven-stage pipeline is the experimental protocol. The locked splits, frozen feature lists, and deterministic master evaluation script are the reproducibility controls. The three research questions (RQ0, RQ1, RQ2) are the experiments themselves.

## Experimental framework

The experimental protocol is encoded as a sequence of seven scripts in the `scripts/` directory, executed in order: `download_fx_data.py` → `clean_fx_data.py` → `features_fx_data.py` → `labels_fx_data.py` → `split_fx_data.py` → `train_model.py` → `master_eval.py`. Each stage writes to a fixed location with a fixed schema, and each stage is its own script so that a failure in one stage can be repaired and re-run without invalidating the rest of the pipeline state.

Reproducibility is treated as a research question in its own right rather than an engineering concern. The platform makes a number of design choices specifically to honour reproducibility, sometimes at the cost of speed or convenience. The locked split dates are constants in `config/constants.py` rather than CLI flags. The LR feature list and both LSTM feature lists are frozen with explicit `do not modify` comments. Every model checkpoint is saved with the random seed and the input feature column list it was trained on. The scaler contract pairs each fitted `StandardScaler` with the column-order list it expects at inference. The pytest suite (twelve files, around 63 tests) covers the kinds of regressions that erode reproducibility silently: wrong index alignment between features and labels, scaler fit on the wrong split, session masks that include or exclude the boundary minute inconsistently, fold-path resolution that drifts between row-index slicing and on-disk Parquet reads.

The master evaluation script is deterministic given a fixed configuration. Running `scripts/master_eval.py` with the same `--pairs`, `--eval-year`, `--spreads`, and on-disk model checkpoints produces diff-equal output CSVs and text reports across runs. This determinism is the operational form of RQ0 and is implicitly tested by every per-stage regression test in the suite.

Output artefacts land in `output/master_eval/` and consist of one text report (`master_report.txt`) plus seven structured CSVs. `results_all.csv` combines rule-based and ML results sorted by composite score. `results_rule_based.csv` and `results_ml.csv` contain the per-class breakdowns. `best_worst_per_pair.csv` summarises per-pair extremes. `dm_test_results.csv` reports the four Diebold-Mariano comparisons per pair. `session_generalisability.csv` summarises the in-domain-versus-transfer pattern across pairs. Per-pair transfer matrices (4 by 4 Sharpe matrices for LR and LSTM) live in `transfer_matrix_lr_{PAIR}.csv` and `transfer_matrix_lstm_{PAIR}.csv`.

## Experiment catalogue

| Experiment ID | Hypothesis | Variables | Controlled | Status |
|---------------|------------|-----------|------------|--------|
| RQ0 | Identical inputs (seeds, splits, code, environment) produce identical evaluation outputs | none (precondition) | seeds, splits, code, environment | ongoing precondition; verified by per-stage regression tests |
| RQ1 | Session-conditional models achieve higher cost-adjusted Sharpe than a single global model | session conditioning (4 levels: global, london, ny, asia) | data, splits, costs, scoring, model class | data ready; awaiting full LSTM grid (10 cells outstanding) |
| RQ2 | A multi-scale LSTM with two branches outperforms Logistic Regression under identical evaluation conditions | model class (LR vs LSTM) | data, splits, costs, scoring, session conditioning | data ready; awaiting full LSTM grid |

The status column reflects the gating constraint for each experiment. RQ0 is ongoing in the sense that every change to the codebase is implicitly an RQ0 test: if the change introduces non-determinism, the regression suite catches it. RQ1 and RQ2 are gated on completing the LSTM grid, since both require comparing 28 cells against 28 cells (seven pairs by four sessions for each model class).

The full LSTM grid is currently 18 of 28 cells trained. NZDUSD is missing all four conditions (global, london, ny, asia). USDCAD is missing all four conditions. AUDUSD is missing the ny and asia conditions. Training the remaining ten cells is the immediate prerequisite for the final research run.

## Running experiments

The simplest meaningful experiment is a single-year master evaluation across all pairs:

```bash
python scripts/master_eval.py --eval-year 2024 --spreads 1.0
```

Expected runtime: 35 to 50 minutes on a recent laptop. Output structure:

```
output/master_eval/
├── master_report.txt              # definitive text report
├── results_all.csv                # all backtest rows, sorted by composite
├── results_rule_based.csv         # T1-T5 breakdown
├── results_ml.csv                 # ML cross-session breakdown
├── best_worst_per_pair.csv        # per-pair extremes
├── transfer_matrix_lr_EURUSD.csv  # one per pair
├── transfer_matrix_lstm_EURUSD.csv
├── session_generalisability.csv
├── dm_test_results.csv
└── cost_breakeven.csv
```

The full-window evaluation (the entire test split, 2024 to 2025, all pairs, all spreads) takes longer and is appropriate for the final research run:

```bash
python scripts/master_eval.py --spreads 1.0 1.5 2.0
```

Expected runtime: 90 to 150 minutes.

For a single-pair smoke test useful when iterating on a strategy implementation:

```bash
python scripts/master_eval.py --pairs EURUSD --eval-year 2024 --spreads 1.0
```

Expected runtime: 8 to 12 minutes.

For an ML-only run that skips the tiered rule-based path:

```bash
python scripts/master_eval.py --ml-only --eval-year 2024 --spreads 1.0
```

Expected runtime: 10 to 15 minutes.

## Reproducibility checklist

Before committing or publishing a result, confirm each item below.

- [x] Random seed fixed in `scripts/train_model.py` (default 42; same seed used for both LR and LSTM training)
- [x] Locked split constants frozen in `config/constants.py` (`TRAIN_END=2021-12-31`, `VAL_START=2022-01-01`, `VAL_END=2023-12-31`, `TEST_START=2024-01-01`, `TEST_END=2025-12-31`)
- [x] Frozen `LR_FEATURES` list (18 items, `do not modify` comment)
- [x] Frozen `LSTM_SHORT_FEATURES` list (5 items)
- [x] Frozen `LSTM_LONG_FEATURES` list (4 items, optionally extended with `same_minute_prev_day_logrange` when present in training data)
- [x] Per-pair scaler saved as a dict with `scaler` and `feature_cols` keys
- [x] Per-pair flat spread fixed in `PAIR_SPREAD_PIPS` constant
- [x] Pytest suite runs on every change (12 files, around 63 tests)
- [x] Master evaluation script is deterministic given fixed inputs
- [x] Per-stage regression tests cover the scaler contract, fold path resolution, session mask boundaries, and label-feature alignment
- [ ] Pinned dependency versions in `requirements.txt` (currently unpinned; flagged as a project TODO)
- [ ] Lockfile (`pip-tools` compile or equivalent) (not yet adopted)
- [x] Environment overrides documented in `.env.example` at the repository root

The final two unchecked items are the only known gaps in the reproducibility envelope. Both are low-effort to close (running `pip-compile` or a similar tool on the current `requirements.txt` and committing the result), and both are flagged in the project roadmap.

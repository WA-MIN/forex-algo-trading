from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from backtest.engine import PAIRS, run_backtest, run_cv_folds
from backtest.report_generator import generate_report
from backtest.strategies import STRATEGY_REGISTRY, get_strategy

CLEANED_DIR    = PROJECT_DIR / "data" / "processed" / "cleaned"
ALL_PAIRS      = PAIRS
ALL_STRATEGIES = list(STRATEGY_REGISTRY.keys())


# - CLI ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FX backtests and generate an HTML report."
    )
    parser.add_argument("--pairs",           nargs="+", default=["all"],
                        help="Pairs to run. Use 'all' for all 7 pairs.")
    parser.add_argument("--strategies",      nargs="+", default=["all"],
                        help="Strategies to run. Use 'all' for all registered strategies.")
    parser.add_argument("--split",           choices=["val", "test"], default="val",
                        help="Data split to use. Never touch 'test' during development.")
    parser.add_argument("--folds",           type=int, default=0,
                        help="Number of walk-forward folds. 0 = single full-period run.")
    parser.add_argument("--spread-override", type=float, default=None,
                        help="Override spread (pips) for all pairs. Uses SPREAD_TABLE if omitted.")
    parser.add_argument("--tp-pips",         type=float, default=None,
                        help="Take-profit in pips. Overrides strategy default. e.g. --tp-pips 10")
    parser.add_argument("--sl-pips",         type=float, default=None,
                        help="Stop-loss in pips. Overrides strategy default. e.g. --sl-pips 5")
    parser.add_argument("--out",             type=Path, default=None,
                        help="Custom output path for the HTML report.")
    parser.add_argument("--no-browser",      action="store_true",
                        help="Save report without opening it in the browser.")
    return parser.parse_args()


# - Helpers ------------------------------------------------------------------

def resolve_pairs(raw: list[str]) -> list[str]:
    if raw == ["all"]:
        return ALL_PAIRS
    invalid = [p for p in raw if p not in ALL_PAIRS]
    if invalid:
        raise ValueError(f"Unknown pairs: {invalid}. Supported: {ALL_PAIRS}")
    return raw


def resolve_strategies(raw: list[str]) -> list[str]:
    if raw == ["all"]:
        return ALL_STRATEGIES
    invalid = [s for s in raw if s not in STRATEGY_REGISTRY]
    if invalid:
        raise ValueError(f"Unknown strategies: {invalid}. Available: {ALL_STRATEGIES}")
    return raw


def load_prices(pair: str) -> pd.DataFrame:
    path = CLEANED_DIR / f"{pair}_2015_2025_clean.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Cleaned data not found for {pair}.\n"
            f"Expected: {path}\n"
            f"Run scripts/clean_fx_data.py first."
        )
    df = pd.read_parquet(path)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    return df


def split_data(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Val: first 80%. Test: final 20% - only use when done with training."""
    cut = int(len(df) * 0.8)
    return df.iloc[:cut] if split == "val" else df.iloc[cut:]


# - Terminal printing --------------------------------------------------------

SEP  = "-" * 115
SEP2 = "=" * 115


def print_header() -> None:
    print(SEP)
    print(
        f"  {'Pair':<10} {'Strategy':<40} {'Sharpe':>9} {'Return':>9} "
        f"{'MaxDD':>8} {'WinRate':>8} {'Trades':>8} {'Sortino':>9} {'Calmar':>9}"
    )
    print(SEP)


def print_run(pair: str, strat_name: str, r) -> None:
    sharpe_str = f"{r.net_sharpe:+.4f}"
    ret_str    = f"{r.total_return * 100:+.2f}%"
    dd_str     = f"{r.max_drawdown * 100:.2f}%"
    wr_str     = f"{r.win_rate * 100:.1f}%"
    trades_str = f"{r.n_trades:,}"
    sort_str   = f"{r.sortino:+.4f}"
    cal_str    = f"{r.calmar:+.4f}"

    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    RESET  = "\033[0m"

    if r.net_sharpe >= 0.5:
        colour = GREEN
    elif r.net_sharpe >= 0:
        colour = YELLOW
    else:
        colour = RED

    print(
        f"  {pair:<10} {strat_name:<40} "
        f"{colour}{sharpe_str:>9}{RESET} {ret_str:>9} "
        f"{dd_str:>8} {wr_str:>8} {trades_str:>8} {sort_str:>9} {cal_str:>9}"
    )


def print_summary(results: list) -> None:
    sharpes = [r.net_sharpe   for r in results]
    returns = [r.total_return for r in results]
    dds     = [r.max_drawdown for r in results]
    best_i  = int(np.argmax(sharpes))
    worst_i = int(np.argmin(sharpes))
    pos_runs = sum(1 for s in sharpes if s >= 0)

    print()
    print(SEP2)
    print(f"  SUMMARY  ({len(results)} run{'s' if len(results) != 1 else ''})")
    print(SEP2)
    print(f"  Avg Net Sharpe    : {np.mean(sharpes):+.4f}")
    print(f"  Avg Return        : {np.mean(returns) * 100:+.2f}%")
    print(f"  Avg Max Drawdown  : {np.mean(dds) * 100:.2f}%")
    print(f"  Positive runs     : {pos_runs} / {len(results)}")
    print(f"  Best run          : {results[best_i].pair}  {results[best_i].strategy}  "
          f"-> Sharpe {sharpes[best_i]:+.4f}  Return {returns[best_i] * 100:+.2f}%")
    print(f"  Worst run         : {results[worst_i].pair}  {results[worst_i].strategy}  "
          f"-> Sharpe {sharpes[worst_i]:+.4f}  Return {returns[worst_i] * 100:+.2f}%")
    print(SEP2)
    print()


# - Main ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    pairs      = resolve_pairs(args.pairs)
    strategies = resolve_strategies(args.strategies)
    total_runs = len(pairs) * len(strategies)

    tp_label = f"{args.tp_pips} pips (override)" if args.tp_pips is not None else "strategy default"
    sl_label = f"{args.sl_pips} pips (override)" if args.sl_pips is not None else "strategy default"

    print()
    print(SEP2)
    print(f"  FXAlgo Backtest Runner")
    print(SEP2)
    print(f"  Pairs       : {', '.join(pairs)}")
    print(f"  Strategies  : {', '.join(strategies)}")
    print(f"  Split       : {args.split.upper()}")
    print(f"  Folds       : {args.folds if args.folds > 0 else 'single run (no CV)'}")
    print(f"  Spread      : {'table default' if args.spread_override is None else f'{args.spread_override} pips (override)'}")
    print(f"  TP          : {tp_label}")
    print(f"  SL          : {sl_label}")
    print(f"  Total runs  : {total_runs}")
    print(SEP2)
    print()

    print_header()

    results   = []
    completed = 0

    for pair in pairs:
        df       = load_prices(pair)
        df_split = split_data(df, args.split)
        prices   = df_split["close"].reset_index(drop=True)

        for strat_name in strategies:
            strat   = get_strategy(strat_name)
            signals = strat.generate_signals(df_split.reset_index(drop=True))
            signals = signals.reset_index(drop=True)

            if args.folds > 0:
                fold_results = run_cv_folds(
                    signals_df=signals.to_frame(),
                    prices_df=prices.to_frame(),
                    pair=pair,
                    strategy=strat.name,
                    split=args.split,
                    n_folds=args.folds,
                    spread_pips=args.spread_override,
                    tp_pips=args.tp_pips,
                    sl_pips=args.sl_pips,
                )
                results.extend(fold_results)
                last = fold_results[-1]
            else:
                last = run_backtest(
                    signals=signals,
                    prices=prices,
                    pair=pair,
                    strategy=strat.name,
                    split=args.split,
                    spread_pips=args.spread_override,
                    tp_pips=args.tp_pips,
                    sl_pips=args.sl_pips,
                )
                results.append(last)

            completed += 1
            print_run(pair, strat.name, last)
            sys.stdout.flush()

    print_summary(results)

    print(f"  Generating report...")
    out = generate_report(
        results=results,
        output_path=args.out,
        open_browser=not args.no_browser,
    )
    print(f"  Done. {out.name}")
    print()


if __name__ == "__main__":
    main()

    # for commit
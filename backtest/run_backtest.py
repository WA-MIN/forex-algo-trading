from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from backtest.engine import PAIRS, run_backtest, run_wf_folds
from backtest.report_generator import generate_report
from backtest.strategies import STRATEGY_REGISTRY, get_strategy

SPLIT_ROOT_DIR = PROJECT_DIR / "datasets"
TRAIN_DIR = SPLIT_ROOT_DIR / "train"
VAL_DIR = SPLIT_ROOT_DIR / "val"
TEST_DIR = SPLIT_ROOT_DIR / "test"

ALL_PAIRS = PAIRS
ALL_STRATEGIES = list(STRATEGY_REGISTRY.keys())


# argument parser

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FX backtests and generate an HTML report.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # data selection
    parser.add_argument(
        "--pair", nargs="+", default=["all"],
        metavar="PAIR",
        help="One or more pairs to run, or 'all'.\nExample: --pair EURUSD GBPUSD",
    )
    parser.add_argument(
        "--strategy", nargs="+", default=["all"],
        metavar="STRATEGY",
        help="One or more strategy names, or 'all'.\nExample: --strategy MACrossover_f20_s50_EMA",
    )
    parser.add_argument(
        "--split", choices=["train", "val", "test"], default="val",
        help="Data split to run on. Default: val",
    )

    # walk-forward CV
    parser.add_argument(
        "--folds", type=int, default=0,
        metavar="N",
        help="Walk-forward folds. 0 = single full-period run.",
    )

    # capital + execution
    parser.add_argument(
        "--capital", type=float, default=10_000.0,
        metavar="AMOUNT",
        help="Starting capital in USD. Default: 10000\nSetting this switches the report badge to SIM.",
    )
    parser.add_argument(
        "--spread-override", type=float, default=None,
        metavar="PIPS",
        help="Override spread (pips) for all pairs.",
    )
    parser.add_argument(
        "--tp-pips", type=float, default=None,
        metavar="PIPS",
        help="Take-profit distance in pips.",
    )
    parser.add_argument(
        "--sl-pips", type=float, default=None,
        metavar="PIPS",
        help="Stop-loss distance in pips.",
    )
    parser.add_argument(
        "--max-hold", type=int, default=None,
        metavar="BARS",
        help="Maximum bars to hold a position.\nWith 1-min data: 60 = 1 hour, 240 = 4 hours.",
    )

    # time filters
    parser.add_argument(
        "--session", type=str, default=None,
        choices=["london", "ny", "tokyo", "sydney"],
        help="Only trade during this session window (UTC).\n"
             "london 07-16  |  ny 13-22  |  tokyo 23-08  |  sydney 21-06",
    )
    parser.add_argument(
        "--entry-time", type=str, default=None,
        metavar="HH:MM",
        help="Only enter trades at or after this UTC time each day.\nExample: --entry-time 09:00",
    )
    parser.add_argument(
        "--resample", type=str, default=None,
        metavar="FREQ",
        help="Resample bars before running. Any pandas offset string.\nExamples: 1H  4H  15min  1D",
    )

    # direction mode
    parser.add_argument(
        "--direction",
        choices=["long_short", "long_only", "short_only"],
        default="long_short",
        metavar="MODE",
        help=(
            "Which signal directions to trade.\n"
            "long_short -- trade both longs and shorts (default)\n"
            "long_only  -- suppress all short (-1) signals\n"
            "short_only -- suppress all long  (+1) signals"
        ),
    )

    # output
    parser.add_argument(
        "--out", type=Path, default=None,
        metavar="PATH",
        help="Custom output path for the HTML report.",
    )
    parser.add_argument(
        "--no-browser", action="store_true",
        help="Save report without opening it in the browser.",
    )

    return parser.parse_args()


# resolvers

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
        raise ValueError(
            f"Unknown strategies: {invalid}. Available: {ALL_STRATEGIES}"
        )
    return raw


# data loader

def load_split_data(pair: str, split: str) -> pd.DataFrame:
    split_map = {"train": TRAIN_DIR, "val": VAL_DIR, "test": TEST_DIR}
    path = split_map[split] / f"{pair}_{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Split parquet not found: {path}\n"
            f"Run scripts/split_fx_data.py first."
        )
    df = pd.read_parquet(path)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df.sort_values("timestamp_utc").reset_index(drop=True)


# console formatting

SEP = "-" * 115
SEP2 = "=" * 115

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
RESET = "\033[0m"


def _sharpe_colour(v: float) -> str:
    if v >= 0.5:
        return GREEN
    if v >= 0.0:
        return YELLOW
    return RED


def print_banner(
    pairs: list[str],
    strategies: list[str],
    args: argparse.Namespace,
    total_runs: int,
    mode: str,
) -> None:
    cap_str = f"${args.capital:,.2f}"
    max_hold_str = f"{args.max_hold} bars" if args.max_hold else "none"
    session_str = args.session if args.session else "all hours"
    entry_str = f"{args.entry_time} UTC" if args.entry_time else "any bar"
    resample_str = args.resample if args.resample else "native 1-min"
    spread_str = (
        f"{args.spread_override} pips (override)"
        if args.spread_override is not None
        else "table default"
    )
    tp_str = f"{args.tp_pips} pips" if args.tp_pips is not None else "none"
    sl_str = f"{args.sl_pips} pips" if args.sl_pips is not None else "none"
    mode_colour = CYAN if mode == "simulation" else YELLOW

    print()
    print(SEP2)
    print(f"  {CYAN}FXAlgo Backtest Runner{RESET}")
    print(SEP2)
    print(f"  Pairs        : {', '.join(pairs)}")
    print(f"  Strategies   : {', '.join(strategies)}")
    print(f"  Split        : {args.split.upper()}")
    print(f"  Folds        : {args.folds if args.folds > 0 else 'single run (no CV)'}")
    print(f"  Mode         : {mode_colour}{mode.upper()}{RESET}")
    print(f"  Direction    : {args.direction}")
    print(f"  Capital      : {cap_str}")
    print(f"  Max Hold     : {max_hold_str}")
    print(f"  Session      : {session_str}")
    print(f"  Entry Time   : {entry_str}")
    print(f"  Resample     : {resample_str}")
    print(f"  Spread       : {spread_str}")
    print(f"  TP           : {tp_str}")
    print(f"  SL           : {sl_str}")
    print(f"  Total runs   : {total_runs}")
    print(SEP2)
    print()


def print_header() -> None:
    print(SEP)
    print(
        f"  {'Pair':<10} {'Strategy':<40} {'Sharpe':>9} {'Return':>9} "
        f"{'MaxDD':>8} {'WinRate':>8} {'Trades':>8} {'Sortino':>9} {'Calmar':>9}"
    )
    print(SEP)


def print_run(pair: str, strat_name: str, r) -> None:
    colour = _sharpe_colour(r.net_sharpe)
    print(
        f"  {pair:<10} {strat_name:<40} "
        f"{colour}{r.net_sharpe:+.4f}{RESET} {r.total_return * 100:+8.2f}% "
        f"{r.max_drawdown * 100:7.2f}% {r.win_rate * 100:7.1f}% "
        f"{r.n_trades:>8,} {r.sortino:+9.4f} {r.calmar:+9.4f}"
    )


def print_summary(results: list) -> None:
    sharpes = [r.net_sharpe for r in results]
    returns = [r.total_return for r in results]
    dds = [r.max_drawdown for r in results]
    best_i = int(np.argmax(sharpes))
    worst_i = int(np.argmin(sharpes))
    pos_runs = sum(1 for s in sharpes if s >= 0)

    print()
    print(SEP2)
    print(f"  {CYAN}SUMMARY{RESET}  ({len(results)} run{'s' if len(results) != 1 else ''})")
    print(SEP2)
    print(f"  Avg Net Sharpe    : {_sharpe_colour(np.mean(sharpes))}{np.mean(sharpes):+.4f}{RESET}")
    print(f"  Avg Return        : {np.mean(returns) * 100:+.2f}%")
    print(f"  Avg Max Drawdown  : {np.mean(dds) * 100:.2f}%")
    print(f"  Positive runs     : {pos_runs} / {len(results)}")
    print(
        f"  Best run          : {results[best_i].pair}  {results[best_i].strategy}  "
        f"-> Sharpe {sharpes[best_i]:+.4f}  Return {returns[best_i] * 100:+.2f}%"
    )
    print(
        f"  Worst run         : {results[worst_i].pair}  {results[worst_i].strategy}  "
        f"-> Sharpe {sharpes[worst_i]:+.4f}  Return {returns[worst_i] * 100:+.2f}%"
    )
    print(SEP2)
    print()


# main

def main() -> None:
    args = parse_args()
    pairs = resolve_pairs(args.pair)
    strategies = resolve_strategies(args.strategy)
    total_runs = len(pairs) * len(strategies)

    # mode -- simulation when user explicitly set a non-default capital
    mode = "simulation" if args.capital != 10_000.0 else "research"

    print_banner(pairs, strategies, args, total_runs, mode)
    print_header()

    results = []
    completed = 0

    for pair in pairs:

        # walk-forward CV path
        if args.folds > 0:
            for strat_name in strategies:
                fold_results = run_wf_folds(
                    pair=pair,
                    strategy=strat_name,
                    n_folds=args.folds,
                    spread_pips=args.spread_override,
                    tp_pips=args.tp_pips,
                    sl_pips=args.sl_pips,
                    session=args.session,
                    entry_time=args.entry_time,
                    resample=args.resample,
                    direction_mode=args.direction,   # fix: was missing
                )
                results.extend(fold_results)
                completed += 1
                print_run(pair, strat_name, fold_results[-1])
                sys.stdout.flush()

        # single-run path
        else:
            df_split = load_split_data(pair, args.split)
            prices = df_split["close"].reset_index(drop=True)
            timestamps = (
                df_split["timestamp_utc"]
                .dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                .tolist()
                if "timestamp_utc" in df_split.columns else []
            )

            for strat_name in strategies:
                strat = get_strategy(strat_name)
                signals = strat.generate_signals(
                    df_split.reset_index(drop=True)
                ).reset_index(drop=True)

                r = run_backtest(
                    signals=signals,
                    prices=prices,
                    pair=pair,
                    strategy=strat.name,
                    split=args.split,
                    spread_pips=args.spread_override,
                    tp_pips=args.tp_pips,
                    sl_pips=args.sl_pips,
                    capital_initial=args.capital,
                    max_hold_bars=args.max_hold,
                    timestamps=timestamps,
                    session=args.session,
                    entry_time=args.entry_time,
                    resample=args.resample,
                    direction_mode=args.direction,
                    mode=mode,
                )
                results.append(r)
                completed += 1
                print_run(pair, strat.name, r)
                sys.stdout.flush()

    print_summary(results)

    print("  Generating report...")
    out = generate_report(
        results=results,
        out_path=args.out,
        open_browser=not args.no_browser,
    )
    print(f"  Done. {out.name}")
    print()


if __name__ == "__main__":
    main()
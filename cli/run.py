from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib  import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from cli.config              import SimConfig
from core.data_loader        import resolve_data
from backtest.strategies     import get_strategy
from backtest.engine         import run_backtest, run_wf_folds, PAIRS as ALL_PAIRS
from backtest.report_generator import generate_report


#  date parser 

def _parse_dt(s: str) -> datetime:
    """Accept 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM', return UTC-aware datetime."""
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(
        f"Cannot parse date '{s}'. Use 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM'."
    )


#  argument parser 

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fxalgo",
        description="FXAlgo — backtest and simulate forex strategies.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Research mode — full val split
  python cli/run.py --pair EURUSD --strategy MACrossover_f20_s50_EMA --split val

  # Simulation mode — custom date range
  python cli/run.py --pair EURUSD --strategy MACrossover_f20_s50_EMA \\
      --from "2021-03-15 09:00" --to "2021-03-15 17:00" --capital 5000

  # TP/SL + max hold bars
  python cli/run.py --pair EURUSD --strategy MACrossover_f20_s50_EMA \\
      --from "2021-06-01" --to "2021-06-30" --capital 10000 \\
      --tp-pips 10 --sl-pips 5 --max-hold 60

  # Walk-forward cross-validation
  python cli/run.py --pair EURUSD --strategy MACrossover_f20_s50_EMA --folds 5

  # All pairs, all strategies
  python cli/run.py --pair all --strategy all --split val
""",
    )

    p.add_argument("--pair",     required=True,
                   help="Currency pair e.g. EURUSD, or 'all'")
    p.add_argument("--strategy", required=True,
                   help="Strategy name, or 'all'")
    p.add_argument("--split",    default="val",
                   choices=["train", "val", "test",
                            "fold_0", "fold_1", "fold_2", "fold_3", "fold_4"],
                   help="Predefined split (ignored when --from/--to are given)")
    p.add_argument("--from",     dest="date_from", type=_parse_dt, default=None,
                   metavar="DATETIME",
                   help="Simulation start — overrides --split")
    p.add_argument("--to",       dest="date_to",   type=_parse_dt, default=None,
                   metavar="DATETIME",
                   help="Simulation end")
    p.add_argument("--capital",  type=float, default=10_000.0,
                   help="Starting capital in USD (default: 10000)")
    p.add_argument("--tp-pips",  type=float, default=None,
                   help="Take-profit in pips")
    p.add_argument("--sl-pips",  type=float, default=None,
                   help="Stop-loss in pips")
    p.add_argument("--max-hold", type=int,   default=None, dest="max_hold_bars",
                   metavar="BARS",
                   help="Force-close after N bars")
    p.add_argument("--entry-time", default=None, metavar="HH:MM",
                   help="First valid entry time UTC e.g. '09:00'")
    p.add_argument("--folds",    type=int,   default=0,
                   help="Walk-forward CV folds (0 = single run)")
    p.add_argument("--spread",   type=float, default=None, dest="spread_override",
                   help="Override default spread in pips")
    p.add_argument("--session",  default=None,
                   choices=["london", "ny", "asia", "overlap"],
                   help="Filter to trading session hours (UTC)")
    p.add_argument("--out",      default=None, dest="output_path",
                   help="Output HTML path (auto-named if omitted)")
    p.add_argument("--no-browser", action="store_true",
                   help="Write report but do not open browser")

    return p


def _build_config(args: argparse.Namespace) -> SimConfig:
    mode = "simulation" if args.date_from is not None else "research"
    return SimConfig(
        pair            = args.pair,
        strategy        = args.strategy,
        split           = args.split,
        date_from       = args.date_from,
        date_to         = args.date_to,
        capital         = args.capital,
        tp_pips         = args.tp_pips,
        sl_pips         = args.sl_pips,
        max_hold_bars   = args.max_hold_bars,
        entry_time      = args.entry_time,
        folds           = args.folds,
        spread_override = args.spread_override,
        session         = args.session,
        open_browser    = not args.no_browser,
        output_path     = args.output_path,
        no_browser      = args.no_browser,
        mode            = mode,
    )


# ── pair / strategy resolution ────────────────────────────────────────────────

def _resolve_pairs(cfg: SimConfig) -> list[str]:
    if cfg.pair.lower() == "all":
        return ALL_PAIRS
    pair = cfg.pair.upper()
    if pair not in ALL_PAIRS:
        raise ValueError(
            f"Unknown pair '{cfg.pair}'. Supported: {ALL_PAIRS}. "
            f"Use 'all' to run every pair."
        )
    return [pair]


def _resolve_strategies(cfg: SimConfig) -> list[str]:
    from backtest.strategies import STRATEGY_REGISTRY
    if cfg.strategy.lower() == "all":
        return list(STRATEGY_REGISTRY.keys())
    if cfg.strategy not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy '{cfg.strategy}'.\n"
            f"Available: {list(STRATEGY_REGISTRY.keys())}"
        )
    return [cfg.strategy]


#  main 

def main() -> None:
    parser     = _build_parser()
    args       = parser.parse_args()
    cfg        = _build_config(args)
    pairs      = _resolve_pairs(cfg)
    strategies = _resolve_strategies(cfg)

    print(f"\n{'='*60}")
    print(f"  FXAlgo  |  mode={cfg.mode}  |  split={cfg.split}")
    print(f"  pairs={pairs}")
    print(f"  strategies={strategies}")
    print(f"  capital=${cfg.capital:,.0f}  folds={cfg.folds}")
    print(f"{'='*60}\n")

    all_results = []

    #  walk-forward CV 
    if cfg.folds > 0:
        for pair in pairs:
            for strat_name in strategies:
                print(f"  [{pair}] {strat_name} — {cfg.folds}-fold WF-CV …")
                fold_results = run_wf_folds(
                    pair        = pair,
                    strategy    = strat_name,
                    n_folds     = cfg.folds,
                    spread_pips = cfg.spread_override,
                    tp_pips     = cfg.tp_pips,
                    sl_pips     = cfg.sl_pips,
                )
                all_results.extend(fold_results)
        _write_report(cfg, all_results)
        return

    #  single run 
    for pair in pairs:
        for strat_name in strategies:
            print(f"  [{pair}] {strat_name} — loading data …")
            df = resolve_data(cfg, pair)

            strat   = get_strategy(strat_name)
            signals = strat.generate_signals(df).reset_index(drop=True)
            prices  = df["close"].reset_index(drop=True)

            timestamps = (
                df["timestamp_utc"]
                .dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                .tolist()
                if "timestamp_utc" in df.columns else []
            )

            split_label = cfg.split or "custom"
            if cfg.mode == "simulation" and cfg.date_from:
                split_label = cfg.date_from.strftime("%Y%m%d")

            print(f"  [{pair}] running backtest ({len(df):,} bars) …")
            result = run_backtest(
                signals         = signals,
                prices          = prices,
                pair            = pair,
                strategy        = strat_name,
                split           = split_label,
                spread_pips     = cfg.spread_override,
                tp_pips         = cfg.tp_pips,
                sl_pips         = cfg.sl_pips,
                capital_initial = cfg.capital,
                max_hold_bars   = cfg.max_hold_bars,
                timestamps      = timestamps,
                session         = cfg.session,
                mode            = cfg.mode,
            )
            all_results.append(result)
            print(
                f"  [{pair}] done — "
                f"Sharpe={result.net_sharpe:.3f}  "
                f"Return={result.total_return * 100:.2f}%  "
                f"Trades={result.n_trades}  "
                f"Final=${result.capital_final:,.2f}"
            )

    _write_report(cfg, all_results)


def _write_report(cfg: SimConfig, results: list) -> None:
    out = generate_report(
        results      = results,
        output_path  = Path(cfg.output_path) if cfg.output_path else None,
        open_browser = cfg.open_browser,
    )
    print(f"\n  Report: {out}\n")


if __name__ == "__main__":
    main()

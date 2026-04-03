from __future__ import annotations

import json
import math
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from backtest.engine import BacktestResult

BACKTEST_DIR  = Path(__file__).resolve().parent
TEMPLATE_PATH = BACKTEST_DIR / "templates" / "report.html"
REPORTS_DIR   = BACKTEST_DIR / "reports"

SPREAD_TABLE_JS = {
    "EURUSD": 1.0,
    "GBPUSD": 1.2,
    "USDJPY": 1.5,
    "USDCHF": 1.5,
    "USDCAD": 1.8,
    "AUDUSD": 1.4,
    "NZDUSD": 1.8,
}


# - Helpers -
def _safe_float(v: float, precision: int = 6) -> float:
    """Round a float and replace inf/nan with 0 so JSON never breaks."""
    if not math.isfinite(v):
        return 0.0
    return round(float(v), precision)


def _safe_list(lst: list, precision: int = 6) -> list:
    """Clean a list of floats — replace inf/nan with 0."""
    return [_safe_float(v, precision) for v in lst]


# - Serialisation -

def _result_to_dict(r: BacktestResult) -> dict:
    return {
        "pair":           r.pair,
        "strategy":       r.strategy,
        "split":          r.split,
        "fold_index":     r.fold_index,
        "equity":         _safe_list(r.equity,         6),
        "rolling_sharpe": _safe_list(r.rolling_sharpe, 4),
        "signal_dist": {
            "Long":  int(r.signal_dist.get("Long",  0)),
            "Short": int(r.signal_dist.get("Short", 0)),
            "Flat":  int(r.signal_dist.get("Flat",  0)),
        },
        "metrics": {
            "net_sharpe":     _safe_float(r.net_sharpe,     4),
            "gross_sharpe":   _safe_float(r.gross_sharpe,   4),
            "total_return":   _safe_float(r.total_return,   6),
            "max_drawdown":   _safe_float(r.max_drawdown,   6),
            "calmar":         _safe_float(r.calmar,         4),
            "win_rate":       _safe_float(r.win_rate,       4),
            "profit_factor":  _safe_float(r.profit_factor,  4),
            "avg_trade_bars": _safe_float(r.avg_trade_bars, 2),
            "turnover":       _safe_float(r.turnover,       6),
            "sortino":        _safe_float(r.sortino,        4),
            "n_trades":       int(r.n_trades),
            "spread_pips":    _safe_float(r.spread_pips,    4),
        },
    }


# - Filename / title -

def _build_filename(results: list[BacktestResult]) -> str:
    """
    BT_EURUSD_AUDUSD_MACrossover_f20_s50_EMA_VAL.html
    Truncates to Xpairs / Xstrats if more than 3 pairs or 2 strategies.
    """
    pairs  = sorted({r.pair     for r in results})
    strats = sorted({r.strategy for r in results})
    splits = sorted({r.split    for r in results})

    p_part  = "_".join(pairs) if len(pairs) <= 3 else f"{len(pairs)}pairs"
    s_part  = "_".join(
        s.replace(" ", "").replace("/", "") for s in strats
    ) if len(strats) <= 2 else f"{len(strats)}strats"
    sp_part = "_".join(s.upper() for s in splits)

    return f"BT_{p_part}_{s_part}_{sp_part}.html"


def _build_title(results: list[BacktestResult]) -> str:
    pairs  = sorted({r.pair     for r in results})
    strats = sorted({r.strategy for r in results})
    splits = sorted({r.split    for r in results})

    p_str  = ", ".join(pairs)  if len(pairs)  <= 3 else f"{len(pairs)} pairs"
    s_str  = ", ".join(strats) if len(strats) <= 2 else f"{len(strats)} strategies"
    sp_str = "/".join(s.upper() for s in splits)

    return f"FXAlgo  {p_str}  {s_str}  ({sp_str})"


# - Payload -
def _build_payload(results: list[BacktestResult]) -> dict:
    pairs  = sorted({r.pair     for r in results})
    strats = sorted({r.strategy for r in results})
    splits = sorted({r.split    for r in results})

    return {
        "meta": {
            "title":        _build_title(results),
            "pairs":        pairs,
            "strategies":   strats,
            "splits":       splits,
            "generated":    datetime.now(timezone.utc).strftime("%d %b %Y %H:%M UTC"),
            "spread_table": SPREAD_TABLE_JS,
        },
        "results": [_result_to_dict(r) for r in results],
    }


# - Inject -
def _inject(template: str, payload: dict) -> str:
    data_json = json.dumps(payload, separators=(",", ":"))
    return (
        template
        .replace("__TITLE__",     payload["meta"]["title"])
        .replace("__DATA_JSON__", data_json)
        .replace("__GENERATED__", payload["meta"]["generated"])
    )


# - Public API -
def generate_report(
    results:      list[BacktestResult],
    output_path:  Optional[Path] = None,
    open_browser: bool = True,
) -> Path:
    """
    Render the HTML report and write it to disk.

    Parameters
    ----------
    results      : list of BacktestResult from engine.run_backtest / run_cv_folds
    output_path  : explicit save path; auto-named if None
    open_browser : open the file in the default browser when done

    Returns
    -------
    Path to the written HTML file.
    """
    if not results:
        raise ValueError("No results to report.")

    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(
            f"Report template not found: {TEMPLATE_PATH}\n"
            f"Expected at backtest/templates/report.html"
        )

    out = Path(output_path) if output_path else REPORTS_DIR / _build_filename(results)
    out.parent.mkdir(parents=True, exist_ok=True)

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    payload  = _build_payload(results)
    html     = _inject(template, payload)

    out.write_text(html, encoding="utf-8")

    size_kb = out.stat().st_size / 1024
    print(f"  Report saved : {out}")
    print(f"  File size    : {size_kb:.1f} KB")

    if open_browser:
        webbrowser.open(out.as_uri())
        print(f"  Opened in browser.")

    return out
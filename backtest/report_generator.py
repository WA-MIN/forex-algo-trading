from __future__ import annotations

import json
import math
import webbrowser
from datetime import datetime, timezone
from pathlib  import Path
from typing   import Optional

from backtest.engine import BacktestResult

BACKTEST_DIR  = Path(__file__).resolve().parent
TEMPLATE_PATH = BACKTEST_DIR / "templates" / "report.html"
REPORTS_DIR   = BACKTEST_DIR / "reports"

SPREAD_TABLE_JS: dict[str, float] = {
    "EURUSD": 1.0,
    "GBPUSD": 1.2,
    "USDJPY": 1.5,
    "USDCHF": 1.5,
    "USDCAD": 1.8,
    "AUDUSD": 1.4,
    "NZDUSD": 1.8,
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _safe_float(v: float, precision: int = 6) -> float:
    """Round a float; replace inf/nan with 0 so JSON never breaks."""
    if not math.isfinite(v):
        return 0.0
    return round(float(v), precision)


def _safe_list(lst: list, precision: int = 6) -> list:
    """Clean a list of floats — replace inf/nan with 0."""
    return [_safe_float(v, precision) for v in lst]


# ── serialisation ─────────────────────────────────────────────────────────────

def _result_to_dict(r: BacktestResult) -> dict:
    return {
        "pair":            r.pair,
        "strategy":        r.strategy,
        "split":           r.split,
        "mode":            r.mode,
        "fold_index":      r.fold_index,
        "equity":          _safe_list(r.equity,          6),
        "equity_dollars":  _safe_list(r.equity_dollars,  2),
        "rolling_sharpe":  _safe_list(r.rolling_sharpe,  4),
        "timestamps":      r.timestamps,
        "signal_dist": {
            "Long":  int(r.signal_dist.get("Long",  0)),
            "Short": int(r.signal_dist.get("Short", 0)),
            "Flat":  int(r.signal_dist.get("Flat",  0)),
        },
        "metrics": {
            "net_sharpe":      _safe_float(r.net_sharpe,     4),
            "gross_sharpe":    _safe_float(r.gross_sharpe,   4),
            "total_return":    _safe_float(r.total_return,   6),
            "max_drawdown":    _safe_float(r.max_drawdown,   6),
            "calmar":          _safe_float(r.calmar,         4),
            "win_rate":        _safe_float(r.win_rate,       4),
            "profit_factor":   _safe_float(r.profit_factor,  4),
            "avg_trade_bars":  _safe_float(r.avg_trade_bars, 2),
            "turnover":        _safe_float(r.turnover,       6),
            "sortino":         _safe_float(r.sortino,        4),
            "n_trades":        int(r.n_trades),
            "spread_pips":     _safe_float(r.spread_pips,    4),
            "capital_initial": _safe_float(r.capital_initial, 2),
            "capital_final":   _safe_float(r.capital_final,   2),
        },
        "trade_log": r.trade_log,
    }


# ── filename / title ──────────────────────────────────────────────────────────

def _build_filename(results: list[BacktestResult]) -> str:
    """
    Research mode  → BT_EURUSD_MACrossover_f20_s50_EMA_VAL.html
    Simulation mode → SIM_EURUSD_MACrossover_f20_s50_EMA_20210601.html
    """
    pairs  = sorted({r.pair     for r in results})
    strats = sorted({r.strategy for r in results})
    splits = sorted({r.split    for r in results})
    mode   = results[0].mode if results else "research"

    prefix  = "SIM" if mode == "simulation" else "BT"
    p_part  = "_".join(pairs)  if len(pairs)  <= 3 else f"{len(pairs)}pairs"
    s_part  = "_".join(
        s.replace(" ", "").replace("/", "") for s in strats
    ) if len(strats) <= 2 else f"{len(strats)}strats"
    sp_part = "_".join(s.upper() for s in splits)

    return f"{prefix}_{p_part}_{s_part}_{sp_part}.html"


def _build_title(results: list[BacktestResult]) -> str:
    pairs  = sorted({r.pair     for r in results})
    strats = sorted({r.strategy for r in results})
    splits = sorted({r.split    for r in results})
    mode   = results[0].mode if results else "research"

    p_str  = ", ".join(pairs)  if len(pairs)  <= 3 else f"{len(pairs)} pairs"
    s_str  = ", ".join(strats) if len(strats) <= 2 else f"{len(strats)} strategies"
    sp_str = "/".join(s.upper() for s in splits)
    mode_label = "Simulation" if mode == "simulation" else "Backtest"

    return f"FXAlgo {mode_label}  {p_str}  {s_str}  ({sp_str})"


# ── payload ───────────────────────────────────────────────────────────────────

def _build_payload(results: list[BacktestResult]) -> dict:
    pairs  = sorted({r.pair     for r in results})
    strats = sorted({r.strategy for r in results})
    splits = sorted({r.split    for r in results})
    mode   = results[0].mode if results else "research"

    # Capital summary for the meta strip
    capitals_initial = list({r.capital_initial for r in results})
    capitals_final   = [r.capital_final for r in results]

    return {
        "meta": {
            "title":            _build_title(results),
            "mode":             mode,
            "pairs":            pairs,
            "strategies":       strats,
            "splits":           splits,
            "generated":        datetime.now(timezone.utc).strftime("%d %b %Y %H:%M UTC"),
            "spread_table":     SPREAD_TABLE_JS,
            "capital_initial":  capitals_initial[0] if len(capitals_initial) == 1 else None,
            "capital_final_avg": round(sum(capitals_final) / len(capitals_final), 2)
                                  if capitals_final else None,
        },
        "results": [_result_to_dict(r) for r in results],
    }


# ── inject ────────────────────────────────────────────────────────────────────

def _inject(template: str, payload: dict) -> str:
    data_json = json.dumps(payload, separators=(",", ":"))
    return (
        template
        .replace("__TITLE__",     payload["meta"]["title"])
        .replace("__DATA_JSON__", data_json)
        .replace("__GENERATED__", payload["meta"]["generated"])
    )


# ── public API ────────────────────────────────────────────────────────────────

def generate_report(
    results:      list[BacktestResult],
    output_path:  Optional[Path] = None,
    open_browser: bool           = True,
) -> Path:
    """
    Render the HTML report and write it to disk.

    Parameters
    ----------
    results      : list of BacktestResult
    output_path  : explicit save path; auto-named if None
    open_browser : open in default browser when done

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

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path
from typing      import List, Optional

import numpy  as np
import pandas as pd

from backtest.strategies import STRATEGY_REGISTRY, get_strategy

# ---------------------------------------------------------------------------
# Supported pairs + pip sizes
# ---------------------------------------------------------------------------

PAIRS: list[str] = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
    "AUDUSD", "USDCAD", "NZDUSD", "EURGBP",
]

# pip size per pair (JPY pairs: 0.01, everything else: 0.0001)
_PIP: dict[str, float] = {p: (0.01 if "JPY" in p else 0.0001) for p in PAIRS}

# default spread table (pips)
_DEFAULT_SPREAD: dict[str, float] = {
    "EURUSD": 0.6, "GBPUSD": 0.8, "USDJPY": 0.7, "USDCHF": 1.0,
    "AUDUSD": 0.8, "USDCAD": 1.0, "NZDUSD": 1.2, "EURGBP": 1.0,
}

# data root (datasets/ lives at project root)
_DATASETS = Path(__file__).resolve().parent.parent / "datasets"


# ---------------------------------------------------------------------------
# Session filters (UTC hour ranges, inclusive start exclusive end)
# ---------------------------------------------------------------------------

_SESSION_HOURS: dict[str, tuple[int, int]] = {
    "london": (7,  16),
    "ny":     (13, 22),
    "tokyo":  (23,  8),   # wraps midnight
    "sydney": (21,  6),   # wraps midnight
}


def _in_session(hour: pd.Series, session: str) -> pd.Series:
    start, end = _SESSION_HOURS[session]
    if start < end:
        return (hour >= start) & (hour < end)
    # wraps midnight
    return (hour >= start) | (hour < end)


# ---------------------------------------------------------------------------
# BacktestResult
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    pair:           str
    strategy:       str
    split:          str
    mode:           str            # "research" | "simulation"
    direction_mode: str            # "long_short" | "long_only" | "short_only"
    fold_index:     int            # 0 = single run; 1-N = WF fold number

    # time-series
    equity:         List[float] = field(default_factory=list)
    equity_dollars: List[float] = field(default_factory=list)
    rolling_sharpe: List[float] = field(default_factory=list)
    timestamps:     List[str]   = field(default_factory=list)

    # signal distribution
    signal_dist:    dict        = field(default_factory=dict)

    # scalar metrics
    net_sharpe:     float = 0.0
    gross_sharpe:   float = 0.0
    total_return:   float = 0.0
    max_drawdown:   float = 0.0
    calmar:         float = 0.0
    win_rate:       float = 0.0
    profit_factor:  float = 0.0
    avg_trade_bars: float = 0.0
    turnover:       float = 0.0
    sortino:        float = 0.0
    n_trades:       int   = 0
    capital_initial: float = 10_000.0
    capital_final:   float = 0.0

    # trade log: list of dicts
    trade_log: List[dict] = field(default_factory=list)

    @property
    def metrics(self) -> dict:
        return {
            "net_sharpe":      self.net_sharpe,
            "gross_sharpe":    self.gross_sharpe,
            "total_return":    self.total_return,
            "max_drawdown":    self.max_drawdown,
            "calmar":          self.calmar,
            "win_rate":        self.win_rate,
            "profit_factor":   self.profit_factor,
            "avg_trade_bars":  self.avg_trade_bars,
            "turnover":        self.turnover,
            "sortino":         self.sortino,
            "n_trades":        self.n_trades,
            "capital_initial": self.capital_initial,
            "capital_final":   self.capital_final,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sharpe(returns: np.ndarray, periods_per_year: int = 252 * 390) -> float:
    """Annualised Sharpe from a bar-level return array (1-min default).

    NOTE: always pass the *full* bar_returns array (including zero bars).
    Passing only non-zero bars inflates Sharpe by shrinking the denominator.
    """
    if len(returns) < 2:
        return 0.0
    std = returns.std()
    if std == 0:
        return 0.0
    return float(returns.mean() / std * np.sqrt(periods_per_year))


def _sortino(returns: np.ndarray, periods_per_year: int = 252 * 390) -> float:
    """Annualised Sortino from a bar-level return array.

    NOTE: always pass the *full* bar_returns array (including zero bars).
    """
    downside = returns[returns < 0]
    if len(downside) < 2:
        return 0.0
    dd_std = downside.std()
    if dd_std == 0:
        return 0.0
    return float(returns.mean() / dd_std * np.sqrt(periods_per_year))


def _max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd   = (equity - peak) / peak
    return float(dd.min())


def _rolling_sharpe(
    returns: np.ndarray,
    window: int = 390,          # BUG FIX: was 21 (21 minutes). 390 = 1 trading day at 1-min
) -> list[float]:
    """Rolling annualised Sharpe over `window` bars.

    window=390  -> 1 trading day  (default, 1-min data)
    window=1950 -> 1 trading week (5 days x 390 min)
    """
    out = []
    for i in range(len(returns)):
        w = returns[max(0, i - window + 1): i + 1]
        out.append(_sharpe(w) if len(w) >= 2 else 0.0)
    return out


def _resample_df(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    df = df.copy()
    df = df.set_index("timestamp_utc")
    agg = {
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }
    existing = {k: v for k, v in agg.items() if k in df.columns}
    df = df.resample(freq).agg(existing).dropna(subset=["close"])
    df = df.reset_index()
    return df


# ---------------------------------------------------------------------------
# Core backtest engine
# ---------------------------------------------------------------------------

def run_backtest(
    signals:        pd.Series,
    prices:         pd.Series,
    pair:           str,
    strategy:       str,
    split:          str          = "val",
    spread_pips:    Optional[float] = None,
    tp_pips:        Optional[float] = None,
    sl_pips:        Optional[float] = None,
    capital_initial: float       = 10_000.0,
    max_hold_bars:  Optional[int]   = None,
    timestamps:     Optional[list]  = None,
    session:        Optional[str]   = None,
    entry_time:     Optional[str]   = None,
    resample:       Optional[str]   = None,
    direction_mode: str          = "long_short",
    mode:           str          = "research",
    fold_index:     int          = 0,
    df_full:        Optional[pd.DataFrame] = None,
) -> BacktestResult:
    """
    Run a single backtest and return a BacktestResult.

    Parameters
    ----------
    signals        : integer Series of crossover events (1, -1, 0)
    prices         : close price Series aligned to signals
    pair           : currency pair string e.g. "EURUSD"
    strategy       : strategy name string
    split          : "train" | "val" | "test"
    spread_pips    : override spread in pips (uses table default if None)
    tp_pips        : take-profit distance in pips (None = no TP)
    sl_pips        : stop-loss distance in pips  (None = no SL)
    capital_initial: starting capital in USD
    max_hold_bars  : maximum bars to hold a position (None = no limit)
    timestamps     : list of ISO timestamp strings aligned to prices
    session        : session filter name ("london"|"ny"|"tokyo"|"sydney")
    entry_time     : "HH:MM" UTC -- only enter at/after this time each day
    resample       : pandas offset string to resample before running
    direction_mode : "long_short" | "long_only" | "short_only"
    mode           : "research" | "simulation"
    fold_index     : walk-forward fold number (0 = single run)
    df_full        : full OHLCV DataFrame (needed for session/entry filtering)
    """

    pip   = _PIP.get(pair, 0.0001)
    spread = (spread_pips if spread_pips is not None
              else _DEFAULT_SPREAD.get(pair, 1.0)) * pip

    prices  = prices.reset_index(drop=True)
    signals = signals.reset_index(drop=True)

    # -- direction filter ----------------------------------------------------
    if direction_mode == "long_only":
        signals = signals.where(signals >= 0, 0)
    elif direction_mode == "short_only":
        signals = signals.where(signals <= 0, 0)

    # -- session / entry-time filter -----------------------------------------
    if (session or entry_time) and df_full is not None:
        ts = pd.to_datetime(df_full["timestamp_utc"], utc=True)
        mask = pd.Series(True, index=signals.index)
        if session:
            mask &= _in_session(ts.dt.hour.values, session)
        if entry_time:
            hh, mm = map(int, entry_time.split(":"))
            mask &= (ts.dt.hour > hh) | ((ts.dt.hour == hh) & (ts.dt.minute >= mm))
        signals = signals.where(mask.values, 0)

    # -- forward-fill crossover events into held positions -------------------
    pos = signals.replace(0, np.nan).ffill().fillna(0).astype(int)

    # shift by 1: enter on the bar AFTER the signal
    pos = pos.shift(1).fillna(0).astype(int)

    n = len(prices)
    price_arr  = prices.values.astype(float)
    pos_arr    = pos.values.astype(int)
    signal_arr = signals.values.astype(int)

    tp_dist = tp_pips * pip if tp_pips is not None else None
    sl_dist = sl_pips * pip if sl_pips is not None else None

    # -- trade simulation ----------------------------------------------------
    equity_curve  = np.ones(n)
    dollar_curve  = np.full(n, capital_initial)
    bar_returns   = np.zeros(n)

    trade_log   = []
    open_trade  = None   # dict: entry_bar, entry_price, direction
    capital     = capital_initial

    for i in range(1, n):
        cur_pos = pos_arr[i]
        price   = price_arr[i]
        prev_p  = price_arr[i - 1]

        # -- manage open trade -----------------------------------------------
        if open_trade is not None:
            d          = open_trade["direction"]
            ep         = open_trade["entry_price"]
            bars_held  = i - open_trade["entry_bar"]
            exit_reason = None

            # TP / SL checks
            if tp_dist and d == 1  and price >= ep + tp_dist:
                exit_reason = "TP"
            elif tp_dist and d == -1 and price <= ep - tp_dist:
                exit_reason = "TP"
            elif sl_dist and d == 1  and price <= ep - sl_dist:
                exit_reason = "SL"
            elif sl_dist and d == -1 and price >= ep + sl_dist:
                exit_reason = "SL"
            elif max_hold_bars and bars_held >= max_hold_bars:
                exit_reason = "MAX_HOLD"
            elif cur_pos != d and cur_pos != 0:
                exit_reason = "SIGNAL_FLIP"
            elif cur_pos == 0 and signal_arr[i] == 0:
                # keep riding until new signal or forced exit
                pass

            if exit_reason:
                raw_ret = d * (price - ep) / ep
                net_ret = raw_ret - spread / ep
                pnl_dollars = net_ret * capital

                capital += pnl_dollars
                bar_returns[i] = net_ret
                equity_curve[i] = capital / capital_initial
                dollar_curve[i] = capital

                trade_log.append({
                    "entry_bar":   open_trade["entry_bar"],
                    "exit_bar":    i,
                    "bars_held":   bars_held,
                    "direction":   d,
                    "entry_price": round(ep, 6),
                    "exit_price":  round(price, 6),
                    "pnl_pct":     round(net_ret * 100, 6),
                    "pnl_dollars": round(pnl_dollars, 4),
                    "exit_reason": exit_reason,
                })
                open_trade = None
            else:
                # mark-to-market
                raw_ret = d * (price - prev_p) / prev_p
                bar_returns[i] = raw_ret
                equity_curve[i] = capital / capital_initial
                dollar_curve[i] = capital

        else:
            equity_curve[i] = equity_curve[i - 1]
            dollar_curve[i] = dollar_curve[i - 1]

        # -- open new trade --------------------------------------------------
        if open_trade is None and cur_pos != 0:
            open_trade = {
                "entry_bar":   i,
                "entry_price": price,
                "direction":   cur_pos,
            }

    # close any remaining open trade at last bar
    if open_trade is not None:
        i   = n - 1
        d   = open_trade["direction"]
        ep  = open_trade["entry_price"]
        price = price_arr[i]
        bars_held = i - open_trade["entry_bar"]
        raw_ret = d * (price - ep) / ep
        net_ret = raw_ret - spread / ep
        pnl_dollars = net_ret * capital
        capital += pnl_dollars
        trade_log.append({
            "entry_bar":   open_trade["entry_bar"],
            "exit_bar":    i,
            "bars_held":   bars_held,
            "direction":   d,
            "entry_price": round(ep, 6),
            "exit_price":  round(price, 6),
            "pnl_pct":     round(net_ret * 100, 6),
            "pnl_dollars": round(pnl_dollars, 4),
            "exit_reason": "END_OF_DATA",
        })

    # -- compute metrics -----------------------------------------------------
    eq        = equity_curve
    ret_arr   = bar_returns

    total_return = float(eq[-1] - 1.0)
    mdd          = _max_drawdown(eq)

    # BUG FIX: use full ret_arr (all bars), NOT nonzero-only.
    # Using only non-zero bars shrinks std artificially -> inflated Sharpe/Sortino.
    net_sh    = _sharpe(ret_arr)
    sortino_v = _sortino(ret_arr)

    # gross_sharpe: only positive-return bars (unchanged — measures raw upside)
    gross_sh  = _sharpe(ret_arr[ret_arr > 0]) if (ret_arr > 0).any() else 0.0

    calmar_v  = (total_return / abs(mdd)) if mdd < 0 else 0.0

    wins  = [t for t in trade_log if t["pnl_pct"] >= 0]
    losses = [t for t in trade_log if t["pnl_pct"] < 0]
    n_trades = len(trade_log)
    win_rate  = len(wins) / n_trades if n_trades else 0.0

    gross_profit = sum(t["pnl_pct"] for t in wins)
    gross_loss   = abs(sum(t["pnl_pct"] for t in losses))
    pf = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
    pf = min(pf, 999.0)  # cap to avoid JSON inf issues

    avg_bars = float(np.mean([t["bars_held"] for t in trade_log])) if trade_log else 0.0
    turnover = float((pos_arr != 0).mean())

    sig_counts = signals.value_counts().to_dict()
    signal_dist = {
        "Long":  int(sig_counts.get(1, 0)),
        "Short": int(sig_counts.get(-1, 0)),
        "Flat":  int(sig_counts.get(0, 0)),
    }

    # BUG FIX: window raised from 21 to 390 (1 trading day at 1-min resolution)
    roll_sh = _rolling_sharpe(ret_arr, window=390)

    ts_list = list(timestamps) if timestamps else []

    return BacktestResult(
        pair            = pair,
        strategy        = strategy,
        split           = split,
        mode            = mode,
        direction_mode  = direction_mode,
        fold_index      = fold_index,
        equity          = [round(float(v), 8) for v in eq],
        equity_dollars  = [round(float(v), 4) for v in dollar_curve],
        rolling_sharpe  = [round(float(v), 6) for v in roll_sh],
        timestamps      = ts_list,
        signal_dist     = signal_dist,
        net_sharpe      = round(net_sh,      6),
        gross_sharpe    = round(gross_sh,    6),
        total_return    = round(total_return, 8),
        max_drawdown    = round(mdd,          8),
        calmar          = round(calmar_v,     6),
        win_rate        = round(win_rate,     6),
        profit_factor   = round(pf,           6),
        avg_trade_bars  = round(avg_bars,     4),
        turnover        = round(turnover,     6),
        sortino         = round(sortino_v,    6),
        n_trades        = n_trades,
        capital_initial = capital_initial,
        capital_final   = round(capital,      4),
        trade_log       = trade_log,
    )


# ---------------------------------------------------------------------------
# Walk-forward cross-validation
# ---------------------------------------------------------------------------

def run_wf_folds(
    pair:           str,
    strategy:       str,
    n_folds:        int          = 5,
    split:          str          = "train",
    spread_pips:    Optional[float] = None,
    tp_pips:        Optional[float] = None,
    sl_pips:        Optional[float] = None,
    capital_initial: float       = 10_000.0,
    max_hold_bars:  Optional[int]   = None,
    session:        Optional[str]   = None,
    entry_time:     Optional[str]   = None,
    resample:       Optional[str]   = None,
    direction_mode: str          = "long_short",
    mode:           str          = "research",
) -> list[BacktestResult]:
    """
    Walk-forward cross-validation over the train split.

    Loads datasets/train/{pair}_train.parquet, divides it into
    n_folds equal windows, and runs a backtest on each window.

    Returns a list of BacktestResult (one per fold).
    """
    path = _DATASETS / "train" / f"{pair}_train.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Train parquet not found: {path}\n"
            f"Run scripts/split_fx_data.py first."
        )

    df = pd.read_parquet(path)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    if resample:
        df = _resample_df(df, resample)

    fold_size = len(df) // n_folds
    results   = []

    strat = get_strategy(strategy)

    for k in range(n_folds):
        start = k * fold_size
        end   = start + fold_size if k < n_folds - 1 else len(df)
        fold_df = df.iloc[start:end].reset_index(drop=True)

        signals = strat.generate_signals(fold_df).reset_index(drop=True)
        prices  = fold_df["close"].reset_index(drop=True)
        timestamps = (
            fold_df["timestamp_utc"]
            .dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            .tolist()
        )

        r = run_backtest(
            signals         = signals,
            prices          = prices,
            pair            = pair,
            strategy        = strat.name,
            split           = f"train_fold{k+1}",
            spread_pips     = spread_pips,
            tp_pips         = tp_pips,
            sl_pips         = sl_pips,
            capital_initial = capital_initial,
            max_hold_bars   = max_hold_bars,
            timestamps      = timestamps,
            session         = session,
            entry_time      = entry_time,
            direction_mode  = direction_mode,
            mode            = mode,
            fold_index      = k + 1,
            df_full         = fold_df,
        )
        results.append(r)

    return results

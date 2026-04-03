from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent

@dataclass
class BacktestResult:
    pair:           str
    strategy:       str
    split:          str
    equity:         List[float]
    rolling_sharpe: List[float]
    signal_dist:    dict
    net_sharpe:     float
    gross_sharpe:   float
    total_return:   float
    max_drawdown:   float
    calmar:         float
    win_rate:       float
    profit_factor:  float
    avg_trade_bars: float
    turnover:       float
    sortino:        float
    n_trades:       int
    spread_pips:    float
    fold_index:     Optional[int] = None




PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD"]

PIP_SIZE: dict[str, float] = {
    "EURUSD": 0.0001,
    "GBPUSD": 0.0001,
    "USDJPY": 0.01,
    "USDCHF": 0.0001,
    "USDCAD": 0.0001,
    "AUDUSD": 0.0001,
    "NZDUSD": 0.0001,
}

SPREAD_TABLE: dict[str, float] = {
    "EURUSD": 1.0,
    "GBPUSD": 1.2,
    "USDJPY": 1.5,
    "USDCHF": 1.5,
    "USDCAD": 1.8,
    "AUDUSD": 1.4,
    "NZDUSD": 1.8,
}

TRADING_DAYS = 252
BARS_PER_DAY = 1440  # 1-minute data


def _validate_inputs(signals: pd.Series, prices: pd.Series, pair: str) -> None:
    if pair not in PAIRS:
        raise ValueError(f"Unknown pair '{pair}'. Supported: {PAIRS}")
    if len(signals) != len(prices):
        raise ValueError("signals and prices must have equal length.")
    if signals.isna().any() or prices.isna().any():
        raise ValueError("signals and prices must not contain NaN.")


def _compute_metrics(
    net_returns:   pd.Series,
    gross_returns: pd.Series,
    positions:     pd.Series,
    prices:        pd.Series,
    pair:          str,
    spread_pips:   float,
    tp_pips:       Optional[float],
    sl_pips:       Optional[float],
) -> dict:
    """Compute all 10 canonical metrics from return series."""

    ann_factor = np.sqrt(BARS_PER_DAY * TRADING_DAYS)

    def sharpe(r: pd.Series) -> float:
        std = r.std()
        if std == 0 or len(r) < 2:
            return 0.0
        return float((r.mean() / std) * ann_factor)

    net_sharpe   = sharpe(net_returns)
    gross_sharpe = sharpe(gross_returns)

    equity       = (1 + net_returns).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)

    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    max_dd   = float(drawdown.min())

    n_bars     = len(net_returns)
    ann_return = float((1 + total_return) ** (BARS_PER_DAY * TRADING_DAYS / n_bars) - 1) if n_bars > 0 else 0.0
    calmar     = ann_return / abs(max_dd) if max_dd != 0 else 0.0

    downside     = net_returns[net_returns < 0]
    downside_std = downside.std()
    sortino      = float((net_returns.mean() / downside_std) * ann_factor) if downside_std > 0 else 0.0

    # - Trade-level stats (fixed) ----------------------------------------
    # Build a list of (entry_bar, exit_bar, direction) by scanning
    # position changes directly — avoids index.get_loc misuse.
    pos_arr  = positions.values
    px_arr   = prices.values
    pip      = PIP_SIZE[pair]

    wins, losses, durations = [], [], []
    i = 0
    n = len(pos_arr)

    while i < n:
        if pos_arr[i] == 0:
            i += 1
            continue

        # Found a trade entry
        direction  = pos_arr[i]
        entry_px   = px_arr[i]
        entry_bar  = i
        tp_price   = (entry_px + direction * tp_pips * pip) if tp_pips else None
        sl_price   = (entry_px - direction * sl_pips * pip) if sl_pips else None

        # Walk forward until exit condition
        j = i + 1
        exit_bar = n - 1
        while j < n:
            # Position flipped or went flat
            if pos_arr[j] != direction:
                exit_bar = j - 1
                break
            # TP hit
            if tp_price is not None:
                if direction == 1 and px_arr[j] >= tp_price:
                    exit_bar = j
                    break
                if direction == -1 and px_arr[j] <= tp_price:
                    exit_bar = j
                    break
            # SL hit
            if sl_price is not None:
                if direction == 1 and px_arr[j] <= sl_price:
                    exit_bar = j
                    break
                if direction == -1 and px_arr[j] >= sl_price:
                    exit_bar = j
                    break
            j += 1

        trade_ret  = net_returns.iloc[entry_bar:exit_bar + 1].sum()
        duration   = exit_bar - entry_bar + 1
        durations.append(duration)

        if trade_ret > 0:
            wins.append(trade_ret)
        else:
            losses.append(abs(trade_ret))

        # Skip to after this trade
        i = exit_bar + 1

    n_trades       = len(wins) + len(losses)
    win_rate       = len(wins) / n_trades if n_trades > 0 else 0.0
    profit_factor  = sum(wins) / sum(losses) if losses else (float("inf") if wins else 0.0)
    avg_trade_bars = float(np.mean(durations)) if durations else 0.0

    signal_changes = int((positions.diff().fillna(0) != 0).sum())
    turnover       = signal_changes / n_bars if n_bars > 0 else 0.0

    return {
        "net_sharpe":     net_sharpe,
        "gross_sharpe":   gross_sharpe,
        "total_return":   total_return,
        "max_drawdown":   max_dd,
        "calmar":         calmar,
        "win_rate":       win_rate,
        "profit_factor":  profit_factor,
        "avg_trade_bars": avg_trade_bars,
        "turnover":       turnover,
        "sortino":        sortino,
        "n_trades":       n_trades,
        "equity":         equity.tolist(),
    }


def run_backtest(
    signals:       pd.Series,
    prices:        pd.Series,
    pair:          str,
    strategy:      str,
    split:         str = "val",
    spread_pips:   Optional[float] = None,
    tp_pips:       Optional[float] = None,
    sl_pips:       Optional[float] = None,
    position_size: float = 1.0,
    fold_index:    Optional[int] = None,
) -> BacktestResult:
    """
    Run a single backtest for one pair/strategy combination.

    signals:      integer series  1 long / -1 short / 0 flat
    prices:       close price series aligned to signals
    spread_pips:  override SPREAD_TABLE default if provided
    tp_pips:      take-profit distance in pips (optional)
    sl_pips:      stop-loss distance in pips (optional)
    position_size: notional multiplier (default 1.0)
    """
    _validate_inputs(signals, prices, pair)

    spread = spread_pips if spread_pips is not None else SPREAD_TABLE[pair]
    pip    = PIP_SIZE[pair]

    # Shift signals by 1 bar — trade executes on next bar open (leakage guard)
    pos = signals.shift(1).fillna(0).astype(float) * position_size

    log_ret       = np.log(prices / prices.shift(1)).fillna(0.0)
    gross_returns = pos * log_ret

    # Transaction cost on every position change (entry + exit each costs half-spread)
    pos_diff        = pos.diff().fillna(pos.iloc[0]).abs()
    cost_per_bar    = pos_diff * spread * pip
    net_returns     = gross_returns - cost_per_bar

    metrics = _compute_metrics(
        net_returns, gross_returns, pos, prices, pair, spread, tp_pips, sl_pips
    )

    # Rolling 21-bar Sharpe (1-minute annualisation)
    roll_std    = net_returns.rolling(21).std().fillna(0)
    roll_mean   = net_returns.rolling(21).mean().fillna(0)
    rolling_sh  = (roll_mean / roll_std.replace(0, np.nan) * np.sqrt(BARS_PER_DAY * TRADING_DAYS)).fillna(0)

    sig_counts  = signals.value_counts().to_dict()
    signal_dist = {
        "Long":  int(sig_counts.get(1,  0)),
        "Short": int(sig_counts.get(-1, 0)),
        "Flat":  int(sig_counts.get(0,  0)),
    }

    return BacktestResult(
        pair=pair,
        strategy=strategy,
        split=split,
        fold_index=fold_index,
        equity=metrics["equity"],
        rolling_sharpe=rolling_sh.tolist(),
        signal_dist=signal_dist,
        net_sharpe=metrics["net_sharpe"],
        gross_sharpe=metrics["gross_sharpe"],
        total_return=metrics["total_return"],
        max_drawdown=metrics["max_drawdown"],
        calmar=metrics["calmar"],
        win_rate=metrics["win_rate"],
        profit_factor=metrics["profit_factor"],
        avg_trade_bars=metrics["avg_trade_bars"],
        turnover=metrics["turnover"],
        sortino=metrics["sortino"],
        n_trades=metrics["n_trades"],
        spread_pips=spread,
    )


def run_cv_folds(
    signals_df:  pd.DataFrame,
    prices_df:   pd.DataFrame,
    pair:        str,
    strategy:    str,
    split:       str = "val",
    n_folds:     int = 5,
    spread_pips: Optional[float] = None,
    tp_pips:     Optional[float] = None,
    sl_pips:     Optional[float] = None,
) -> list[BacktestResult]:
    """
    Walk-forward cross-validation. Each fold is a contiguous equal-length slice.
    Returns one BacktestResult per fold.
    """
    n         = len(signals_df)
    fold_size = n // n_folds
    results   = []

    for i in range(n_folds):
        start      = i * fold_size
        end        = start + fold_size if i < n_folds - 1 else n
        sig_fold   = signals_df.iloc[start:end].squeeze()
        px_fold    = prices_df.iloc[start:end].squeeze()
        r = run_backtest(
            sig_fold, px_fold, pair, strategy, split,
            spread_pips=spread_pips,
            tp_pips=tp_pips,
            sl_pips=sl_pips,
            fold_index=i,
        )
        results.append(r)

    return results

if __name__ == "__main__":
    import random
    random.seed(42)
    np.random.seed(42)

    n  = 5000
    px = pd.Series(np.exp(np.random.randn(n).cumsum() * 0.001 + 7.0))
    # Proper crossover-style signals — mostly flat, occasional trades
    sig = pd.Series(np.zeros(n, dtype=int))
    sig.iloc[100:300]  =  1
    sig.iloc[300:310]  =  0
    sig.iloc[310:600]  = -1
    sig.iloc[600:610]  =  0
    sig.iloc[610:900]  =  1

    result = run_backtest(sig, px, pair="EURUSD", strategy="SmokeTest", split="val")

    print(f"Pair:          {result.pair}")
    print(f"Net Sharpe:    {result.net_sharpe:.4f}")
    print(f"Total Return:  {result.total_return:.4f}")
    print(f"Max Drawdown:  {result.max_drawdown:.4f}")
    print(f"Calmar:        {result.calmar:.4f}")
    print(f"Win Rate:      {result.win_rate:.4f}")
    print(f"Profit Factor: {result.profit_factor:.4f}")
    print(f"Sortino:       {result.sortino:.4f}")
    print(f"N Trades:      {result.n_trades}")
    print(f"Turnover:      {result.turnover:.4f}")
    print("Smoke test passed.")


    # for commit

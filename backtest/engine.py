from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

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
BARS_PER_DAY = 1440


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

        direction  = pos_arr[i]
        entry_px   = px_arr[i]
        entry_bar  = i
        tp_price   = (entry_px + direction * tp_pips * pip) if tp_pips else None
        sl_price   = (entry_px - direction * sl_pips * pip) if sl_pips else None

        j = i + 1
        exit_bar = n - 1
        while j < n:
            if pos_arr[j] != direction:
                exit_bar = j - 1
                break
            if tp_price is not None:
                if direction == 1 and px_arr[j] >= tp_price:
                    exit_bar = j
                    break
                if direction == -1 and px_arr[j] <= tp_price:
                    exit_bar = j
                    break
            if sl_price is not None:
                if direction == 1 and px_arr[j] <= sl_price:
                    exit_bar = j
                    break
                if direction == -1 and px_arr[j] >= sl_price:
                    exit_bar = j
                    break
            j += 1

        trade_ret = float(
            np.exp(net_returns.iloc[entry_bar:exit_bar + 1].sum()) - 1.0
        )
        duration  = exit_bar - entry_bar + 1
        durations.append(duration)

        if trade_ret > 0:
            wins.append(trade_ret)
        else:
            losses.append(abs(trade_ret))

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
    _validate_inputs(signals, prices, pair)

    spread = spread_pips if spread_pips is not None else SPREAD_TABLE[pair]
    pip    = PIP_SIZE[pair]

    pos = signals.replace(0, np.nan).ffill().fillna(0).shift(1).fillna(0).astype(float) * position_size

    log_ret       = np.log(prices / prices.shift(1)).fillna(0.0)
    gross_returns = pos * log_ret

    pos_change       = pos.diff().abs()
    pos_change.iloc[0] = abs(pos.iloc[0])
    cost_fraction    = (spread * pip) / prices
    cost_per_bar     = pos_change * cost_fraction
    net_returns      = gross_returns - cost_per_bar

    metrics = _compute_metrics(
        net_returns, gross_returns, pos, prices, pair, spread, tp_pips, sl_pips
    )

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


def run_wf_folds(
    pair:        str,
    strategy:    str,
    n_folds:     int = 5,
    spread_pips: Optional[float] = None,
    tp_pips:     Optional[float] = None,
    sl_pips:     Optional[float] = None,
) -> list[BacktestResult]:
    from backtest.strategies import get_strategy
    strat   = get_strategy(strategy)
    results = []

    for fold_idx in range(n_folds):
        fold_dir  = PROJECT_DIR / "datasets" / "folds" / f"fold_{fold_idx}"
        val_path  = fold_dir / f"{pair}_val.parquet"
        if not val_path.exists():
            raise FileNotFoundError(
                f"Fold {fold_idx} val parquet missing: {val_path}\n"
                f"Run scripts/split_fx_data.py first."
            )
        val_df = pd.read_parquet(val_path)
        val_df["timestamp_utc"] = pd.to_datetime(val_df["timestamp_utc"], utc=True)
        val_df = val_df.sort_values("timestamp_utc").reset_index(drop=True)

        signals = strat.generate_signals(val_df)
        signals = signals.reset_index(drop=True)
        prices  = val_df["close"].reset_index(drop=True)

        r = run_backtest(
            signals=signals,
            prices=prices,
            pair=pair,
            strategy=strategy,
            split=f"fold_{fold_idx}",
            spread_pips=spread_pips,
            tp_pips=tp_pips,
            sl_pips=sl_pips,
            fold_index=fold_idx,
        )
        results.append(r)

    return results


if __name__ == "__main__":
    import random
    random.seed(42)
    np.random.seed(42)

    n  = 5000
    px = pd.Series(np.exp(np.random.randn(n).cumsum() * 0.001 + 7.0))
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

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseStrategy(ABC):
    """Abstract base — all strategies must implement generate_signals."""

    name: str = ""

    @abstractmethod
    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        """
        Accepts a cleaned OHLCV DataFrame, returns an integer signal Series.
        Values: 1 (long), -1 (short), 0 (flat). Same index as prices.

        IMPORTANT: signals are crossover events, not continuous positions.
        The engine converts these to held positions internally via forward-fill
        in run_backtest (shift + hold). Do NOT forward-fill here.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"


class MACrossover(BaseStrategy):
    """
    Dual moving average crossover.

    Signal fires ONLY on the bar where the fast MA crosses the slow MA.
    Between crossovers the signal is 0 (flat) — the engine holds the
    last position via the shift-and-hold mechanism in run_backtest.

    Long  signal (+1) when fast MA crosses UP through slow MA.
    Short signal (-1) when fast MA crosses DOWN through slow MA.
    Flat  signal ( 0) on all other bars and during warm-up.

    Parameters
    ----------
    fast    : look-back for fast MA (default 20).
    slow    : look-back for slow MA (default 50).
    ma_type : 'ema' (default) or 'sma'.
    """

    name = "MACrossover"

    def __init__(self, fast: int = 20, slow: int = 50, ma_type: str = "ema") -> None:
        if fast >= slow:
            raise ValueError(f"fast ({fast}) must be less than slow ({slow}).")
        if ma_type not in ("ema", "sma"):
            raise ValueError("ma_type must be 'ema' or 'sma'.")
        self.fast    = fast
        self.slow    = slow
        self.ma_type = ma_type
        self.name    = f"MACrossover_f{fast}_s{slow}_{ma_type.upper()}"

    def _ma(self, series: pd.Series, period: int) -> pd.Series:
        if self.ma_type == "ema":
            return series.ewm(span=period, adjust=False).mean()
        return series.rolling(period).mean()

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        close   = prices["close"]
        fast_ma = self._ma(close, self.fast)
        slow_ma = self._ma(close, self.slow)

        # Boolean: is fast above slow on this bar?
        fast_above = fast_ma > slow_ma

        # Crossover = state CHANGED from previous bar
        crossed_up   = fast_above & ~fast_above.shift(1).fillna(False)
        crossed_down = ~fast_above & fast_above.shift(1).fillna(True)

        signals = pd.Series(0, index=prices.index, dtype=int)
        signals[crossed_up]   =  1
        signals[crossed_down] = -1

        # Zero out warm-up period — MAs are not valid before slow-1 bars
        signals.iloc[: self.slow - 1] = 0

        return signals


class MomentumStrategy(BaseStrategy):
    """
    Simple price momentum.

    Long  signal (+1) when close is above its n-bar high set at look-back bars ago.
    Short signal (-1) when close is below its n-bar low set at look-back bars ago.
    Flat  signal ( 0) during warm-up and when neither condition is met.

    Parameters
    ----------
    lookback : bars to look back for high/low comparison (default 60).
    """

    name = "Momentum"

    def __init__(self, lookback: int = 60) -> None:
        if lookback < 2:
            raise ValueError("lookback must be >= 2.")
        self.lookback = lookback
        self.name     = f"Momentum_lb{lookback}"

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        close        = prices["close"]
        rolling_high = close.shift(1).rolling(self.lookback).max()
        rolling_low  = close.shift(1).rolling(self.lookback).min()

        broke_up   = (close > rolling_high) & close.shift(1).le(rolling_high)
        broke_down = (close < rolling_low)  & close.shift(1).ge(rolling_low)

        signals = pd.Series(0, index=prices.index, dtype=int)
        signals[broke_up]   =  1
        signals[broke_down] = -1

        signals.iloc[: self.lookback] = 0

        return signals


# - Registry -----------------------------------------------------------------

STRATEGY_REGISTRY: dict[str, BaseStrategy] = {
    "MACrossover_f20_s50_EMA": MACrossover(fast=20, slow=50, ma_type="ema"),
    "MACrossover_f10_s30_EMA": MACrossover(fast=10, slow=30, ma_type="ema"),
    "MACrossover_f20_s50_SMA": MACrossover(fast=20, slow=50, ma_type="sma"),
    "Momentum_lb60":           MomentumStrategy(lookback=60),
    "Momentum_lb120":          MomentumStrategy(lookback=120),
}


def get_strategy(name: str) -> BaseStrategy:
    """Return a pre-instantiated strategy by registry name."""
    if name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy '{name}'.\n"
            f"Available: {list(STRATEGY_REGISTRY.keys())}"
        )
    return STRATEGY_REGISTRY[name]


# - Smoke test ---------------------------------------------------------------

if __name__ == "__main__":
    n = 2000
    np.random.seed(0)
    close  = pd.Series(
        np.exp(np.random.randn(n).cumsum() * 0.001 + 7.0), name="close"
    )
    prices = pd.DataFrame({"close": close})

    print("Testing MACrossover strategies...")
    for key, strat in STRATEGY_REGISTRY.items():
        sigs = strat.generate_signals(prices)

        assert sigs.isin([-1, 0, 1]).all(), f"{key}: invalid signal values"
        assert len(sigs) == n,              f"{key}: length mismatch"

        counts      = sigs.value_counts().to_dict()
        n_crossovers = counts.get(1, 0) + counts.get(-1, 0)
        flat_count   = counts.get(0, 0)
        flat_pct     = flat_count / n * 100

        print(
            f"  {key:<35} "
            f"Long: {counts.get(1,0):>4}  "
            f"Short: {counts.get(-1,0):>4}  "
            f"Flat: {flat_count:>5} ({flat_pct:.1f}%)  "
            f"Crossovers: {n_crossovers}"
        )

        # A crossover strategy should be flat on most bars
        assert flat_pct > 90, (
            f"{key}: only {flat_pct:.1f}% flat bars — "
            f"likely outputting positions not crossovers"
        )

    print()
    print("All strategy smoke tests passed.")
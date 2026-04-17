from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseStrategy(ABC):

    name: str = ""

    @abstractmethod
    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"


class MACrossover(BaseStrategy):
    """Dual moving average crossover."""

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
        close      = prices["close"]
        fast_ma    = self._ma(close, self.fast)
        slow_ma    = self._ma(close, self.slow)
        fast_above = fast_ma > slow_ma

        crossed_up   = fast_above  & ~fast_above.shift(1).fillna(False)
        crossed_down = ~fast_above &  fast_above.shift(1).fillna(True)

        signals = pd.Series(0, index=prices.index, dtype=int)
        signals[crossed_up]   =  1
        signals[crossed_down] = -1
        signals.iloc[: self.slow - 1] = 0
        return signals


class MomentumStrategy(BaseStrategy):
    """Price momentum via rolling high/low breakout."""

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


class DonchianBreakout(BaseStrategy):
    """Donchian channel breakout."""

    name = "Donchian"

    def __init__(self, period: int = 20) -> None:
        if period < 2:
            raise ValueError("period must be >= 2.")
        self.period = period
        self.name   = f"Donchian_p{period}"

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        high = prices["high"] if "high" in prices.columns else prices["close"]
        low  = prices["low"]  if "low"  in prices.columns else prices["close"]

        upper = high.shift(1).rolling(self.period).max()
        lower = low.shift(1).rolling(self.period).min()

        broke_up   = prices["close"] > upper
        broke_down = prices["close"] < lower

        prev_broke_up   = broke_up.shift(1).fillna(False)
        prev_broke_down = broke_down.shift(1).fillna(False)

        signals = pd.Series(0, index=prices.index, dtype=int)
        signals[broke_up   & ~prev_broke_up]   =  1
        signals[broke_down & ~prev_broke_down] = -1
        signals.iloc[: self.period] = 0
        return signals


class RSIMeanReversion(BaseStrategy):
    """RSI threshold crossover for mean reversion."""

    name = "RSI"

    def __init__(
        self,
        period:     int   = 14,
        oversold:   float = 30.0,
        overbought: float = 70.0,
    ) -> None:
        if period < 2:
            raise ValueError("period must be >= 2.")
        if oversold >= overbought:
            raise ValueError("oversold must be less than overbought.")
        self.period     = period
        self.oversold   = oversold
        self.overbought = overbought
        self.name       = f"RSI_p{period}_os{int(oversold)}_ob{int(overbought)}"

    def _rsi(self, close: pd.Series) -> pd.Series:
        delta    = close.diff()
        gain     = delta.clip(lower=0)
        loss     = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1.0 / self.period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / self.period, adjust=False).mean()
        rs       = avg_gain / avg_loss.replace(0, np.nan)
        rsi      = 100.0 - (100.0 / (1.0 + rs))
        return rsi.fillna(50.0)

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        rsi = self._rsi(prices["close"])

        was_oversold   = rsi.shift(1) <= self.oversold
        was_overbought = rsi.shift(1) >= self.overbought
        now_above_os   = rsi > self.oversold
        now_below_ob   = rsi < self.overbought

        crossed_up   = was_oversold   & now_above_os
        crossed_down = was_overbought & now_below_ob

        signals = pd.Series(0, index=prices.index, dtype=int)
        signals[crossed_up]   =  1
        signals[crossed_down] = -1
        signals.iloc[: self.period] = 0
        return signals


class BollingerBreakout(BaseStrategy):
    """Bollinger Band volatility breakout."""

    name = "BB"

    def __init__(self, period: int = 20, std_dev: float = 2.0) -> None:
        if period < 2:
            raise ValueError("period must be >= 2.")
        if std_dev <= 0:
            raise ValueError("std_dev must be positive.")
        self.period  = period
        self.std_dev = std_dev
        self.name    = f"BB_p{period}_std{str(std_dev).replace('.', '_')}"

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        close = prices["close"]
        mid   = close.rolling(self.period).mean()
        sigma = close.rolling(self.period).std()
        upper = mid + self.std_dev * sigma
        lower = mid - self.std_dev * sigma

        above_upper = close > upper
        below_lower = close < lower

        broke_up   = above_upper & ~above_upper.shift(1).fillna(False)
        broke_down = below_lower & ~below_lower.shift(1).fillna(False)

        signals = pd.Series(0, index=prices.index, dtype=int)
        signals[broke_up]   =  1
        signals[broke_down] = -1
        signals.iloc[: self.period] = 0
        return signals


class MACDSignalCross(BaseStrategy):
    """MACD line crosses signal line."""

    name = "MACD"

    def __init__(
        self,
        fast:          int = 12,
        slow:          int = 26,
        signal_period: int = 9,
    ) -> None:
        if fast >= slow:
            raise ValueError(f"fast ({fast}) must be less than slow ({slow}).")
        if signal_period < 1:
            raise ValueError("signal_period must be >= 1.")
        self.fast          = fast
        self.slow          = slow
        self.signal_period = signal_period
        self.name          = f"MACD_f{fast}_s{slow}_sig{signal_period}"

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        close       = prices["close"]
        ema_fast    = close.ewm(span=self.fast,   adjust=False).mean()
        ema_slow    = close.ewm(span=self.slow,   adjust=False).mean()
        macd_line   = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()

        macd_above = macd_line > signal_line

        crossed_up   = macd_above  & ~macd_above.shift(1).fillna(False)
        crossed_down = ~macd_above &  macd_above.shift(1).fillna(True)

        warmup = self.slow + self.signal_period - 1

        signals = pd.Series(0, index=prices.index, dtype=int)
        signals[crossed_up]   =  1
        signals[crossed_down] = -1
        signals.iloc[: warmup] = 0
        return signals


# 13 strategies total
STRATEGY_REGISTRY: dict[str, BaseStrategy] = {
    # MA Crossover (3)
    "MACrossover_f20_s50_EMA": MACrossover(fast=20, slow=50, ma_type="ema"),
    "MACrossover_f10_s30_EMA": MACrossover(fast=10, slow=30, ma_type="ema"),
    "MACrossover_f20_s50_SMA": MACrossover(fast=20, slow=50, ma_type="sma"),

    # Momentum (2)
    "Momentum_lb60":           MomentumStrategy(lookback=60),
    "Momentum_lb120":          MomentumStrategy(lookback=120),

    # Donchian channel breakout (2)
    "Donchian_p20":            DonchianBreakout(period=20),
    "Donchian_p55":            DonchianBreakout(period=55),

    # RSI mean reversion (2)
    "RSI_p14_os30_ob70":       RSIMeanReversion(period=14, oversold=30, overbought=70),
    "RSI_p7_os25_ob75":        RSIMeanReversion(period=7,  oversold=25, overbought=75),

    # Bollinger Band breakout (2)
    "BB_p20_std2_0":           BollingerBreakout(period=20, std_dev=2.0),
    "BB_p14_std1_5":           BollingerBreakout(period=14, std_dev=1.5),

    # MACD signal cross (2)
    "MACD_f12_s26_sig9":       MACDSignalCross(fast=12, slow=26, signal_period=9),
    "MACD_f8_s21_sig5":        MACDSignalCross(fast=8,  slow=21, signal_period=5),
}


def get_strategy(name: str) -> BaseStrategy:
    """Return a pre-instantiated strategy by registry name."""
    if name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy '{name}'.\n"
            f"Available: {list(STRATEGY_REGISTRY.keys())}"
        )
    return STRATEGY_REGISTRY[name]


if __name__ == "__main__":
    n = 2000
    np.random.seed(0)

    close  = pd.Series(np.exp(np.random.randn(n).cumsum() * 0.001 + 7.0))
    high   = close * (1 + np.abs(np.random.randn(n)) * 0.0005)
    low    = close * (1 - np.abs(np.random.randn(n)) * 0.0005)
    prices = pd.DataFrame({"close": close, "high": high, "low": low})

    print("Running strategy smoke tests...")

    for key, strat in STRATEGY_REGISTRY.items():
        sigs = strat.generate_signals(prices)

        assert sigs.isin([-1, 0, 1]).all(), f"{key}: invalid signal values"
        assert len(sigs) == n,              f"{key}: length mismatch"
        assert not sigs.isna().any(),       f"{key}: NaN in signals"

        counts   = sigs.value_counts().to_dict()
        n_long   = counts.get(1,  0)
        n_short  = counts.get(-1, 0)
        n_flat   = counts.get(0,  0)
        flat_pct = n_flat / n * 100

        print(
            f"  {key:<35} "
            f"Long: {n_long:>4}  Short: {n_short:>4}  "
            f"Flat: {n_flat:>5} ({flat_pct:.1f}%)  "
            f"Events: {n_long + n_short}"
        )

        assert flat_pct > 80, (
            f"{key}: only {flat_pct:.1f}% flat bars -- "
            f"likely outputting positions not crossover events"
        )

    print("All strategy smoke tests passed.")

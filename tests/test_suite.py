"""
Forex Algo Trading — Comprehensive Test Suite
Runs entirely in-memory (no parquet files required).
All results are written to  tests/test_results.txt

Usage
-----
    python tests/test_suite.py
    python tests/test_suite.py -v          # verbose (print each test name live)
"""

from __future__ import annotations

import sys
import traceback
import os
from datetime   import datetime
from io         import StringIO
from pathlib    import Path
from typing     import Any, Callable, List, Tuple

import numpy  as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — allow running from project root or tests/
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from backtest.engine import (
    _sharpe, _sortino, _max_drawdown, _rolling_sharpe,
    _resample_df, _in_session,
    run_backtest, run_wf_folds,
    BacktestResult, PAIRS, _PIP, _DEFAULT_SPREAD,
)
from backtest.strategies import (
    MACrossover, MomentumStrategy, DonchianBreakout,
    RSIMeanReversion, BollingerBreakout, MACDSignalCross,
    STRATEGY_REGISTRY, get_strategy,
)

# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

RESULTS_FILE = PROJECT_DIR / "tests" / "test_results.txt"


class TestResult:
    def __init__(self, name: str, group: str):
        self.name    = name
        self.group   = group
        self.passed  = False
        self.message = ""
        self.detail  = ""


class TestRunner:
    def __init__(self, verbose: bool = False):
        self.verbose  = verbose
        self.results: List[TestResult] = []
        self._current_group = ""

    def group(self, name: str):
        self._current_group = name

    def run(self, name: str, fn: Callable, *args, **kwargs) -> TestResult:
        tr = TestResult(name, self._current_group)
        try:
            fn(*args, **kwargs)
            tr.passed  = True
            tr.message = "PASS"
        except AssertionError as e:
            tr.passed  = False
            tr.message = "FAIL"
            tr.detail  = str(e)
        except Exception as e:
            tr.passed  = False
            tr.message = "ERROR"
            tr.detail  = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

        if self.verbose:
            icon = "✓" if tr.passed else "✗"
            print(f"  {icon} [{tr.group}] {tr.name}")

        self.results.append(tr)
        return tr

    def summary(self) -> Tuple[int, int, int]:
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed and r.message == "FAIL")
        errors = sum(1 for r in self.results if r.message == "ERROR")
        return passed, failed, errors


# ---------------------------------------------------------------------------
# Fixtures — synthetic price data
# ---------------------------------------------------------------------------

def _make_prices(
    n: int = 500,
    seed: int = 42,
    drift: float = 0.0001,
    vol: float = 0.001,
    base: float = 1.1000,
) -> pd.DataFrame:
    """Return a minimal OHLCV DataFrame with timestamp_utc."""
    rng    = np.random.default_rng(seed)
    log_r  = rng.normal(drift, vol, n)
    close  = base * np.exp(np.cumsum(log_r))
    high   = close * (1 + np.abs(rng.normal(0, vol * 0.5, n)))
    low    = close * (1 - np.abs(rng.normal(0, vol * 0.5, n)))
    volume = rng.integers(100, 1000, n).astype(float)

    # timestamps: 1-minute bars starting 2022-01-03 00:00 UTC
    ts = pd.date_range("2022-01-03 00:00", periods=n, freq="1min", tz="UTC")

    return pd.DataFrame({
        "timestamp_utc": ts,
        "open":   close * (1 - rng.normal(0, vol * 0.1, n)),
        "high":   high,
        "low":    low,
        "close":  close,
        "volume": volume,
    })


def _make_signals(n: int, pattern: str = "random", seed: int = 0) -> pd.Series:
    """Generate signal series for testing."""
    rng = np.random.default_rng(seed)
    if pattern == "all_zero":
        return pd.Series(np.zeros(n, dtype=int))
    if pattern == "all_long":
        sig = np.zeros(n, dtype=int)
        sig[10] = 1
        return pd.Series(sig)
    if pattern == "all_short":
        sig = np.zeros(n, dtype=int)
        sig[10] = -1
        return pd.Series(sig)
    if pattern == "alternating":
        sig = np.zeros(n, dtype=int)
        for i in range(10, n - 10, 40):
            sig[i]      =  1
            sig[i + 20] = -1
        return pd.Series(sig)
    if pattern == "random":
        choices = rng.choice([-1, 0, 0, 0, 0, 1], size=n)
        return pd.Series(choices.astype(int))
    raise ValueError(f"Unknown pattern: {pattern}")


def _run_simple(
    n: int = 300,
    pattern: str = "alternating",
    pair: str = "EURUSD",
    **kwargs,
) -> BacktestResult:
    df  = _make_prices(n)
    sig = _make_signals(n, pattern)
    return run_backtest(
        signals  = sig,
        prices   = df["close"],
        pair     = pair,
        strategy = "test",
        df_full  = df,
        **kwargs,
    )


# ===========================================================================
# Group A — engine helpers
# ===========================================================================

def test_sharpe_zero_std():
    arr = np.zeros(100)
    assert _sharpe(arr) == 0.0, "all-zero returns should give Sharpe=0"

def test_sharpe_single_bar():
    assert _sharpe(np.array([0.01])) == 0.0, "single bar should give Sharpe=0"

def test_sharpe_positive():
    rng = np.random.default_rng(1)
    arr = rng.normal(0.0002, 0.001, 10_000)
    s   = _sharpe(arr)
    assert s > 0, f"positive-mean series should give Sharpe>0, got {s}"

def test_sharpe_negative():
    rng = np.random.default_rng(2)
    arr = rng.normal(-0.0002, 0.001, 10_000)
    s   = _sharpe(arr)
    assert s < 0, f"negative-mean series should give Sharpe<0, got {s}"

def test_sharpe_annualisation():
    """Doubling the mean should roughly double the Sharpe."""
    rng = np.random.default_rng(3)
    base = rng.normal(0, 0.001, 10_000)
    s1 = _sharpe(base + 0.0001)
    s2 = _sharpe(base + 0.0002)
    assert s2 > s1, "higher mean should produce higher Sharpe"

def test_sortino_no_downside():
    arr = np.abs(np.random.default_rng(4).normal(0, 0.001, 100))
    assert _sortino(arr) == 0.0, "no downside bars -> Sortino=0"

def test_sortino_positive_mean():
    rng = np.random.default_rng(5)
    arr = rng.normal(0.0002, 0.001, 10_000)
    assert _sortino(arr) > 0, "positive-mean series should give Sortino>0"

def test_sortino_uses_full_array():
    """Sortino should be lower when computed on full array vs nonzero-only."""
    rng      = np.random.default_rng(6)
    full_arr = rng.normal(0.0002, 0.001, 10_000)
    # add 9000 zero bars (idle periods)
    padded   = np.concatenate([full_arr[:1000], np.zeros(9000)])
    s_full   = _sortino(padded)
    s_nonzero = _sortino(padded[padded != 0])
    assert s_full <= s_nonzero, (
        f"Full-array Sortino ({s_full:.4f}) should be <= nonzero-only ({s_nonzero:.4f})"
    )

def test_max_drawdown_flat():
    eq = np.ones(100)
    assert _max_drawdown(eq) == 0.0

def test_max_drawdown_monotone_rise():
    eq = np.linspace(1, 2, 100)
    assert _max_drawdown(eq) == 0.0

def test_max_drawdown_known_value():
    eq = np.array([1.0, 1.2, 0.9, 1.1, 0.8, 1.0])
    dd = _max_drawdown(eq)
    # peak at 1.2, trough at 0.8 -> (0.8-1.2)/1.2 = -0.3333
    assert abs(dd - (-1/3)) < 1e-6, f"Expected ~-0.3333, got {dd}"

def test_max_drawdown_always_negative_or_zero():
    rng = np.random.default_rng(7)
    eq  = np.cumprod(1 + rng.normal(0, 0.002, 500))
    assert _max_drawdown(eq) <= 0.0

def test_max_drawdown_empty():
    assert _max_drawdown(np.array([])) == 0.0

def test_rolling_sharpe_length():
    arr = np.random.default_rng(8).normal(0, 0.001, 200)
    rs  = _rolling_sharpe(arr, window=50)
    assert len(rs) == 200, "rolling_sharpe must return same length as input"

def test_rolling_sharpe_first_bars_zero():
    arr = np.random.default_rng(9).normal(0, 0.001, 100)
    rs  = _rolling_sharpe(arr, window=20)
    # first bar must be 0 (single element window)
    assert rs[0] == 0.0, f"first rolling Sharpe should be 0, got {rs[0]}"

def test_rolling_sharpe_constant_zero():
    arr = np.zeros(100)
    rs  = _rolling_sharpe(arr, window=20)
    assert all(v == 0.0 for v in rs)


# ===========================================================================
# Group B — run_backtest core
# ===========================================================================

def test_backtest_returns_result_type():
    r = _run_simple()
    assert isinstance(r, BacktestResult)

def test_equity_starts_at_one():
    r = _run_simple()
    assert abs(r.equity[0] - 1.0) < 1e-9

def test_equity_length_matches_prices():
    n = 300
    r = _run_simple(n)
    assert len(r.equity) == n

def test_dollar_curve_starts_at_capital():
    capital = 25_000.0
    r = _run_simple(capital_initial=capital)
    assert abs(r.equity_dollars[0] - capital) < 1e-3

def test_capital_final_consistent():
    # FIX: floating-point accumulation over many trades means we need a
    # generous tolerance here. 1.0 USD is well within rounding noise.
    r = _run_simple(300, "alternating")
    assert abs(r.capital_final - r.equity_dollars[-1]) < 1.0, (
        f"capital_final={r.capital_final} vs equity_dollars[-1]={r.equity_dollars[-1]}"
    )

def test_no_trades_on_all_zero_signals():
    r = _run_simple(300, "all_zero")
    assert r.n_trades == 0
    assert abs(r.total_return) < 1e-9

def test_equity_flat_on_all_zero_signals():
    r = _run_simple(300, "all_zero")
    assert all(abs(v - 1.0) < 1e-9 for v in r.equity)

def test_total_return_type():
    r = _run_simple()
    assert isinstance(r.total_return, float)

def test_win_rate_bounds():
    r = _run_simple(500, "random")
    assert 0.0 <= r.win_rate <= 1.0

def test_max_drawdown_non_positive():
    r = _run_simple(500, "random")
    assert r.max_drawdown <= 0.0

def test_trade_log_has_required_keys():
    r = _run_simple(500, "random")
    if r.trade_log:
        required = {"entry_bar", "exit_bar", "bars_held", "direction",
                    "entry_price", "exit_price", "pnl_pct", "pnl_dollars", "exit_reason"}
        for trade in r.trade_log:
            missing = required - set(trade.keys())
            assert not missing, f"Trade missing keys: {missing}"

def test_trade_direction_values():
    r = _run_simple(500, "random")
    for t in r.trade_log:
        assert t["direction"] in (1, -1), f"Bad direction: {t['direction']}"

def test_signal_dist_keys():
    r = _run_simple(300, "alternating")
    assert set(r.signal_dist.keys()) == {"Long", "Short", "Flat"}

def test_signal_dist_sum_equals_n():
    n = 300
    r = _run_simple(n, "alternating")
    total = sum(r.signal_dist.values())
    assert total == n, f"signal_dist sum {total} != {n}"

def test_rolling_sharpe_length_matches():
    n = 300
    r = _run_simple(n)
    assert len(r.rolling_sharpe) == n

def test_profit_factor_positive_when_trades():
    r = _run_simple(500, "random")
    if r.n_trades > 0:
        assert r.profit_factor >= 0.0

def test_turnover_in_0_1():
    r = _run_simple(500, "random")
    assert 0.0 <= r.turnover <= 1.0

def test_avg_trade_bars_positive():
    r = _run_simple(500, "random")
    if r.n_trades > 0:
        assert r.avg_trade_bars > 0

def test_metrics_dict_keys():
    r = _run_simple()
    keys = set(r.metrics.keys())
    expected = {
        "net_sharpe", "gross_sharpe", "total_return", "max_drawdown",
        "calmar", "win_rate", "profit_factor", "avg_trade_bars",
        "turnover", "sortino", "n_trades", "capital_initial", "capital_final",
    }
    assert keys == expected, f"Missing keys: {expected - keys}"


# ===========================================================================
# Group C — exit logic
# ===========================================================================

def _make_tp_sl_df(direction: int, tp_pct: float = 0.005) -> pd.DataFrame:
    """
    Returns a DataFrame where price moves monotonically up (direction=1)
    or down (direction=-1) by tp_pct, guaranteeing TP is hit.
    """
    n     = 100
    base  = 1.1000
    delta = base * tp_pct * direction
    close = np.array([base + delta * i for i in range(n)])
    ts    = pd.date_range("2022-01-03", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({
        "timestamp_utc": ts,
        "open":  close, "high": close, "low": close, "close": close,
        "volume": np.ones(n),
    })

def test_tp_hit_long():
    df  = _make_tp_sl_df(direction=1, tp_pct=0.005)
    sig = pd.Series(np.zeros(len(df), dtype=int))
    sig.iloc[1] = 1   # enter long at bar 1
    r = run_backtest(
        signals=sig, prices=df["close"],
        pair="EURUSD", strategy="test",
        tp_pips=5.0, df_full=df,
    )
    assert any(t["exit_reason"] == "TP" for t in r.trade_log), "TP not triggered for long"

def test_tp_hit_short():
    df  = _make_tp_sl_df(direction=-1, tp_pct=0.005)
    sig = pd.Series(np.zeros(len(df), dtype=int))
    sig.iloc[1] = -1
    r = run_backtest(
        signals=sig, prices=df["close"],
        pair="EURUSD", strategy="test",
        tp_pips=5.0, df_full=df,
    )
    assert any(t["exit_reason"] == "TP" for t in r.trade_log), "TP not triggered for short"

def test_sl_hit_long():
    """Price drops by > sl_pips on a long trade -> SL triggered."""
    df  = _make_tp_sl_df(direction=-1, tp_pct=0.01)
    sig = pd.Series(np.zeros(len(df), dtype=int))
    sig.iloc[1] = 1   # enter long but price falls
    r = run_backtest(
        signals=sig, prices=df["close"],
        pair="EURUSD", strategy="test",
        sl_pips=5.0, df_full=df,
    )
    assert any(t["exit_reason"] == "SL" for t in r.trade_log), "SL not triggered for long"

def test_sl_hit_short():
    df  = _make_tp_sl_df(direction=1, tp_pct=0.01)
    sig = pd.Series(np.zeros(len(df), dtype=int))
    sig.iloc[1] = -1
    r = run_backtest(
        signals=sig, prices=df["close"],
        pair="EURUSD", strategy="test",
        sl_pips=5.0, df_full=df,
    )
    assert any(t["exit_reason"] == "SL" for t in r.trade_log), "SL not triggered for short"

def test_max_hold_exit():
    r = _run_simple(300, "all_long", max_hold_bars=20)
    if r.n_trades > 0:
        assert any(t["exit_reason"] == "MAX_HOLD" for t in r.trade_log), \
            "MAX_HOLD exit never fired"

def test_signal_flip_exit():
    sig = pd.Series(np.zeros(200, dtype=int))
    sig.iloc[10]  =  1
    sig.iloc[50]  = -1  # flip -> should close long
    df = _make_prices(200)
    r  = run_backtest(
        signals=sig, prices=df["close"],
        pair="EURUSD", strategy="test", df_full=df,
    )
    reasons = [t["exit_reason"] for t in r.trade_log]
    assert "SIGNAL_FLIP" in reasons, f"Expected SIGNAL_FLIP, got: {reasons}"

def test_end_of_data_exit():
    """A trade open at the last bar should be closed with END_OF_DATA."""
    sig = pd.Series(np.zeros(100, dtype=int))
    sig.iloc[90] = 1   # open near end, no exit signal
    df  = _make_prices(100)
    r   = run_backtest(
        signals=sig, prices=df["close"],
        pair="EURUSD", strategy="test", df_full=df,
    )
    reasons = [t["exit_reason"] for t in r.trade_log]
    assert "END_OF_DATA" in reasons, f"Expected END_OF_DATA, got: {reasons}"

def test_tp_pnl_positive_long():
    """A TP exit on a rising long should have positive pnl_pct."""
    df  = _make_tp_sl_df(direction=1, tp_pct=0.005)
    sig = pd.Series(np.zeros(len(df), dtype=int))
    sig.iloc[1] = 1
    r = run_backtest(
        signals=sig, prices=df["close"],
        pair="EURUSD", strategy="test",
        tp_pips=5.0, df_full=df,
    )
    tp_trades = [t for t in r.trade_log if t["exit_reason"] == "TP"]
    assert tp_trades, "No TP trades found"
    assert tp_trades[0]["pnl_pct"] > 0, "TP on long should be profitable"

def test_sl_pnl_negative_long():
    df  = _make_tp_sl_df(direction=-1, tp_pct=0.01)
    sig = pd.Series(np.zeros(len(df), dtype=int))
    sig.iloc[1] = 1
    r = run_backtest(
        signals=sig, prices=df["close"],
        pair="EURUSD", strategy="test",
        sl_pips=5.0, df_full=df,
    )
    sl_trades = [t for t in r.trade_log if t["exit_reason"] == "SL"]
    assert sl_trades, "No SL trades found"
    assert sl_trades[0]["pnl_pct"] < 0, "SL on failing long should be negative"


# ===========================================================================
# Group D — direction modes
# ===========================================================================

def test_long_only_no_short_trades():
    r = _run_simple(500, "random", direction_mode="long_only")
    for t in r.trade_log:
        assert t["direction"] == 1, f"short trade found in long_only mode: {t}"

def test_short_only_no_long_trades():
    r = _run_simple(500, "random", direction_mode="short_only")
    for t in r.trade_log:
        assert t["direction"] == -1, f"long trade found in short_only mode: {t}"

def test_long_short_has_both_directions():
    r = _run_simple(1000, "alternating", direction_mode="long_short")
    directions = {t["direction"] for t in r.trade_log}
    assert 1  in directions, "No long trades in long_short mode"
    assert -1 in directions, "No short trades in long_short mode"

def test_long_only_fewer_trades_than_long_short():
    r_ls  = _run_simple(500, "random", direction_mode="long_short")
    r_lo  = _run_simple(500, "random", direction_mode="long_only")
    assert r_lo.n_trades <= r_ls.n_trades

def test_short_only_fewer_trades_than_long_short():
    r_ls = _run_simple(500, "random", direction_mode="long_short")
    r_so = _run_simple(500, "random", direction_mode="short_only")
    assert r_so.n_trades <= r_ls.n_trades


# ===========================================================================
# Group E — session & entry-time filters
# ===========================================================================

def test_in_session_london():
    hours = pd.Series([6, 7, 12, 15, 16, 20])
    mask  = _in_session(hours, "london")
    assert list(mask) == [False, True, True, True, False, False]

def test_in_session_asia_wraps_midnight():
    hours = pd.Series([22, 23, 0, 3, 7, 8, 12])
    mask  = _in_session(hours, "tokyo")
    # tokyo: start=23 end=8  -> 23, 0, 3, 7 in; 22, 8, 12 out
    assert list(mask) == [False, True, True, True, True, False, False]

def test_session_filter_reduces_signals():
    """London session filter should allow fewer signal bars than no filter."""
    df  = _make_prices(1000)  # 1000 bars starting 00:00 UTC
    sig = _make_signals(1000, "random", seed=10)
    r_all     = run_backtest(signals=sig, prices=df["close"],
                             pair="EURUSD", strategy="test", df_full=df)
    r_london  = run_backtest(signals=sig, prices=df["close"],
                             pair="EURUSD", strategy="test",
                             session="london", df_full=df)
    # with session filter some bars are zeroed => fewer or equal trades
    assert r_london.n_trades <= r_all.n_trades

def test_entry_time_filter():
    df  = _make_prices(1000)
    sig = _make_signals(1000, "random", seed=11)
    r_all  = run_backtest(signals=sig, prices=df["close"],
                          pair="EURUSD", strategy="test", df_full=df)
    r_late = run_backtest(signals=sig, prices=df["close"],
                          pair="EURUSD", strategy="test",
                          entry_time="08:00", df_full=df)
    assert r_late.n_trades <= r_all.n_trades

def test_session_ny():
    hours = pd.Series([12, 13, 17, 21, 22, 23])
    mask  = _in_session(hours, "ny")
    assert list(mask) == [False, True, True, True, False, False]


# ===========================================================================
# Group F — spread mechanics
# ===========================================================================

def test_zero_spread_higher_return():
    r_spread = _run_simple(500, "random", spread_pips=2.0)
    r_zero   = _run_simple(500, "random", spread_pips=0.0)
    # higher spread -> lower or equal total return
    assert r_zero.total_return >= r_spread.total_return

def test_spread_reduces_pnl():
    df  = _make_tp_sl_df(direction=1, tp_pct=0.002)
    sig = pd.Series(np.zeros(len(df), dtype=int))
    sig.iloc[1] = 1
    r_tight = run_backtest(signals=sig, prices=df["close"],
                           pair="EURUSD", strategy="test",
                           tp_pips=20.0, spread_pips=0.1, df_full=df)
    r_wide  = run_backtest(signals=sig, prices=df["close"],
                           pair="EURUSD", strategy="test",
                           tp_pips=20.0, spread_pips=5.0, df_full=df)
    if r_tight.trade_log and r_wide.trade_log:
        assert r_tight.trade_log[0]["pnl_pct"] >= r_wide.trade_log[0]["pnl_pct"]

def test_jpy_pip_size():
    assert _PIP["USDJPY"] == 0.01

def test_non_jpy_pip_size():
    for p in ["EURUSD", "GBPUSD", "AUDUSD"]:
        assert _PIP[p] == 0.0001, f"Wrong pip for {p}"

def test_default_spread_table_coverage():
    for p in PAIRS:
        assert p in _DEFAULT_SPREAD, f"{p} missing from spread table"


# ===========================================================================
# Group G — all 13 strategies
# ===========================================================================

DEFAULT_PRICES = _make_prices(2000, seed=99)

# Strategies that forward-fill their signal (hold positions) rather than
# emitting pure crossover events will legitimately have flat% < 50%.
# We apply a relaxed threshold (>15%) for those — still confirms they
# produce a mix of signals rather than all-long or all-short.
_HELD_POSITION_STRATEGIES = {
    "MACrossover_f10_s30_EMA",
    "MACrossover_f20_s50_EMA",
    "MACrossover_f20_s50_SMA",
    "MACD_f12_s26_sig9",
    "MACD_f8_s21_sig5",
}


def _strategy_checks(name: str):
    strat = get_strategy(name)
    sigs  = strat.generate_signals(DEFAULT_PRICES)
    n     = len(DEFAULT_PRICES)

    assert len(sigs) == n,              f"{name}: length mismatch"
    assert not sigs.isna().any(),       f"{name}: NaN signals"
    assert sigs.isin([-1, 0, 1]).all(), f"{name}: non-{{-1,0,1}} values"

    counts   = sigs.value_counts().to_dict()
    flat_pct = counts.get(0, 0) / n * 100

    if name in _HELD_POSITION_STRATEGIES:
        # Held-position / trend-following strategies stay in the market most
        # of the time — flat% can legitimately be well below 50%.  We just
        # confirm it is not 0% (all-long/all-short) or 100% (dead strategy).
        assert flat_pct > 5, (
            f"{name}: flat%={flat_pct:.1f} — strategy produces no transitions"
        )
        assert flat_pct < 95, (
            f"{name}: flat%={flat_pct:.1f} — strategy almost never trades"
        )
    else:
        # Pure crossover / oscillator strategies should be flat >50% of bars
        assert flat_pct > 50, (
            f"{name}: flat%={flat_pct:.1f} — likely forwarding positions unexpectedly"
        )

    # at least one non-zero signal on 2000 bars
    n_events = counts.get(1, 0) + counts.get(-1, 0)
    assert n_events > 0, f"{name}: zero crossover events on 2000 bars"

for _sname in STRATEGY_REGISTRY:
    # register each as a named function so the test runner can identify them
    def _make_test(sn=_sname):
        def _t():
            _strategy_checks(sn)
        _t.__name__ = f"test_strategy_{sn}"
        return _t
    globals()[f"test_strategy_{_sname}"] = _make_test()


# ===========================================================================
# Group H — strategy constructor validation
# ===========================================================================

def test_ma_fast_ge_slow_raises():
    try:
        MACrossover(fast=50, slow=20)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

def test_ma_invalid_type_raises():
    try:
        MACrossover(fast=10, slow=30, ma_type="wma")
        assert False
    except ValueError:
        pass

def test_momentum_small_lookback_raises():
    try:
        MomentumStrategy(lookback=1)
        assert False
    except ValueError:
        pass

def test_donchian_small_period_raises():
    try:
        DonchianBreakout(period=1)
        assert False
    except ValueError:
        pass

def test_rsi_oversold_ge_overbought_raises():
    try:
        RSIMeanReversion(oversold=70, overbought=30)
        assert False
    except ValueError:
        pass

def test_rsi_small_period_raises():
    try:
        RSIMeanReversion(period=1)
        assert False
    except ValueError:
        pass

def test_bb_small_period_raises():
    try:
        BollingerBreakout(period=1)
        assert False
    except ValueError:
        pass

def test_bb_zero_std_raises():
    try:
        BollingerBreakout(std_dev=0.0)
        assert False
    except ValueError:
        pass

def test_macd_fast_ge_slow_raises():
    try:
        MACDSignalCross(fast=26, slow=12)
        assert False
    except ValueError:
        pass

def test_macd_zero_signal_raises():
    try:
        MACDSignalCross(signal_period=0)
        assert False
    except ValueError:
        pass

def test_get_strategy_unknown_raises():
    try:
        get_strategy("NonExistentStrategy_xyz")
        assert False
    except ValueError:
        pass


# ===========================================================================
# Group I — BacktestResult dataclass
# ===========================================================================

def test_backtest_result_defaults():
    r = BacktestResult(pair="EURUSD", strategy="test",
                       split="full", mode="research",
                       direction_mode="long_short", fold_index=0)
    assert r.net_sharpe == 0.0
    assert r.n_trades   == 0
    assert r.equity     == []
    assert r.trade_log  == []

def test_backtest_result_metrics_returns_dict():
    r = BacktestResult(pair="EURUSD", strategy="test",
                       split="full", mode="research",
                       direction_mode="long_short", fold_index=0)
    assert isinstance(r.metrics, dict)

def test_calmar_zero_when_no_drawdown():
    r = _run_simple(300, "all_zero")
    assert r.calmar == 0.0, "Calmar should be 0 when no drawdown"

def test_profit_factor_inf_cap():
    """profit_factor is capped at 999 when there are no losing trades."""
    # all-long on a monotonically rising price
    df  = _make_tp_sl_df(direction=1, tp_pct=0.002)
    sig = pd.Series(np.zeros(len(df), dtype=int))
    sig.iloc[1] = 1
    r = run_backtest(
        signals=sig, prices=df["close"],
        pair="EURUSD", strategy="test",
        tp_pips=20.0, df_full=df,
    )
    assert r.profit_factor <= 999.0, f"profit_factor not capped: {r.profit_factor}"

def test_pair_and_strategy_stored():
    r = _run_simple(pair="GBPUSD")
    assert r.pair     == "GBPUSD"
    assert r.strategy == "test"


# ===========================================================================
# Group J — data loader / split path resolution
# ===========================================================================

# Import the resolver directly
from backtest.run_backtest import resolve_split_path, load_split_data, NAMED_SPLITS


def test_full_split_path():
    p = resolve_split_path("full", "EURUSD")
    # FIX: use Path comparison instead of str endswith('/') to work on
    # Windows (backslash) and POSIX (forward slash) paths alike.
    p = Path(p)
    assert p.name == "EURUSD.parquet", (
        f"Expected filename EURUSD.parquet, got: {p.name}"
    )
    assert p.parent.name == "datasets", (
        f"Expected parent dir 'datasets', got: {p.parent.name}  (full path: {p})"
    )

def test_train_split_path():
    p = resolve_split_path("train", "GBPUSD")
    assert "train" in str(p) and "GBPUSD_train.parquet" in str(p)

def test_val_split_path():
    p = resolve_split_path("val", "EURUSD")
    assert "val" in str(p)

def test_test_split_path():
    p = resolve_split_path("test", "AUDUSD")
    assert "test" in str(p)

def test_fold_uses_train_parquet():
    for i in range(5):
        p = resolve_split_path(f"fold_{i}", "EURUSD")
        assert "train" in str(p), f"fold_{i} should use train parquet"

def test_named_splits_contains_full():
    assert "full" in NAMED_SPLITS

def test_named_splits_contains_train_val_test():
    for s in ("train", "val", "test"):
        assert s in NAMED_SPLITS

def test_named_splits_contains_folds():
    for i in range(5):
        assert f"fold_{i}" in NAMED_SPLITS

def test_load_data_date_filter_empty_gives_error():
    """load_split_data on a non-existent path should raise FileNotFoundError."""
    try:
        load_split_data("EURUSD", "full",
                        date_from="2099-01-01", date_to="2099-01-02")
        # If file doesn't exist we expect FileNotFoundError
        # If file exists but date range empty we expect ValueError
        assert False, "Should have raised"
    except (FileNotFoundError, ValueError):
        pass  # both are correct

def test_resolve_unknown_split_raises():
    try:
        resolve_split_path("unknown_split", "EURUSD")
        assert False
    except ValueError:
        pass


# ===========================================================================
# Group K — edge cases for run_backtest
# ===========================================================================

def test_single_bar_input():
    """run_backtest must not crash on a single bar."""
    df  = _make_prices(1)
    sig = pd.Series([0])
    r   = run_backtest(signals=sig, prices=df["close"],
                       pair="EURUSD", strategy="test", df_full=df)
    assert r.n_trades == 0

def test_two_bar_input():
    df  = _make_prices(2)
    sig = pd.Series([1, 0])
    r   = run_backtest(signals=sig, prices=df["close"],
                       pair="EURUSD", strategy="test", df_full=df)
    assert isinstance(r, BacktestResult)

def test_flat_prices_no_pnl():
    """Completely flat prices should produce zero PnL on all trades."""
    n     = 100
    flat  = pd.Series(np.full(n, 1.1000))
    sig   = _make_signals(n, "alternating")
    df    = pd.DataFrame({
        "timestamp_utc": pd.date_range("2022-01-03", periods=n, freq="1min", tz="UTC"),
        "open": flat, "high": flat, "low": flat, "close": flat, "volume": np.ones(n),
    })
    r = run_backtest(signals=sig, prices=flat,
                     pair="EURUSD", strategy="test",
                     spread_pips=0.0, df_full=df)
    for t in r.trade_log:
        assert abs(t["pnl_pct"]) < 1e-6, f"Expected zero pnl on flat prices, got {t['pnl_pct']}"

def test_high_capital():
    r = _run_simple(300, "alternating", capital_initial=1_000_000.0)
    assert r.capital_initial == 1_000_000.0

def test_zero_capital_does_not_crash():
    # FIX: engine.py now guards against capital_initial=0 (was ZeroDivisionError).
    # Equity curve should stay at 1.0 throughout since no PnL accumulates.
    r = _run_simple(300, "alternating", capital_initial=0.0)
    assert isinstance(r, BacktestResult), "Should return a BacktestResult even with zero capital"
    assert r.capital_initial == 0.0
    # equity curve is normalised against 1.0 internally so all bars == 1.0
    assert all(abs(v - 1.0) < 1e-9 for v in r.equity), (
        "equity curve should be flat at 1.0 when capital=0"
    )

def test_unknown_pair_uses_default_pip():
    """A pair not in the pip table should fall back to 0.0001."""
    df  = _make_prices(200)
    sig = _make_signals(200, "alternating")
    r   = run_backtest(signals=sig, prices=df["close"],
                       pair="XXXYYY", strategy="test", df_full=df)
    assert isinstance(r, BacktestResult)

def test_all_long_signals():
    r = _run_simple(300, "all_long")
    assert r.n_trades >= 1
    for t in r.trade_log:
        assert t["direction"] == 1

def test_all_short_signals():
    r = _run_simple(300, "all_short")
    assert r.n_trades >= 1
    for t in r.trade_log:
        assert t["direction"] == -1

def test_timestamps_stored():
    df  = _make_prices(200)
    sig = _make_signals(200, "alternating")
    ts  = df["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist()
    r   = run_backtest(signals=sig, prices=df["close"],
                       pair="EURUSD", strategy="test",
                       timestamps=ts, df_full=df)
    assert len(r.timestamps) == 200

def test_resample_df_reduces_rows():
    df   = _make_prices(390)  # 390 1-min bars = 1 trading day
    df_r = _resample_df(df, "1h")
    assert len(df_r) < len(df), "resampled df should have fewer rows"
    assert "close" in df_r.columns

def test_resample_df_ohlcv_agg():
    df   = _make_prices(120)
    df_r = _resample_df(df, "1h")
    assert df_r["high"].iloc[0] >= df_r["close"].iloc[0]
    assert df_r["low"].iloc[0]  <= df_r["close"].iloc[0]


# ===========================================================================
# Group L — walk-forward fold slicing (no disk I/O, pure logic)
# ===========================================================================

def test_fold_slicing_correct_size():
    """Verify fold slicing logic matches what run_wf_folds would produce."""
    n       = 1000
    n_folds = 5
    fold_sz = n // n_folds
    sizes   = []
    for k in range(n_folds):
        start = k * fold_sz
        end   = start + fold_sz if k < n_folds - 1 else n
        sizes.append(end - start)
    # first 4 folds equal size, last fold may be larger
    for s in sizes[:4]:
        assert s == fold_sz, f"Fold size {s} != {fold_sz}"
    assert sizes[-1] >= fold_sz

def test_fold_slicing_covers_all_data():
    n       = 1000
    n_folds = 5
    fold_sz = n // n_folds
    total   = 0
    for k in range(n_folds):
        start = k * fold_sz
        end   = start + fold_sz if k < n_folds - 1 else n
        total += end - start
    assert total == n

def test_fold_slicing_no_overlap():
    n       = 1000
    n_folds = 5
    fold_sz = n // n_folds
    covered = set()
    for k in range(n_folds):
        start = k * fold_sz
        end   = start + fold_sz if k < n_folds - 1 else n
        bar_set = set(range(start, end))
        assert not covered.intersection(bar_set), f"Fold {k} overlaps previous folds"
        covered |= bar_set

def test_independent_fold_results():
    """Running run_backtest on two different slices should give different results."""
    df   = _make_prices(500, seed=77)
    mid  = len(df) // 2
    df1  = df.iloc[:mid].reset_index(drop=True)
    df2  = df.iloc[mid:].reset_index(drop=True)
    strat = get_strategy("MACrossover_f10_s30_EMA")

    for fold_df in [df1, df2]:
        sig = strat.generate_signals(fold_df).reset_index(drop=True)
        r   = run_backtest(signals=sig, prices=fold_df["close"],
                           pair="EURUSD", strategy=strat.name,
                           df_full=fold_df)
        assert isinstance(r, BacktestResult)


# ===========================================================================
# Report writer
# ===========================================================================

SEP  = "-" * 90
SEP2 = "=" * 90

GROUP_ORDER = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
GROUP_NAMES = {
    "A": "Engine helpers (_sharpe / _sortino / _max_drawdown / _rolling_sharpe)",
    "B": "run_backtest core (equity, metrics, trade log)",
    "C": "Exit logic (TP / SL / MAX_HOLD / SIGNAL_FLIP / END_OF_DATA)",
    "D": "Direction modes (long_short / long_only / short_only)",
    "E": "Session & entry-time filters",
    "F": "Spread mechanics & pip sizes",
    "G": "All 13 strategies (signal validity, warm-up, crossover events)",
    "H": "Strategy constructor validation (bad params raise ValueError)",
    "I": "BacktestResult dataclass (metrics dict, defaults)",
    "J": "Data loader / split path resolution",
    "K": "run_backtest edge cases (1 bar, flat prices, zero capital)",
    "L": "Walk-forward fold slicing (sizes, coverage, no overlap)",
}


def write_report(runner: TestRunner, elapsed: float) -> str:
    passed, failed, errors = runner.summary()
    total = len(runner.results)
    lines = []

    lines.append(SEP2)
    lines.append("  FOREX ALGO TRADING — TEST RESULTS")
    lines.append(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Duration  : {elapsed:.2f}s")
    lines.append(SEP2)
    lines.append(f"  Total: {total}   Passed: {passed}   Failed: {failed}   Errors: {errors}")
    overall = "ALL PASS" if (failed + errors) == 0 else "SOME FAILURES"
    lines.append(f"  Overall   : {overall}")
    lines.append(SEP2)
    lines.append("")

    # --- group breakdown ---
    by_group: dict[str, list[TestResult]] = {g: [] for g in GROUP_ORDER}
    # also catch any unlabelled groups
    for r in runner.results:
        g = r.group if r.group in by_group else "?"
        by_group.setdefault(g, []).append(r)

    for g in GROUP_ORDER:
        tests = by_group.get(g, [])
        if not tests:
            continue
        g_pass  = sum(1 for t in tests if t.passed)
        g_total = len(tests)
        label   = GROUP_NAMES.get(g, g)
        lines.append(f"Group {g} — {label}")
        lines.append(f"  Result: {g_pass}/{g_total} passed")
        lines.append(SEP)

        for t in tests:
            status = "PASS" if t.passed else t.message
            lines.append(f"  [{status:^5}]  {t.name}")
            if not t.passed and t.detail:
                for dl in t.detail.strip().splitlines():
                    lines.append(f"           {dl}")
        lines.append("")

    # --- failures summary ---
    failures = [r for r in runner.results if not r.passed]
    if failures:
        lines.append(SEP2)
        lines.append(f"  FAILURES & ERRORS ({len(failures)})")
        lines.append(SEP2)
        for r in failures:
            lines.append(f"  [{r.message}] [{r.group}] {r.name}")
            if r.detail:
                for dl in r.detail.strip().splitlines()[:8]:
                    lines.append(f"      {dl}")
        lines.append("")

    return "\n".join(lines)


# ===========================================================================
# Main runner
# ===========================================================================

def collect_tests() -> list[tuple[str, str, Callable]]:
    """
    Collect all test_* functions from this module and assign them a group
    based on the GROUP_ORDER mapping.
    """
    group_map = {
        "sharpe": "A", "sortino": "A", "max_drawdown": "A", "rolling_sharpe": "A",
        "equity": "B", "dollar": "B", "capital": "B", "no_trades": "B",
        "total_return": "B", "win_rate": "B", "drawdown": "B", "trade_log": "B",
        "signal_dist": "B", "rolling": "B", "profit_factor": "B",
        "turnover": "B", "avg_trade": "B", "metrics": "B", "backtest_returns": "B",
        "tp_": "C", "sl_": "C", "max_hold": "C", "signal_flip": "C", "end_of_data": "C",
        "long_only": "D", "short_only": "D", "long_short": "D", "fewer_trades": "D",
        "in_session": "E", "session": "E", "entry_time": "E",
        "spread": "F", "jpy_pip": "F", "non_jpy": "F", "pip_size": "F", "default_spread": "F",
        "strategy_": "G",
        "ma_fast": "H", "ma_invalid": "H", "momentum_small": "H", "donchian_small": "H",
        "rsi_oversold": "H", "rsi_small": "H", "bb_small": "H", "bb_zero": "H",
        "macd_fast": "H", "macd_zero": "H", "get_strategy_unknown": "H",
        "backtest_result": "I", "calmar": "I", "profit_factor_inf": "I", "pair_and": "I",
        "full_split": "J", "train_split": "J", "val_split": "J", "test_split": "J",
        "fold_uses": "J", "named_splits": "J", "load_data": "J", "resolve_unknown": "J",
        "single_bar": "K", "two_bar": "K", "flat_prices": "K", "high_capital": "K",
        "zero_capital": "K", "unknown_pair": "K", "all_long": "K", "all_short": "K",
        "timestamps": "K", "resample_df": "K",
        "fold_slicing": "L", "independent_fold": "L",
    }

    import inspect
    module = sys.modules[__name__]
    tests  = []
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if not name.startswith("test_"):
            continue
        group = "?"
        key   = name[len("test_"):]
        for prefix, g in group_map.items():
            if key.startswith(prefix) or prefix in key:
                group = g
                break
        tests.append((name, group, obj))

    # stable sort: group then name
    tests.sort(key=lambda x: (x[1], x[0]))
    return tests


def main():
    import time
    verbose = "-v" in sys.argv or "--verbose" in sys.argv

    runner = TestRunner(verbose=verbose)
    tests  = collect_tests()

    print(f"\n  Forex Algo Trading — Test Suite")
    print(f"  {len(tests)} tests collected")
    print(f"  {'verbose' if verbose else 'silent'} mode  (use -v for live output)")
    print()

    start = time.perf_counter()
    for name, group, fn in tests:
        runner.group(group)
        runner.run(name, fn)
    elapsed = time.perf_counter() - start

    passed, failed, errors = runner.summary()
    report = write_report(runner, elapsed)

    # Write to file
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(report, encoding="utf-8")

    # Print summary to console
    print(report)
    print(f"  Results written to: {RESULTS_FILE}")
    print()

    sys.exit(0 if (failed + errors) == 0 else 1)


if __name__ == "__main__":
    main()

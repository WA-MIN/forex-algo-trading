from __future__ import annotations # I adeded a comment over here

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data"
PARQUET_DIR = DATA_DIR / "parquet"

EDA_ROOT_DIR = PROJECT_DIR / "eda"
RAW_SNAPSHOT_DIR = EDA_ROOT_DIR / "raw_snapshot"
SAMPLE_DIR = EDA_ROOT_DIR / "samples"
REPORTS_DIR = EDA_ROOT_DIR / "reports"
GLOBAL_DIR = REPORTS_DIR / "global"

CORE_COLUMNS = [
    "timestamp_est",
    "timestamp_utc",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "pair",
    "year",
    "month",
    "session",
]

DEFAULT_DPI = 600
DEFAULT_SAMPLE_FRAC = 0.01
DEFAULT_ACF_LAGS = 60
EXPECTED_MINUTES_PER_DAY = 1440
SCATTER_SAMPLE_SIZE = 10000
HIST_BINS = 100

EXTREME_JUMP_STD_MULTIPLIER = 10.0
LOW_COVERAGE_THRESHOLD = 0.5

GLOBAL_SCATTER_PAIRS = [
    ("EURUSD", "GBPUSD"),
    ("AUDUSD", "NZDUSD"),
    ("USDJPY", "USDCHF"),
]

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"


@dataclass(frozen=True)
class PairInput:
    """Container for one canonical FX parquet file descriptor."""
    pair: str
    dataset_tag: str
    parquet_path: Path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def save_plot(path: Path, dpi: int) -> None:
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def positive_int(value: str) -> int:
    """Custom argparse type: reject zero and negative integers."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return parsed


def fraction_0_1(value: str) -> float:
    """Custom argparse type: accept floats in (0, 1] only."""
    parsed = float(value)
    if parsed <= 0 or parsed > 1:
        raise argparse.ArgumentTypeError("Value must be in the range (0, 1].")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reproducible EDA for FX canonical parquet datasets."
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=None,
        help="Optional subset of pairs to process, for example EURUSD GBPUSD.",
    )
    parser.add_argument(
        "--dpi",
        type=positive_int,
        default=DEFAULT_DPI,
        help="Plot DPI.",
    )
    parser.add_argument(
        "--sample-frac",
        type=fraction_0_1,
        default=DEFAULT_SAMPLE_FRAC,
        help="Sample fraction for lightweight sample parquet.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute outputs even if they already exist.",
    )
    parser.add_argument(
        "--skip-global",
        action="store_true",
        help="Skip cross-pair global EDA artifacts.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def parse_pair_input(path: Path) -> PairInput:
    """Parse pair code and dataset tag from filename. Expected: PAIR_TAG.parquet"""
    stem = path.stem
    if "_" not in stem:
        raise ValueError(
            f"Parquet filename must follow '<PAIR>_<TAG>.parquet': {path.name}"
        )

    pair, dataset_tag = stem.split("_", 1)
    if not pair or not dataset_tag:
        raise ValueError(
            f"Parquet filename must contain both pair and dataset tag: {path.name}"
        )

    return PairInput(pair=pair.upper(), dataset_tag=dataset_tag, parquet_path=path)


def discover_pair_inputs(
    parquet_dir: Path,
    selected_pairs: list[str] | None = None,
) -> list[PairInput]:
    """Scan parquet_dir for canonical files, optionally filtered by pair list."""
    if not parquet_dir.exists():
        raise FileNotFoundError(f"Missing parquet directory: {parquet_dir}")

    selected = None if selected_pairs is None else {pair.upper() for pair in selected_pairs}

    inputs: list[PairInput] = []
    for path in sorted(parquet_dir.glob("*.parquet")):
        pair_input = parse_pair_input(path)
        if selected is not None and pair_input.pair not in selected:
            continue
        inputs.append(pair_input)

    if not inputs:
        if selected_pairs:
            raise ValueError(
                f"No parquet files found for requested pairs: {', '.join(selected_pairs)}"
            )
        raise ValueError(f"No canonical parquet files found in: {parquet_dir}")

    return sorted(inputs, key=lambda item: item.pair)


def snapshot_path_for(pair_input: PairInput) -> Path:
    return RAW_SNAPSHOT_DIR / f"{pair_input.pair}_{pair_input.dataset_tag}_raw_snapshot.parquet"


def sample_path_for(pair_input: PairInput) -> Path:
    return SAMPLE_DIR / f"{pair_input.pair}_{pair_input.dataset_tag}_sample.parquet"


def load_pair_parquet(pair_input: PairInput) -> pd.DataFrame:
    """Load and validate one canonical pair parquet."""
    path = pair_input.parquet_path
    if not path.exists():
        raise FileNotFoundError(f"Missing canonical parquet: {path}")

    df = pd.read_parquet(path)
    missing = [col for col in CORE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{pair_input.pair}: missing required columns: {missing}")
    if df.empty:
        raise ValueError(f"{pair_input.pair}: parquet is empty")

    df = df.copy()
    df["timestamp_est"] = pd.to_datetime(df["timestamp_est"], errors="coerce")
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    return df


def build_snapshot(pair_input: PairInput, df: pd.DataFrame, sample_frac: float, force: bool) -> Path:
    """Write immutable raw snapshot and a random sample parquet (random_state=42)."""
    snapshot_path = snapshot_path_for(pair_input)
    sample_path = sample_path_for(pair_input)

    if snapshot_path.exists() and sample_path.exists() and not force:
        return snapshot_path

    ensure_dir(RAW_SNAPSHOT_DIR)
    ensure_dir(SAMPLE_DIR)

    df.to_parquet(snapshot_path, index=False)

    sample_df = df.sample(frac=sample_frac, random_state=42) if len(df) else df.copy()
    sample_df = sample_df.sort_values("timestamp_utc").reset_index(drop=True)
    sample_df.to_parquet(sample_path, index=False)

    return snapshot_path


def add_eda_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret_1"] = out["close"].pct_change()

    # mask non-positive close before log to avoid -inf / NaN propagation
    close_positive = out["close"].where(out["close"] > 0)
    out["log_ret_1"] = np.log(close_positive).diff()

    out["range"] = out["high"] - out["low"]
    out["abs_ret_1"] = out["ret_1"].abs()
    out["date"] = out["timestamp_utc"].dt.date
    out["hour"] = out["timestamp_utc"].dt.hour
    out["year_month"] = out["timestamp_utc"].dt.strftime("%Y-%m")
    return out


def compute_overview(df: pd.DataFrame) -> pd.DataFrame:
    invalid_ohlc = (
        (df["high"] < df["low"]) |
        (df["open"] > df["high"]) |
        (df["open"] < df["low"]) |
        (df["close"] > df["high"]) |
        (df["close"] < df["low"])
    )
    return pd.DataFrame([{
        "rows": len(df),
        "start_utc": df["timestamp_utc"].min(),
        "end_utc": df["timestamp_utc"].max(),
        "duplicate_timestamps": int(df["timestamp_utc"].duplicated().sum()),
        "missing_open": int(df["open"].isna().sum()),
        "missing_high": int(df["high"].isna().sum()),
        "missing_low": int(df["low"].isna().sum()),
        "missing_close": int(df["close"].isna().sum()),
        "missing_volume": int(df["volume"].isna().sum()),
        "invalid_ohlc_rows": int(invalid_ohlc.sum()),
        "zero_volume_rows": int((df["volume"] == 0).sum()),
    }])


def compute_counts_by_year_month(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["year", "month"], as_index=False)
        .size()
        .rename(columns={"size": "row_count"})
        .sort_values(["year", "month"])
        .reset_index(drop=True)
    )


def compute_daily_coverage(df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df.groupby("date", as_index=False)
        .size()
        .rename(columns={"size": "observed_minutes"})
    )
    daily["expected_minutes"] = EXPECTED_MINUTES_PER_DAY
    daily["coverage_ratio"] = daily["observed_minutes"] / daily["expected_minutes"]
    return daily


def compute_monthly_coverage(daily_df: pd.DataFrame) -> pd.DataFrame:
    temp = daily_df.copy()
    temp["year_month"] = pd.to_datetime(temp["date"]).dt.strftime("%Y-%m")
    return (
        temp.groupby("year_month", as_index=False)
        .agg(
            observed_minutes=("observed_minutes", "sum"),
            avg_daily_coverage=("coverage_ratio", "mean"),
            min_daily_coverage=("coverage_ratio", "min"),
            max_daily_coverage=("coverage_ratio", "max"),
            num_days=("date", "count"),
        )
        .sort_values("year_month")
        .reset_index(drop=True)
    )


def compute_session_stats(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("session", as_index=False)
        .agg(
            avg_range=("range", "mean"),
            avg_abs_ret_1=("abs_ret_1", "mean"),
            avg_volume=("volume", "mean"),
            median_range=("range", "median"),
            obs=("session", "size"),
        )
        .sort_values("session")
        .reset_index(drop=True)
    )


def compute_hourly_stats(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("hour", as_index=False)
        .agg(
            avg_range=("range", "mean"),
            avg_abs_ret_1=("abs_ret_1", "mean"),
            avg_volume=("volume", "mean"),
            obs=("hour", "size"),
        )
        .sort_values("hour")
        .reset_index(drop=True)
    )


def compute_returns_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ["ret_1", "log_ret_1", "range", "abs_ret_1"]:
        series = df[col].dropna()
        if series.empty:
            continue
        rows.append({
            "metric": col,
            "count": int(series.count()),
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "p01": float(series.quantile(0.01)),
            "p05": float(series.quantile(0.05)),
            "median": float(series.median()),
            "p95": float(series.quantile(0.95)),
            "p99": float(series.quantile(0.99)),
            "max": float(series.max()),
            "skew": float(series.skew()),
            "kurtosis": float(series.kurtosis()),
        })
    return pd.DataFrame(rows)


def simple_acf(series: pd.Series, nlags: int) -> pd.DataFrame:
    """
    Compute sample autocorrelation up to nlags using the biased estimator.
    Normalised by lag-0 autocovariance (np.dot(x, x)).
    Returns a DataFrame with columns [lag, acf].
    """
    x = series.dropna().astype("float64").to_numpy()
    if len(x) < 3:
        return pd.DataFrame({"lag": [], "acf": []})

    x = x - x.mean()
    denom = np.dot(x, x)
    if denom == 0:
        return pd.DataFrame({
            "lag": list(range(nlags + 1)),
            "acf": [np.nan] * (nlags + 1),
        })

    max_lag = min(nlags, len(x) - 1)
    values = []
    for lag in range(max_lag + 1):
        num = np.dot(x[:-lag] if lag > 0 else x, x[lag:])
        values.append(num / denom)

    return pd.DataFrame({"lag": list(range(max_lag + 1)), "acf": values})


def build_quality_summary(df: pd.DataFrame, eda_df: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
    """Summarise key quality flags: duplicates, OHLC violations, extreme jumps, low-coverage days."""
    invalid_ohlc = (
        (df["high"] < df["low"]) |
        (df["open"] > df["high"]) |
        (df["open"] < df["low"]) |
        (df["close"] > df["high"]) |
        (df["close"] < df["low"])
    )
    ret_std = eda_df["ret_1"].std()
    extreme_jumps = (
        0
        if pd.isna(ret_std) or ret_std == 0
        else int((eda_df["ret_1"].abs() > EXTREME_JUMP_STD_MULTIPLIER * ret_std).sum())
    )
    low_coverage_days = int((daily_df["coverage_ratio"] < LOW_COVERAGE_THRESHOLD).sum())

    return pd.DataFrame([{
        "duplicate_timestamps": int(df["timestamp_utc"].duplicated().sum()),
        "missing_core_ohlc_rows": int(df[["open", "high", "low", "close"]].isna().any(axis=1).sum()),
        "invalid_ohlc_rows": int(invalid_ohlc.sum()),
        "extreme_jump_rows": extreme_jumps,
        "low_coverage_days_below_threshold": low_coverage_days,
        "extreme_jump_std_multiplier": EXTREME_JUMP_STD_MULTIPLIER,
        "low_coverage_threshold": LOW_COVERAGE_THRESHOLD,
    }])


def plot_line(x, y, title: str, xlabel: str, ylabel: str, path: Path, dpi: int, rotate_xticks: bool = False) -> None:
    plt.figure(figsize=(14, 6))
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if rotate_xticks:
        plt.xticks(rotation=90)
    plt.grid(True, alpha=0.3)
    save_plot(path, dpi)


def plot_bar(x, y, title: str, xlabel: str, ylabel: str, path: Path, dpi: int) -> None:
    plt.figure(figsize=(10, 6))
    plt.bar(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", alpha=0.3)
    save_plot(path, dpi)


def plot_hist(series: pd.Series, title: str, xlabel: str, path: Path, dpi: int) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(series.dropna(), bins=HIST_BINS)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    save_plot(path, dpi)


def plot_acf_chart(acf_df: pd.DataFrame, title: str, path: Path, dpi: int) -> None:
    plt.figure(figsize=(10, 6))
    plt.bar(acf_df["lag"], acf_df["acf"])
    plt.title(title)
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.grid(True, axis="y", alpha=0.3)
    save_plot(path, dpi)


def plot_corr_matrix(corr_df: pd.DataFrame, title: str, path: Path, dpi: int) -> None:
    plt.figure(figsize=(10, 8))
    im = plt.imshow(corr_df.values, aspect="auto")
    plt.colorbar(im)
    plt.xticks(range(len(corr_df.columns)), corr_df.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr_df.index)), corr_df.index)
    plt.title(title)
    for i in range(len(corr_df.index)):
        for j in range(len(corr_df.columns)):
            plt.text(j, i, f"{corr_df.iloc[i, j]:.2f}", ha="center", va="center")
    save_plot(path, dpi)


def plot_scatter(x: pd.Series, y: pd.Series, title: str, xlabel: str, ylabel: str, path: Path, dpi: int) -> None:
    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, s=5, alpha=0.4)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    save_plot(path, dpi)


def run_pair_eda(pair_input: PairInput, df: pd.DataFrame, dpi: int, force: bool) -> None:
    """Run all per-pair EDA: compute tables, save CSVs, generate plots."""
    report_dir = ensure_dir(REPORTS_DIR / pair_input.pair)

    df = df.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    eda_df = add_eda_columns(df)
    daily_df = compute_daily_coverage(eda_df)
    monthly_df = compute_monthly_coverage(daily_df)

    outputs = {
        "overview":  report_dir / "E1_overview.csv",
        "counts":    report_dir / "E1_counts_by_year_month.csv",
        "daily":     report_dir / "E2_time_coverage_daily.csv",
        "monthly":   report_dir / "E2_time_coverage_monthly.csv",
        "session":   report_dir / "E2_session_stats.csv",
        "hourly":    report_dir / "E2_hourly_stats.csv",
        "returns":   report_dir / "E3_returns_summary.csv",
        "quality":   report_dir / "E4_data_quality_summary.csv",
    }

    tables = {
        outputs["overview"]:  compute_overview(df),
        outputs["counts"]:    compute_counts_by_year_month(df),
        outputs["daily"]:     daily_df,
        outputs["monthly"]:   monthly_df,
        outputs["session"]:   compute_session_stats(eda_df),
        outputs["hourly"]:    compute_hourly_stats(eda_df),
        outputs["returns"]:   compute_returns_summary(eda_df),
        outputs["quality"]:   build_quality_summary(df, eda_df, daily_df),
    }

    for path, table in tables.items():
        if force or not path.exists():
            save_csv(table, path)

    session_df = tables[outputs["session"]]
    hourly_df = tables[outputs["hourly"]]
    monthly_x = pd.to_datetime(monthly_df["year_month"], format="%Y-%m", errors="coerce")

    plot_specs = [
        (
            report_dir / "E2_daily_observations.png",
            lambda: plot_line(
                pd.to_datetime(daily_df["date"]),
                daily_df["observed_minutes"],
                f"{pair_input.pair} daily observation counts",
                "Date", "Observed minutes",
                report_dir / "E2_daily_observations.png", dpi,
            ),
        ),
        (
            report_dir / "E2_monthly_coverage.png",
            lambda: plot_line(
                monthly_x,
                monthly_df["avg_daily_coverage"],
                f"{pair_input.pair} average daily coverage by month",
                "Year-Month", "Average daily coverage ratio",
                report_dir / "E2_monthly_coverage.png", dpi,
                rotate_xticks=True,
            ),
        ),
        (
            report_dir / "E2_session_volatility.png",
            lambda: plot_bar(
                session_df["session"],
                session_df["avg_abs_ret_1"],
                f"{pair_input.pair} average absolute return by session",
                "Session", "Average absolute 1-minute return",
                report_dir / "E2_session_volatility.png", dpi,
            ),
        ),
        (
            report_dir / "E2_hourly_volatility.png",
            lambda: plot_line(
                hourly_df["hour"],
                hourly_df["avg_abs_ret_1"],
                f"{pair_input.pair} intraday volatility profile by hour",
                "Hour of day UTC", "Average absolute 1-minute return",
                report_dir / "E2_hourly_volatility.png", dpi,
            ),
        ),
        (
            report_dir / "E3_returns_hist.png",
            lambda: plot_hist(
                eda_df["ret_1"],
                f"{pair_input.pair} 1-minute returns",
                "ret_1",
                report_dir / "E3_returns_hist.png", dpi,
            ),
        ),
        (
            report_dir / "E3_log_returns_hist.png",
            lambda: plot_hist(
                eda_df["log_ret_1"],
                f"{pair_input.pair} 1-minute log returns",
                "log_ret_1",
                report_dir / "E3_log_returns_hist.png", dpi,
            ),
        ),
        (
            report_dir / "E3_range_hist.png",
            lambda: plot_hist(
                eda_df["range"],
                f"{pair_input.pair} intraminute range",
                "range",
                report_dir / "E3_range_hist.png", dpi,
            ),
        ),
        (
            report_dir / "E3_acf_returns.png",
            lambda: plot_acf_chart(
                simple_acf(eda_df["ret_1"], DEFAULT_ACF_LAGS),
                f"{pair_input.pair} ACF of 1-minute returns",
                report_dir / "E3_acf_returns.png", dpi,
            ),
        ),
        (
            report_dir / "E3_acf_abs_returns.png",
            lambda: plot_acf_chart(
                simple_acf(eda_df["abs_ret_1"], DEFAULT_ACF_LAGS),
                f"{pair_input.pair} ACF of absolute 1-minute returns",
                report_dir / "E3_acf_abs_returns.png", dpi,
            ),
        ),
    ]

    for path, plot_fn in plot_specs:
        if force or not path.exists():
            plot_fn()


def run_global_eda(pair_inputs: list[PairInput], dpi: int, force: bool) -> None:
    """
    Compute cross-pair return and volatility correlations, save CSVs and heatmaps.
    Merges on inner join so only timestamps present in all pairs are used.
    """
    if not pair_inputs:
        logging.warning("Skipping global EDA because no valid pair inputs were processed.")
        return

    ensure_dir(GLOBAL_DIR)
    pair_returns: list[pd.DataFrame] = []
    pair_daily_vol: list[pd.DataFrame] = []

    for pair_input in pair_inputs:
        path = snapshot_path_for(pair_input)
        if not path.exists():
            logging.warning("Skipping global input because snapshot is missing: %s", path)
            continue

        df = pd.read_parquet(path)
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
        df = df.sort_values("timestamp_utc").reset_index(drop=True)
        temp = add_eda_columns(df)

        pair_returns.append(
            temp[["timestamp_utc", "ret_1"]].rename(columns={"ret_1": pair_input.pair})
        )
        pair_daily_vol.append(
            temp.groupby("date", as_index=False)["range"].mean().rename(columns={"range": pair_input.pair})
        )

    if not pair_returns or not pair_daily_vol:
        logging.warning("Skipping global EDA because aligned pair inputs are unavailable.")
        return

    merged_returns = pair_returns[0]
    for frame in pair_returns[1:]:
        merged_returns = merged_returns.merge(frame, on="timestamp_utc", how="inner")

    merged_vol = pair_daily_vol[0]
    for frame in pair_daily_vol[1:]:
        merged_vol = merged_vol.merge(frame, on="date", how="inner")

    if merged_returns.shape[1] <= 1 or merged_vol.shape[1] <= 1:
        logging.warning("Skipping global EDA because there are not enough aligned pairs.")
        return

    ret_corr = merged_returns.drop(columns=["timestamp_utc"]).corr()
    vol_corr = merged_vol.drop(columns=["date"]).corr()

    ret_csv = GLOBAL_DIR / "E4_cross_pair_corr_returns.csv"
    vol_csv = GLOBAL_DIR / "E4_cross_pair_corr_vol.csv"

    if force or not ret_csv.exists():
        ret_corr.to_csv(ret_csv)
    if force or not vol_csv.exists():
        vol_corr.to_csv(vol_csv)

    ret_png = GLOBAL_DIR / "E4_cross_pair_corr_returns.png"
    vol_png = GLOBAL_DIR / "E4_cross_pair_corr_vol.png"

    if force or not ret_png.exists():
        plot_corr_matrix(ret_corr, "Cross-pair return correlation", ret_png, dpi)
    if force or not vol_png.exists():
        plot_corr_matrix(vol_corr, "Cross-pair daily volatility correlation", vol_png, dpi)

    available_pairs = set(ret_corr.columns)
    for left, right in GLOBAL_SCATTER_PAIRS:
        if left not in available_pairs or right not in available_pairs:
            continue

        aligned = merged_returns[[left, right]].dropna()
        if aligned.empty:
            continue

        sample = aligned.sample(n=min(SCATTER_SAMPLE_SIZE, len(aligned)), random_state=42)
        path = GLOBAL_DIR / f"E4_scatter_{left}_vs_{right}.png"
        if force or not path.exists():
            plot_scatter(
                sample[left], sample[right],
                f"{left} vs {right} 1-minute returns",
                left, right, path, dpi,
            )


def main() -> None:
    configure_logging()
    args = parse_args()

    ensure_dir(RAW_SNAPSHOT_DIR)
    ensure_dir(SAMPLE_DIR)
    ensure_dir(REPORTS_DIR)
    ensure_dir(GLOBAL_DIR)

    pair_inputs = discover_pair_inputs(PARQUET_DIR, args.pairs)
    logging.info("Discovered %d canonical parquet input(s).", len(pair_inputs))

    successful_pairs: list[PairInput] = []
    failed_pairs: list[str] = []

    for pair_input in pair_inputs:
        try:
            logging.info("Processing pair: %s", pair_input.pair)
            df = load_pair_parquet(pair_input)
            build_snapshot(pair_input, df, args.sample_frac, args.force)
            run_pair_eda(pair_input, df, args.dpi, args.force)
            successful_pairs.append(pair_input)
            logging.info("Completed pair: %s", pair_input.pair)
        except Exception as exc:
            failed_pairs.append(pair_input.pair)
            logging.exception("Failed pair %s: %s", pair_input.pair, exc)

    if not args.skip_global:
        run_global_eda(successful_pairs, args.dpi, args.force)
    else:
        logging.info("Global EDA skipped by CLI flag.")

    logging.info("EDA complete.")
    logging.info("Successful pairs: %s", ", ".join(p.pair for p in successful_pairs) or "None")
    logging.info("Failed pairs: %s", ", ".join(failed_pairs) or "None")
    logging.info("Snapshots saved in: %s", RAW_SNAPSHOT_DIR)
    logging.info("Reports saved in: %s", REPORTS_DIR)


if __name__ == "__main__":
    main()
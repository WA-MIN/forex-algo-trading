from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent

LABEL_ROOT_DIR   = PROJECT_DIR / "labels"
LABEL_PAIR_DIR   = LABEL_ROOT_DIR / "pair"

SPLIT_ROOT_DIR   = PROJECT_DIR / "datasets"
TRAIN_DIR        = SPLIT_ROOT_DIR / "train"
VAL_DIR          = SPLIT_ROOT_DIR / "val"
TEST_DIR         = SPLIT_ROOT_DIR / "test"
REPORTS_DIR      = SPLIT_ROOT_DIR / "reports"

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD"]

REQUIRED_COLUMNS = [
    "timestamp_utc",
    "pair",
    "session",
    "close",
    "label",
]

# ------------------- Split boundaries -------------------------------
# Train : 2015-01-02 to 2021-12-31   (66.3 % of data - confirmed by EDA)
# Val   : 2022-01-01 to 2023-12-31   (14.8 % of data - confirmed by EDA)
# Test  : 2024-01-01 to 2025-12-31   (18.9 % of data - confirmed by EDA)
# Purge : 15 rows dropped from the tail of train before val starts,
#         and from the tail of val before test starts.
#         Matches the maximum label horizon (horizon_secondary = 15) so
#         forward-return labels at the boundary cannot see into the next split.

DEFAULT_TRAIN_END  = "2021-12-31 23:59:59+00:00"
DEFAULT_VAL_END    = "2023-12-31 23:59:59+00:00"
DEFAULT_PURGE_ROWS = 15


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def parse_timestamp(ts_str: str) -> pd.Timestamp:
    # force UTC regardless of whether the string carries +00:00 or not
    ts = pd.Timestamp(ts_str)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split labeled FX datasets into deterministic train/val/test sets."
    )
    parser.add_argument("--train-end",   type=str, default=DEFAULT_TRAIN_END)
    parser.add_argument("--val-end",     type=str, default=DEFAULT_VAL_END)
    parser.add_argument("--purge-rows",  type=int, default=DEFAULT_PURGE_ROWS,
                        help="Rows to drop from the tail of train and val at each boundary.")
    parser.add_argument("--force",       action="store_true",
                        help="Recompute outputs even if they already exist.")
    return parser.parse_args()


def load_labeled_pair(pair: str) -> pd.DataFrame:
    """Load and validate one labeled pair parquet."""
    path = LABEL_PAIR_DIR / f"{pair}_2015_2025_labeled.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing labeled parquet: {path}")

    df = pd.read_parquet(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{pair}: missing required columns: {missing}")
    if df.empty:
        raise ValueError(f"{pair}: labeled parquet is empty")

    df = df.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    return df


def split_pair(
    df: pd.DataFrame,
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
    purge_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological split with purge gaps at both boundaries.

    Timeline:
      [─── train ───][purge][─── val ───][purge][─── test ───]

    Purge removes the last `purge_rows` rows of train before val begins,
    and the last `purge_rows` rows of val before test begins.
    This prevents forward-return labels at the boundary from seeing
    prices that belong to the next split.
    """
    raw_train = df.loc[df["timestamp_utc"] <= train_end].copy()
    raw_val   = df.loc[(df["timestamp_utc"] > train_end) &
                       (df["timestamp_utc"] <= val_end)].copy()
    test_df   = df.loc[df["timestamp_utc"] > val_end].copy()

    # drop purge tail from train and val
    train_df = raw_train.iloc[: len(raw_train) - purge_rows].copy() if len(raw_train) > purge_rows else raw_train.copy()
    val_df   = raw_val.iloc[: len(raw_val)   - purge_rows].copy() if len(raw_val)   > purge_rows else raw_val.copy()

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def label_distribution(df: pd.DataFrame, prefix: str) -> dict:
    """Compute label counts and class percentages for one split."""
    total = len(df)
    vc = df["label"].value_counts(dropna=True)
    long_n     = int(vc.get(1,  0))
    flat_n     = int(vc.get(0,  0))
    short_n    = int(vc.get(-1, 0))
    na_n       = int(df["label"].isna().sum())
    signal_n   = long_n + short_n

    return {
        f"{prefix}_rows":          total,
        f"{prefix}_long_n":        long_n,
        f"{prefix}_flat_n":        flat_n,
        f"{prefix}_short_n":       short_n,
        f"{prefix}_na_n":          na_n,
        f"{prefix}_signal_pct":    round(signal_n / total * 100, 2) if total else 0.0,
        f"{prefix}_long_pct":      round(long_n   / total * 100, 2) if total else 0.0,
        f"{prefix}_flat_pct":      round(flat_n   / total * 100, 2) if total else 0.0,
        f"{prefix}_short_pct":     round(short_n  / total * 100, 2) if total else 0.0,
        f"{prefix}_start":         df["timestamp_utc"].min() if total else pd.NaT,
        f"{prefix}_end":           df["timestamp_utc"].max() if total else pd.NaT,
    }


def session_distribution(df: pd.DataFrame, prefix: str) -> dict:
    """Compute session row percentages for one split."""
    total = len(df)
    vc = df["session"].value_counts(normalize=True) if total else pd.Series(dtype=float)
    return {
        f"{prefix}_asia_pct":    round(float(vc.get("Asia",     0)) * 100, 2),
        f"{prefix}_london_pct":  round(float(vc.get("London",   0)) * 100, 2),
        f"{prefix}_overlap_pct": round(float(vc.get("Overlap",  0)) * 100, 2),
        f"{prefix}_ny_pct":      round(float(vc.get("New_York", 0)) * 100, 2),
    }


def build_split_summary(
    pair: str,
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
    purge_rows: int,
) -> pd.DataFrame:
    """Build one-row audit record for one pair."""
    record = {
        "pair":                 pair,
        "total_rows":           len(full_df),
        "configured_train_end": train_end,
        "configured_val_end":   val_end,
        "purge_rows":           purge_rows,
    }
    record.update(label_distribution(train_df, "train"))
    record.update(label_distribution(val_df,   "val"))
    record.update(label_distribution(test_df,  "test"))
    record.update(session_distribution(train_df, "train"))
    record.update(session_distribution(test_df,  "test"))
    return pd.DataFrame([record])


def build_manifest(summaries: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate all pair summaries into one global manifest."""
    return pd.concat(summaries, ignore_index=True)


def process_pair(
    pair: str,
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
    purge_rows: int,
    force: bool,
) -> pd.DataFrame:
    """Split one pair and save artifacts. Returns the summary row."""
    train_path   = TRAIN_DIR  / f"{pair}_train.parquet"
    val_path     = VAL_DIR    / f"{pair}_val.parquet"
    test_path    = TEST_DIR   / f"{pair}_test.parquet"
    summary_path = REPORTS_DIR / f"{pair}_split_summary.csv"

    if (train_path.exists() and val_path.exists() and
            test_path.exists() and summary_path.exists() and not force):
        return pd.read_csv(summary_path)

    full_df = load_labeled_pair(pair)
    train_df, val_df, test_df = split_pair(full_df, train_end, val_end, purge_rows)

    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if split.empty:
            raise ValueError(
                f"{pair}: {name} split is empty. "
                f"Check date range and split boundaries."
            )

    summary_df = build_split_summary(
        pair=pair,
        full_df=full_df,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_end=train_end,
        val_end=val_end,
        purge_rows=purge_rows,
    )

    ensure_dir(TRAIN_DIR)
    ensure_dir(VAL_DIR)
    ensure_dir(TEST_DIR)
    ensure_dir(REPORTS_DIR)

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path,     index=False)
    test_df.to_parquet(test_path,   index=False)
    save_csv(summary_df, summary_path)

    return summary_df


def main() -> None:
    args = parse_args()

    train_end  = parse_timestamp(args.train_end)
    val_end    = parse_timestamp(args.val_end)
    purge_rows = args.purge_rows

    if train_end >= val_end:
        raise ValueError("--train-end must be earlier than --val-end.")

    ensure_dir(TRAIN_DIR)
    ensure_dir(VAL_DIR)
    ensure_dir(TEST_DIR)
    ensure_dir(REPORTS_DIR)

    summaries: list[pd.DataFrame] = []

    for pair in PAIRS:
        summary = process_pair(
            pair=pair,
            train_end=train_end,
            val_end=val_end,
            purge_rows=purge_rows,
            force=args.force,
        )
        summaries.append(summary)

    manifest = build_manifest(summaries)
    save_csv(manifest, REPORTS_DIR / "split_manifest.csv")

    print("Splitting complete.")
    print(f"Train   ->  {TRAIN_DIR}")
    print(f"Val     ->  {VAL_DIR}")
    print(f"Test    ->  {TEST_DIR}")
    print(f"Reports ->  {REPORTS_DIR}")
    print(f"\nManifest preview:")
    print(manifest[[
        "pair", "total_rows",
        "train_rows", "val_rows", "test_rows",
        "train_signal_pct", "val_signal_pct", "test_signal_pct",
        "purge_rows",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
from __future__ import annotations

from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
PARQUET_DIR = PROJECT_DIR / "data" / "parquet"

EXPECTED_COLUMNS = [
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


def inspect_pair(path: Path) -> None:
    """Print structural diagnostics for one parquet file."""

    pair = path.stem.replace("_2015_2025", "")
    print("\n" + "=" * 60)
    print(f"PAIR: {pair}")
    print("=" * 60)

    df = pd.read_parquet(path)

    print(f"Rows: {len(df)}")

    print("\nColumns present:")
    print(list(df.columns))

    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        print("\nMissing expected columns:", missing_cols)

    print("\nNull values per column:")
    print(df.isnull().sum())

    if "timestamp_utc" in df.columns:
        duplicates = df["timestamp_utc"].duplicated().sum()
        print("\nDuplicate timestamps:", duplicates)

    if {"open", "high", "low", "close"}.issubset(df.columns):
        invalid_ohlc = (
            (df["high"] < df["low"])
            | (df["open"] > df["high"])
            | (df["open"] < df["low"])
            | (df["close"] > df["high"])
            | (df["close"] < df["low"])
        )
        print("\nInvalid OHLC rows:", int(invalid_ohlc.sum()))

    if "volume" in df.columns:
        print("Zero volume rows:", int((df["volume"] == 0).sum()))

    if "timestamp_utc" in df.columns:
        ts = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
        df["date"] = ts.dt.date

        daily_counts = (
            df.groupby("date")
            .size()
            .rename("observations")
            .reset_index()
        )

        print("\nDaily coverage summary:")
        print("Min observations/day:", int(daily_counts["observations"].min()))
        print("Max observations/day:", int(daily_counts["observations"].max()))
        print("Median observations/day:", int(daily_counts["observations"].median()))

        low_days = (daily_counts["observations"] < 1200).sum()
        print("Days with <1200 observations:", int(low_days))


def main() -> None:
    if not PARQUET_DIR.exists():
        raise FileNotFoundError(f"Missing directory: {PARQUET_DIR}")

    files = sorted(PARQUET_DIR.glob("*_2015_2025.parquet"))

    if not files:
        raise FileNotFoundError("No parquet files found.")

    print("\nFX DATASET INSPECTION")
    print("Directory:", PARQUET_DIR)

    for file in files:
        inspect_pair(file)

    print("\nInspection complete.")


if __name__ == "__main__":
    main()

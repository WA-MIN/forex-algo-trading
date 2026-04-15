from __future__ import annotations

from datetime import datetime
from pathlib  import Path
from typing   import Optional

import pandas as pd

PROJECT_DIR  = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_DIR / "datasets"
CLEANED_DIR  = PROJECT_DIR / "data" / "cleaned"

# UTC hour ranges (inclusive start, exclusive end)
SESSION_HOURS_UTC: dict[str, tuple[int, int]] = {
    "london":  (7,  16),
    "ny":      (13, 22),
    "asia":    (0,   9),
    "overlap": (13, 16),
}


def _load_split(pair: str, split: str) -> pd.DataFrame:
    """
    Load a pre-built split parquet from datasets/.
    Supports: train, val, test, fold_0 through fold_4.
    """
    if split.startswith("fold_"):
        fold_idx = split.split("_")[1]
        path = DATASETS_DIR / "folds" / f"fold_{fold_idx}" / f"{pair}_val.parquet"
    else:
        path = DATASETS_DIR / split / f"{pair}_{split}.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"Split parquet not found: {path}\n"
            f"Run scripts/split_fx_data.py to generate splits."
        )
    return pd.read_parquet(path)


def _load_date_range(pair: str, date_from: datetime, date_to: Optional[datetime]) -> pd.DataFrame:
    """
    Slice cleaned data to a custom date range.
    Falls back to stitching all splits if the cleaned parquet is missing.
    """
    cleaned_path = CLEANED_DIR / f"{pair}_1min_clean.parquet"

    if cleaned_path.exists():
        df = pd.read_parquet(cleaned_path)
    else:
        frames = []
        for split in ("train", "val", "test"):
            p = DATASETS_DIR / split / f"{pair}_{split}.parquet"
            if p.exists():
                frames.append(pd.read_parquet(p))
        if not frames:
            raise FileNotFoundError(
                f"No data found for {pair}. "
                f"Expected cleaned parquet at {cleaned_path} "
                f"or split parquets in {DATASETS_DIR}."
            )
        df = pd.concat(frames, ignore_index=True)

    df = _normalise(df)
    mask = df["timestamp_utc"] >= date_from
    if date_to is not None:
        mask &= df["timestamp_utc"] <= date_to
    return df[mask].reset_index(drop=True)


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure timestamp column is UTC-aware and rows are sorted."""
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        df = df.sort_values("timestamp_utc").reset_index(drop=True)
    return df


def _apply_session_mask(df: pd.DataFrame, session: str) -> pd.DataFrame:
    """Keep only bars that fall within the requested trading session."""
    if "timestamp_utc" not in df.columns:
        return df
    start_h, end_h = SESSION_HOURS_UTC[session]
    hour = df["timestamp_utc"].dt.hour
    mask = (hour >= start_h) & (hour < end_h)
    return df[mask].reset_index(drop=True)


def _apply_entry_time(df: pd.DataFrame, entry_time: str) -> pd.DataFrame:
    """
    Mark bars before entry_time on each calendar day.
    Rows before entry_time are kept for feature continuity but produce
    a flat position signal downstream.
    """
    if "timestamp_utc" not in df.columns:
        return df
    h, m = map(int, entry_time.split(":"))
    minutes_since_midnight = df["timestamp_utc"].dt.hour * 60 + df["timestamp_utc"].dt.minute
    entry_minutes = h * 60 + m
    df = df.copy()
    df.loc[minutes_since_midnight < entry_minutes, "_before_entry"] = True
    return df


def resolve_data(cfg, pair: str) -> pd.DataFrame:
    """
    Single entry point for data loading.

    Reads SimConfig and returns a filtered DataFrame ready for signal
    generation.

    Parameters
    ----------
    cfg  : SimConfig  (cli.config.SimConfig)
    pair : str        e.g. 'EURUSD'

    Returns
    -------
    pd.DataFrame with at minimum: timestamp_utc, open, high, low, close
    """
    if cfg.date_from is not None:
        df = _load_date_range(pair, cfg.date_from, cfg.date_to)
    else:
        df = _load_split(pair, cfg.split)
        df = _normalise(df)

    if df.empty:
        raise ValueError(
            f"No data for {pair} with the given parameters "
            f"(split={cfg.split}, from={cfg.date_from}, to={cfg.date_to})."
        )

    if cfg.session is not None:
        if cfg.session not in SESSION_HOURS_UTC:
            raise ValueError(
                f"Unknown session '{cfg.session}'. "
                f"Choose from: {list(SESSION_HOURS_UTC.keys())}"
            )
        df = _apply_session_mask(df, cfg.session)

    if cfg.entry_time is not None:
        df = _apply_entry_time(df, cfg.entry_time)

    if df.empty:
        raise ValueError(
            f"DataFrame for {pair} is empty after applying filters "
            f"(session={cfg.session}, entry_time={cfg.entry_time})."
        )

    return df

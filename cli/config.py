from __future__ import annotations

from dataclasses import dataclass, field
from typing      import Optional
from datetime    import datetime


@dataclass
class SimConfig:
    # ── REQUIRED ──────────────────────────────────────────────────────────────
    pair:       str
    strategy:   str

    # ── DATA WINDOW ───────────────────────────────────────────────────────────
    split:      Optional[str]      = "val"   # "train" | "val" | "test" | "fold_0..4"
    date_from:  Optional[datetime] = None
    date_to:    Optional[datetime] = None

    # ── CAPITAL ───────────────────────────────────────────────────────────────
    capital:    float = 10_000.0

    # ── EXECUTION PARAMETERS ──────────────────────────────────────────────────
    tp_pips:       Optional[float] = None
    sl_pips:       Optional[float] = None
    max_hold_bars: Optional[int]   = None
    entry_time:    Optional[str]   = None    # e.g. "09:00" UTC

    # ── RESEARCH PARAMETERS ───────────────────────────────────────────────────
    folds:           int             = 0
    spread_override: Optional[float] = None
    session:         Optional[str]   = None  # None | "london" | "ny" | "asia" | "overlap"

    # ── OUTPUT ────────────────────────────────────────────────────────────────
    open_browser: bool          = True
    output_path:  Optional[str] = None
    no_browser:   bool          = False
    mode:         str           = "research"  # "research" | "simulation"

from __future__ import annotations

from dataclasses import dataclass, field
from typing      import Optional
from datetime    import datetime


@dataclass
class SimConfig:
    pair:       str
    strategy:   str

    split:      Optional[str]      = "val"  
    date_from:  Optional[datetime] = None
    date_to:    Optional[datetime] = None

    capital:    float = 10_000.0

    tp_pips:       Optional[float] = None
    sl_pips:       Optional[float] = None
    max_hold_bars: Optional[int]   = None
    entry_time:    Optional[str]   = None    

    folds:           int             = 0
    spread_override: Optional[float] = None
    session:         Optional[str]   = None  

    open_browser: bool          = True
    output_path:  Optional[str] = None
    no_browser:   bool          = False
    mode:         str           = "research" 

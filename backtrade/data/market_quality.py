from __future__ import annotations

import numpy as np
import pandas as pd


BOOK_PRICE_COLUMNS = [
    f"{side}{level}_prc"
    for side in ("bid", "ask")
    for level in range(1, 6)
]
BOOK_QTY_COLUMNS = [
    f"{side}{level}_qty"
    for side in ("bid", "ask")
    for level in range(1, 6)
]
BOOK_COLUMNS = [*BOOK_PRICE_COLUMNS, *BOOK_QTY_COLUMNS]


def _as_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def derive_market_quality_flags(frame: pd.DataFrame, *, tick_size: float) -> pd.DataFrame:
    """Derive conservative maker quality flags from raw five-level snapshots.

    Existing upstream flags can only make a row less usable. They never override
    structural validation derived from the raw book.
    """

    if tick_size <= 0 or not np.isfinite(float(tick_size)):
        raise ValueError("tick_size must be positive and finite")
    missing = sorted(set(BOOK_COLUMNS) - set(frame.columns))
    if missing:
        raise KeyError(f"market is missing raw five-level columns: {missing}")

    out = frame.copy()
    values = {column: _as_numeric(out, column) for column in BOOK_COLUMNS}
    invalid = pd.Series(False, index=out.index)
    for column in BOOK_PRICE_COLUMNS:
        series = values[column]
        invalid |= series.isna() | ~np.isfinite(series.to_numpy(dtype=float, na_value=np.nan)) | series.le(0)
        off_tick = (series / float(tick_size) - (series / float(tick_size)).round()).abs() > 1e-9
        invalid |= off_tick.fillna(True)
    for column in BOOK_QTY_COLUMNS:
        series = values[column]
        invalid |= series.isna() | ~np.isfinite(series.to_numpy(dtype=float, na_value=np.nan)) | series.lt(0)
        invalid |= ((series - series.round()).abs() > 1e-9).fillna(True)

    invalid |= values["bid1_prc"].ge(values["ask1_prc"]).fillna(True)
    for level in range(1, 5):
        invalid |= values[f"bid{level}_prc"].le(values[f"bid{level + 1}_prc"]).fillna(True)
        invalid |= values[f"ask{level}_prc"].ge(values[f"ask{level + 1}_prc"]).fillna(True)

    last = _as_numeric(out, "last_prc") if "last_prc" in out else pd.Series(np.nan, index=out.index)
    volume = _as_numeric(out, "vol_inc") if "vol_inc" in out else pd.Series(np.nan, index=out.index)
    clear_touch = last.ge(values["ask1_prc"]) | last.le(values["bid1_prc"])
    derived_ambiguous = volume.gt(0) & ~clear_touch
    if "side_ambiguous_flag" in out:
        derived_ambiguous |= out["side_ambiguous_flag"].fillna(True).astype(bool)
    if "is_anomaly" in out:
        invalid |= out["is_anomaly"].fillna(True).astype(bool)

    out["is_anomaly"] = invalid.astype(bool)
    out["side_ambiguous_flag"] = derived_ambiguous.astype(bool)
    if "is_stale" not in out:
        out["is_stale"] = False
    else:
        out["is_stale"] = out["is_stale"].fillna(True).astype(bool)

    out["mid1"] = (values["bid1_prc"] + values["ask1_prc"]) / 2.0
    out["spread1"] = values["ask1_prc"] - values["bid1_prc"]
    return out


__all__ = ["BOOK_COLUMNS", "BOOK_PRICE_COLUMNS", "BOOK_QTY_COLUMNS", "derive_market_quality_flags"]

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backtrade.config.schema import BacktradeConfig
from backtrade.data.future_l2 import load_future_l2_frame
from backtrade.simulation.compact_v9 import audit_compact_v9, read_compact_v9


_PLOT_ORIGIN = pd.Timestamp("2000-01-01")


def _numeric_column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[name], errors="coerce")


def _front_adjusted_price(frame: pd.DataFrame) -> tuple[pd.Series, str]:
    """Return complete adjusted prices, or raw prices when adjustment data is incomplete."""
    raw = _numeric_column(frame, "last_prc")
    if raw.isna().all():
        raw = _numeric_column(frame, "mid1")
    if "last_prc_adj" in frame:
        adjusted = _numeric_column(frame, "last_prc_adj")
        if bool(adjusted.notna().all()) and bool(np.isfinite(adjusted.to_numpy(float)).all()):
            return adjusted, "front_adjusted"
    if "adj_factor" in frame:
        factor = _numeric_column(frame, "adj_factor")
        if bool(raw.notna().all()) and bool(factor.notna().all()) and bool(np.isfinite(factor.to_numpy(float)).all()):
            return raw * factor, "front_adjusted"
    return raw, "raw"


def _attach_trading_timeline(frame: pd.DataFrame) -> pd.DataFrame:
    """Compress overnight, lunch, and weekend gaps while retaining actual timestamps."""
    out = frame.copy()
    if out.empty:
        out["actual_ts"] = pd.Series(dtype="datetime64[ns]")
        out["plot_ts"] = pd.Series(dtype="datetime64[ns]")
        out["front_adjusted_price"] = pd.Series(dtype="float64")
        out.attrs["price_basis"] = "raw"
        return out
    out["actual_ts"] = pd.to_datetime(out["tick_ts"], errors="raise")
    sort_columns = ["actual_ts"]
    if "source_seq" in out:
        sort_columns.append("source_seq")
    out = out.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    out["front_adjusted_price"], price_basis = _front_adjusted_price(out)
    out.attrs["price_basis"] = price_basis
    out["_plot_seconds"] = np.nan
    group_columns = [column for column in ("trading_day", "session_id") if column in out]
    if not group_columns:
        group_columns = ["_single_group"]
        out["_single_group"] = "all"
    cursor = 0.0
    for _, group in out.groupby(group_columns, sort=False, dropna=False):
        local = (group["actual_ts"] - group["actual_ts"].iloc[0]).dt.total_seconds()
        out.loc[group.index, "_plot_seconds"] = cursor + local.to_numpy()
        cursor += float(local.iloc[-1]) + 1e-6
    out["plot_ts"] = _PLOT_ORIGIN + pd.to_timedelta(out["_plot_seconds"], unit="s")
    return out.drop(columns=["_plot_seconds", "_single_group"], errors="ignore")


def _map_event_timeline(events: pd.DataFrame, market: pd.DataFrame, *, event_column: str = "event_ts") -> pd.DataFrame:
    out = events.copy()
    if out.empty:
        out["actual_ts"] = pd.Series(dtype="datetime64[ns]")
        out["plot_ts"] = pd.Series(dtype="datetime64[ns]")
        return out
    out["actual_ts"] = pd.to_datetime(out[event_column], errors="raise")
    timeline = market[["actual_ts", "plot_ts"]].drop_duplicates("actual_ts").sort_values("actual_ts")
    if timeline.empty:
        out["plot_ts"] = pd.NaT
        return out
    out["_event_order"] = np.arange(len(out), dtype="int64")
    mapped = pd.merge_asof(
        out.sort_values("actual_ts", kind="stable"),
        timeline,
        on="actual_ts",
        direction="backward",
        allow_exact_matches=True,
    )
    mapped["plot_ts"] = mapped["plot_ts"].fillna(timeline["plot_ts"].iloc[0])
    return mapped.sort_values("_event_order", kind="stable").drop(columns="_event_order")


def _last_valid(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.iloc[-1]) if not values.empty else None


def _hourly_market(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["event_ts", "actual_ts", "trading_day", "price", "factor_q10", "factor_q90", "volume"]
    if frame.empty or "plot_ts" not in frame:
        return pd.DataFrame(columns=columns)
    data = frame.copy().sort_values("plot_ts", kind="stable")
    data["hour_bucket"] = pd.to_datetime(data["plot_ts"]).dt.floor("1h")
    rows = []
    for _, group in data.groupby("hour_bucket", sort=True, dropna=False):
        row = {
            "event_ts": group["plot_ts"].iloc[-1],
            "actual_ts": group["actual_ts"].iloc[-1],
            "trading_day": group["trading_day"].iloc[-1] if "trading_day" in group else None,
            "price": _last_valid(group["front_adjusted_price"]),
        }
        factor = pd.to_numeric(group["active_factor"], errors="coerce").dropna() if "active_factor" in group else pd.Series(dtype=float)
        row["factor_q10"] = float(factor.quantile(0.10)) if not factor.empty else np.nan
        row["factor_q90"] = float(factor.quantile(0.90)) if not factor.empty else np.nan
        row["volume"] = float(_numeric_column(group, "vol_inc").sum(min_count=1)) if "vol_inc" in group else np.nan
        rows.append(row)
    return pd.DataFrame(rows, columns=columns).dropna(how="all", subset=columns[3:])


def _hourly_drawdown(snapshots: pd.DataFrame) -> pd.DataFrame:
    columns = ["event_ts", "actual_ts", "drawdown"]
    if snapshots.empty or not {"plot_ts", "equity"}.issubset(snapshots.columns):
        return pd.DataFrame(columns=columns)
    data = snapshots.copy().sort_values("plot_ts", kind="stable")
    equity = pd.to_numeric(data["equity"], errors="coerce")
    data["drawdown_value"] = equity - equity.cummax()
    data["hour_bucket"] = pd.to_datetime(data["plot_ts"]).dt.floor("1h")
    rows = []
    for _, group in data.groupby("hour_bucket", sort=True, dropna=False):
        rows.append({
            "event_ts": group["plot_ts"].iloc[-1],
            "actual_ts": group["actual_ts"].iloc[-1],
            "drawdown": float(group["drawdown_value"].min()),
        })
    return pd.DataFrame(rows, columns=columns)


def load_report_data(run_root: str | Path) -> dict:
    root = Path(run_root).expanduser().resolve()
    manifest = read_compact_v9(root)
    audit = audit_compact_v9(root, require_final_flat=True)
    if not audit.get("passed"):
        raise ValueError(f"cannot report an unaudited compact_v9 run: {audit.get('errors', [])}")
    activity = pd.read_parquet(root / "activity_ledger.parquet")
    accounts = pd.read_parquet(root / "account_ledger.parquet")
    snapshots = pd.read_parquet(root / "state_snapshots.parquet")
    maker_events = pd.read_parquet(root / "maker_events.parquet") if manifest.get("match_mode") == "maker" else pd.DataFrame()
    config = BacktradeConfig.model_validate(manifest["config"])
    runtime = manifest.get("runtime", {})
    max_events = runtime.get("max_events") or runtime.get("max_ticks")
    market = _attach_trading_timeline(load_future_l2_frame(config, max_events=int(max_events) if max_events else None))
    activity = _map_event_timeline(activity, market)
    accounts = _map_event_timeline(accounts, market)
    snapshots = _map_event_timeline(snapshots, market)
    maker_events = _map_event_timeline(maker_events, market)
    return {
        "root": root,
        "manifest": manifest,
        "audit": audit,
        "activity": activity,
        "accounts": accounts,
        "snapshots": snapshots,
        "maker_events": maker_events,
        "market": market,
        "price_basis": market.attrs.get("price_basis", "raw"),
        "hourly_market": _hourly_market(market),
        "hourly_drawdown": _hourly_drawdown(snapshots),
    }


__all__ = ["load_report_data"]

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value) if pd.notna(value) else default
    except (TypeError, ValueError):
        return default


def _round(value: float | None) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def _max_drawdown(equity: pd.Series) -> tuple[float, int, str | None, str | None]:
    values = pd.to_numeric(equity, errors="coerce").dropna()
    if values.empty:
        return 0.0, 0, None, None
    running = values.cummax()
    drawdown = values - running
    trough = drawdown.idxmin()
    peak = values.loc[:trough].idxmax()
    duration = max(0, int((trough - peak).total_seconds())) if hasattr(trough, "__sub__") else 0
    return float(drawdown.min()), duration, str(peak), str(trough)


def compute_report_metrics(
    accounts: pd.DataFrame,
    snapshots: pd.DataFrame,
    activity: pd.DataFrame,
    initial_cash: float,
    *,
    signal_mode: str = "signed_factor",
    short_threshold: float | None = None,
    long_threshold: float | None = None,
    snapshot_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    accounts = accounts.copy()
    snapshots = snapshots.copy()
    activity = activity.copy()
    if "event_ts" in accounts:
        accounts["event_ts"] = pd.to_datetime(accounts["event_ts"])
    if "event_ts" in snapshots:
        snapshots["event_ts"] = pd.to_datetime(snapshots["event_ts"])
    fills = accounts
    net_pnl = float(pd.to_numeric(fills.get("net_pnl", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    gross_pnl = float(pd.to_numeric(fills.get("gross_pnl", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    open_fee = float(pd.to_numeric(fills.get("open_fee", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    close_fee = float(pd.to_numeric(fills.get("close_fee", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    total_fee = open_fee + close_fee
    final_cash = _number(fills.iloc[-1].get("cash_after"), initial_cash) if not fills.empty else initial_cash
    final_realized = _number(fills.iloc[-1].get("realized_pnl_after"), net_pnl + total_fee) if not fills.empty else 0.0
    final_total_fee = _number(fills.iloc[-1].get("total_fee_after"), total_fee) if not fills.empty else total_fee
    round_trips = int(((fills.get("position_before", pd.Series(dtype=float)) != 0) & (fills.get("position_after", pd.Series(dtype=float)) == 0)).sum())
    if snapshot_stats is None:
        max_dd, dd_duration, dd_peak, dd_trough = _max_drawdown(snapshots.set_index("event_ts")["equity"] if {"event_ts", "equity"}.issubset(snapshots.columns) else pd.Series(dtype=float))
        daily_returns_override = None
        final_position_flat = bool(snapshots.empty or _number(snapshots.iloc[-1].get("position_qty"), 0.0) == 0)
    else:
        max_dd = float(snapshot_stats.get("max_drawdown", 0.0))
        dd_duration = int(snapshot_stats.get("max_drawdown_duration_seconds", 0))
        dd_peak = snapshot_stats.get("drawdown_peak_ts")
        dd_trough = snapshot_stats.get("drawdown_trough_ts")
        daily_returns_override = pd.Series(snapshot_stats.get("daily_returns", []), dtype="float64")
        final_position_flat = bool(snapshot_stats.get("final_position_flat", True))

    factor_decisions = activity[activity.get("record_type", pd.Series(index=activity.index)).eq("target")] if not activity.empty else activity
    if not factor_decisions.empty and "factor_score" in factor_decisions:
        scores = pd.to_numeric(factor_decisions["factor_score"], errors="coerce")
        decision_mask = factor_decisions.get("factor_decision", pd.Series(True, index=factor_decisions.index)).fillna(False).astype(bool)
        decision_scores = scores[decision_mask]
        if signal_mode == "ecdf_tail":
            if short_threshold is None or long_threshold is None:
                raise ValueError("ecdf_tail report metrics require short and long thresholds")
            active = decision_scores.le(float(short_threshold)) | decision_scores.ge(float(long_threshold))
        else:
            active = decision_scores.fillna(0).abs() > 0
        active_rate = float(active.mean()) if bool(decision_mask.any()) else 0.0
    else:
        active_rate = 0.0

    daily = pd.DataFrame()
    if {"event_ts", "equity"}.issubset(snapshots.columns) and not snapshots.empty:
        snap = snapshots.sort_values("event_ts").set_index("event_ts")
        daily = snap["equity"].resample("1D").last().dropna().to_frame("equity")
        daily["return"] = daily["equity"].pct_change().fillna(daily["equity"].iloc[0] / initial_cash - 1 if initial_cash else 0.0)
    daily_returns = daily_returns_override if daily_returns_override is not None else daily["return"] if "return" in daily else pd.Series(dtype=float)
    sharpe = float(np.sqrt(252.0) * daily_returns.mean() / daily_returns.std(ddof=1)) if len(daily_returns) > 1 and daily_returns.std(ddof=1) > 0 else None
    win_days = int((daily_returns > 0).sum())
    loss_days = int((daily_returns < 0).sum())

    return {
        "initial_cash": _round(float(initial_cash)),
        "final_cash": _round(final_cash),
        "final_cash_delta": _round(final_cash - initial_cash),
        "net_pnl": _round(net_pnl),
        "gross_pnl": _round(gross_pnl),
        "total_fee": _round(total_fee),
        "final_realized_pnl": _round(final_realized),
        "final_total_fee": _round(final_total_fee),
        "net_return": _round((final_cash - initial_cash) / initial_cash if initial_cash else None),
        "fill_count": int(len(fills)),
        "round_trips": round_trips,
        "max_drawdown": _round(max_dd),
        "max_drawdown_duration_seconds": dd_duration,
        "drawdown_peak_ts": dd_peak,
        "drawdown_trough_ts": dd_trough,
        "daily_sharpe": _round(sharpe),
        "winning_days": win_days,
        "losing_days": loss_days,
        "factor_active_rate": _round(active_rate),
        "target_count": int(len(factor_decisions)),
        "maker_fill_count": int(((activity.get("record_type", pd.Series(index=activity.index)).eq("fill")) & (activity.get("maker_taker_role", pd.Series(index=activity.index)).eq("maker"))).sum()) if not activity.empty else 0,
        "reason_counts": {str(key): int(value) for key, value in Counter(fills.get("reason_code", pd.Series(dtype=object)).dropna()).items()},
        "reconciliation": {
            "net_pnl_equals_cash_delta": bool(np.isclose(net_pnl, final_cash - initial_cash, atol=1e-6)),
            "net_pnl_equals_realized_minus_fee": bool(np.isclose(net_pnl, final_realized - final_total_fee, atol=1e-6)),
            "final_position_flat": final_position_flat,
        },
    }


__all__ = ["compute_report_metrics"]

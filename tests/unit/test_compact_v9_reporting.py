from __future__ import annotations

import pandas as pd

from backtrade.reporting.data import _attach_trading_timeline
from backtrade.reporting.html import _time_axis, _transaction_cash_points
from backtrade.reporting.metrics import compute_report_metrics


def test_report_metrics_reconcile_cash_and_include_core_statistics():
    accounts = pd.DataFrame(
        [
            {
                "event_ts": pd.Timestamp("2026-01-05 09:00:00"),
                "fill_seq": 1,
                "position_before": 0,
                "position_after": 1,
                "cash_before": 1000.0,
                "cash_after": 999.0,
                "equity_after": 1000.0,
                "gross_pnl": 0.0,
                "open_fee": 1.0,
                "close_fee": 0.0,
                "net_pnl": -1.0,
                "open_fill_seq": 1,
                "holding_ms": None,
                "reason_code": "open",
            },
            {
                "event_ts": pd.Timestamp("2026-01-05 10:00:00"),
                "fill_seq": 2,
                "position_before": 1,
                "position_after": 0,
                "cash_before": 999.0,
                "cash_after": 1009.0,
                "equity_after": 1009.0,
                "gross_pnl": 12.0,
                "open_fee": 0.0,
                "close_fee": 2.0,
                "net_pnl": 10.0,
                "open_fill_seq": 1,
                "holding_ms": 3_600_000,
                "reason_code": "close",
            },
        ]
    )
    snapshots = pd.DataFrame(
        {
            "event_ts": pd.to_datetime(["2026-01-05 09:00:00", "2026-01-05 10:00:00"]),
            "equity": [1000.0, 1009.0],
            "cash": [999.0, 1009.0],
            "position_qty": [1, 0],
        }
    )
    activity = pd.DataFrame(
        {
            "record_type": ["target", "target"],
            "factor_score": [0.0, 2.0],
            "factor_decision": [True, True],
        }
    )

    metrics = compute_report_metrics(accounts, snapshots, activity, initial_cash=1000.0)

    assert metrics["net_pnl"] == 9.0
    assert metrics["final_cash_delta"] == 9.0
    assert metrics["total_fee"] == 3.0
    assert metrics["round_trips"] == 1
    assert metrics["factor_active_rate"] == 0.5
    assert metrics["net_return"] == 0.009


def test_report_cash_points_follow_fills_and_axis_has_one_tick_per_day():
    snapshots = pd.DataFrame(
        {
            "plot_ts": pd.to_datetime(["2000-01-01 00:00:00", "2000-01-01 00:01:00", "2000-01-02 00:00:00"]),
            "actual_ts": pd.to_datetime(["2026-01-05 09:00:00", "2026-01-05 09:01:00", "2026-01-06 09:00:00"]),
            "cash": [1000.0, 1000.0, 1009.0],
        }
    )
    accounts = pd.DataFrame(
        {
            "plot_ts": pd.to_datetime(["2000-01-01 00:00:30", "2000-01-02 00:00:30"]),
            "actual_ts": pd.to_datetime(["2026-01-05 09:00:30", "2026-01-06 09:00:30"]),
            "cash_after": [999.0, 1009.0],
            "net_pnl": [-1.0, 10.0],
        }
    )

    cash_points = _transaction_cash_points(accounts, snapshots)

    assert cash_points["cash"].tolist() == [999.0, 1009.0]
    assert cash_points["actual_ts"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist() == [
        "2026-01-05 09:00:30",
        "2026-01-06 09:00:30",
    ]
    ticks, labels, year_ticks, year_labels = _time_axis(
        pd.DataFrame(
            {
                "plot_ts": pd.to_datetime(["2000-01-01 00:00:00", "2000-01-01 00:01:00", "2000-01-02 00:00:00"]),
                "actual_ts": pd.to_datetime(["2026-01-05 09:00:00", "2026-01-05 09:01:00", "2026-01-06 09:00:00"]),
            }
        )
    )
    assert len(ticks) == 2
    assert labels == ["1/5", "1/6"]
    assert len(year_ticks) == 1
    assert year_labels == ["2026"]


def test_report_uses_raw_price_when_adjustment_is_missing_or_incomplete():
    base = pd.DataFrame(
        {
            "tick_ts": pd.to_datetime(["2026-01-05 09:00:00", "2026-01-05 09:00:05"]),
            "trading_day": ["2026-01-05", "2026-01-05"],
            "session_id": ["day", "day"],
            "last_prc": [100.0, 101.0],
        }
    )
    raw = _attach_trading_timeline(base)
    assert raw["front_adjusted_price"].tolist() == [100.0, 101.0]
    assert raw.attrs["price_basis"] == "raw"

    incomplete = base.assign(last_prc_adj=[110.0, None], adj_factor=[1.1, None])
    result = _attach_trading_timeline(incomplete)
    assert result["front_adjusted_price"].tolist() == [100.0, 101.0]
    assert result.attrs["price_basis"] == "raw"


def test_report_display_formats_net_return_as_percent():
    from backtrade.reporting.html import _display_value

    assert _display_value("净收益率", 0.0125) == "1.25%"
    assert _display_value("净 PnL", 1.25) == "1.25"

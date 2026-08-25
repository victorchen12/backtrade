from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pandas as pd
import pyarrow.parquet as pq
import pytest

from backtrade.config.schema import BacktradeConfig, StrategyConfig
from backtrade.reporting.html import _build_figures
from backtrade.reporting.metrics import compute_report_metrics
from backtrade.simulation.compact_v9 import CompactV9ParquetOutput, _audit_signal_execution
from backtrade.simulation.compact_v9_runner import CompactV9Runner
from backtrade.simulation.events import MarketTick, Order, StrategyView
from backtrade.simulation.state import MatchMode, OrderSide, OrderType, TimeInForce


TS = datetime(2025, 4, 15, 9, 0, tzinfo=timezone.utc)
FACTOR = "v2p3_test_factor"


def _view(score: float) -> StrategyView:
    return StrategyView(
        product="ag",
        contract="AG2506",
        tick_ts=TS,
        mid=100.0,
        factors={"active_factor": score},
        factor_decision=True,
        factor_source_ts=TS,
        factor_age_ms=0.0,
    )


def _runner_config(mode: str) -> BacktradeConfig:
    return BacktradeConfig.model_validate(
        {
            "initial_cash": 1_000_000,
            "data": {
                "source": "future_l2",
                "product": "ag",
                "max_ticks": 4,
                "factor_grid_mode": "decision_grid",
                "eof_is_day_end": False,
            },
            "strategy": {
                "factor_name": FACTOR,
                "factor_column": FACTOR,
                "signal_mode": "ecdf_tail",
                "short_threshold": -0.8,
                "long_threshold": 0.8,
            },
            "execution": {"latency_ms": 5, "day_end_flatten_window_ms": 0},
            "match": {"mode": mode},
            "limit_reference": {"mode": "disabled"},
            "contracts": {
                "ag": {
                    "code": "AG",
                    "exchange": "SHFE",
                    "tick_size": 1.0,
                    "multiplier": 15.0,
                    "fee": {
                        "open": {"mode": "per_lot", "value": 0.0},
                        "close": {"mode": "per_lot", "value": 0.0},
                        "close_today": {"mode": "per_lot", "value": 0.0},
                    },
                    "price_limit": {"mode": "none", "value": 0.0},
                }
            },
        }
    )


def _tick(ts: datetime, seq: int, score: float) -> MarketTick:
    return MarketTick(
        product="ag",
        contract="AG2506",
        tick_ts=ts,
        last_price=100.0,
        bid_prices=(99.0, 98.0, 97.0, 96.0, 95.0),
        bid_qtys=(5, 5, 5, 5, 5),
        ask_prices=(101.0, 102.0, 103.0, 104.0, 105.0),
        ask_qtys=(5, 5, 5, 5, 5),
        trading_day="2025-04-15",
        session_id="day",
        source_seq=seq,
        factors={"active_factor": score},
        factor_decision=True,
        factor_source_ts=ts,
        factor_age_ms=0.0,
    )


def test_strategy_config_defaults_to_signed_and_requires_explicit_ecdf_thresholds() -> None:
    default = StrategyConfig()
    assert default.signal_mode == "signed_factor"
    assert default.short_threshold is None
    assert default.long_threshold is None

    configured = StrategyConfig(
        factor_name=FACTOR,
        factor_column=FACTOR,
        signal_mode="ecdf_tail",
        short_threshold=-0.8,
        long_threshold=0.8,
    )
    assert configured.short_threshold == -0.8
    assert configured.long_threshold == 0.8
    with pytest.raises(ValueError, match="threshold"):
        StrategyConfig(signal_mode="ecdf_tail")
    with pytest.raises(ValueError, match="short_threshold.*long_threshold"):
        StrategyConfig(signal_mode="ecdf_tail", short_threshold=0.8, long_threshold=-0.8)


def test_ecdf_tail_strategy_includes_boundaries_zeros_middle_and_steps_reversals() -> None:
    from backtrade.strategies.ecdf_tail import EcdfTailStrategy

    strategy = EcdfTailStrategy(FACTOR, short_threshold=-0.8, long_threshold=0.8)

    assert strategy.on_decision(_view(-0.8), 0).target_qty == -1
    assert strategy.on_decision(_view(0.8), 0).target_qty == 1
    assert strategy.on_decision(_view(-0.799999), 0).target_qty == 0
    assert strategy.on_decision(_view(0.0), 0).target_qty == 0
    assert strategy.on_decision(_view(0.799999), 0).target_qty == 0

    neutral_flatten = strategy.on_decision(_view(0.0), 1)
    assert (neutral_flatten.target_qty, neutral_flatten.reduce_only) == (0, True)
    reversal_flatten = strategy.on_decision(_view(-0.8), 1)
    assert (reversal_flatten.target_qty, reversal_flatten.reduce_only) == (0, True)
    reopened = strategy.on_decision(_view(-0.8), 0)
    assert (reopened.target_qty, reopened.factor_semantics_version) == (-1, "ecdf_tail_v1")


@pytest.mark.parametrize("mode", ["maker", "taker"])
def test_ecdf_tail_runner_keeps_single_lot_positions_and_finishes_flat_without_timeout(mode: str) -> None:
    ticks = [_tick(TS + timedelta(seconds=5 * index), index + 1, score) for index, score in enumerate([0.8, 0.8, 0.0, 0.0])]
    ticks[2] = replace(
        ticks[2],
        last_price=99.0,
        vol_inc=5,
        amount_inc=99.0 * 5 * 15,
        trade_direction=OrderSide.SELL,
        trade_direction_source="bid_touch",
        trade_direction_confidence="high",
        trade_direction_quality="normal",
    )

    result = CompactV9Runner(_runner_config(mode), ticks).run()

    targets = [row for row in result.activity_rows if row.get("record_type") == "target"]
    assert {row["target_qty"] for row in targets} <= {-1, 0, 1}
    assert {row["factor_semantics_version"] for row in targets} == {"ecdf_tail_v1"}
    assert {row["position_qty"] for row in result.snapshots} <= {-1, 0, 1}
    assert result.final_snapshot["net_qty"].get("ag", 0) == 0
    assert not any("timeout" in str(event.get("reason_code", "")).lower() for event in result.maker_events)


def test_snapshot_sink_keeps_runner_result_streaming(tmp_path) -> None:
    sink = CompactV9ParquetOutput(tmp_path, maker_enabled=False, batch_size=2)
    ticks = [_tick(TS + timedelta(seconds=5 * index), index + 1, score) for index, score in enumerate([0.8, 0.8, 0.0, 0.0])]

    result = CompactV9Runner(_runner_config("taker"), ticks, snapshot_sink=sink).run()
    sink.close()

    assert result.snapshots == []
    assert pq.ParquetFile(tmp_path / "state_snapshots.parquet").metadata.num_rows == len(ticks)


def test_order_activity_updates_use_order_index_instead_of_scanning_history() -> None:
    runner = CompactV9Runner(_runner_config("maker"), [])
    order = Order.create(
        contract="AG2506",
        product="ag",
        side=OrderSide.BUY,
        qty=1,
        limit_price=99.0,
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.GTC,
        match_mode=MatchMode.MAKER,
        decision_ts=TS,
        reason_code="signal",
    )
    runner._append_order_activity(order)

    class NoReverseRows(list):
        def __reversed__(self):
            raise AssertionError("order activity updates must use the order index")

    runner.activity_rows = NoReverseRows(runner.activity_rows)
    order.actual_arrival_ts = TS + timedelta(milliseconds=5)
    order.arrival_bid1 = 99.0
    runner._append_order_activity(order)

    assert runner._order_activity_rows[order.order_id] is runner.activity_rows[0]
    assert runner.activity_rows[0]["actual_arrival_ts"] == order.actual_arrival_ts


def _ecdf_audit_config() -> dict[str, object]:
    return {"signal_mode": "ecdf_tail", "short_threshold": -0.8, "long_threshold": 0.8}


def test_ecdf_audit_accepts_tail_direction_middle_flat_and_stepwise_reversal() -> None:
    activity = [
        {"event_seq": 1, "record_type": "target", "target_seq": 1, "product": "ag", "event_ts": TS, "factor_score": 0.8, "factor_decision": True, "factor_source_ts": TS, "factor_age_ms": 0.0, "factor_semantics_version": "ecdf_tail_v1", "target_qty": 1},
        {"event_seq": 2, "record_type": "fill", "fill_seq": 1, "product": "ag", "boundary_reason": None},
        {"event_seq": 3, "record_type": "target", "target_seq": 2, "product": "ag", "event_ts": TS + timedelta(seconds=5), "factor_score": -0.8, "factor_decision": True, "factor_source_ts": TS + timedelta(seconds=5), "factor_age_ms": 0.0, "factor_semantics_version": "ecdf_tail_v1", "target_qty": 0},
        {"event_seq": 4, "record_type": "fill", "fill_seq": 2, "product": "ag", "boundary_reason": None},
        {"event_seq": 5, "record_type": "target", "target_seq": 3, "product": "ag", "event_ts": TS + timedelta(seconds=10), "factor_score": -0.8, "factor_decision": True, "factor_source_ts": TS + timedelta(seconds=10), "factor_age_ms": 0.0, "factor_semantics_version": "ecdf_tail_v1", "target_qty": -1},
        {"event_seq": 6, "record_type": "target", "target_seq": 4, "product": "au", "event_ts": TS, "factor_score": 0.0, "factor_decision": True, "factor_source_ts": TS, "factor_age_ms": 0.0, "factor_semantics_version": "ecdf_tail_v1", "target_qty": 0},
    ]
    accounts = [
        {"fill_seq": 1, "position_before": 0, "position_after": 1},
        {"fill_seq": 2, "position_before": 1, "position_after": 0},
    ]

    assert _audit_signal_execution(activity, accounts, strategy_config=_ecdf_audit_config()) == []


@pytest.mark.parametrize("score,target", [(0.8, -1), (-0.8, 1), (0.0, 1)])
def test_ecdf_audit_rejects_wrong_tail_or_middle_target(score: float, target: int) -> None:
    activity = [
        {"event_seq": 1, "record_type": "target", "target_seq": 1, "product": "ag", "event_ts": TS, "factor_score": score, "factor_decision": True, "factor_source_ts": TS, "factor_age_ms": 0.0, "factor_semantics_version": "ecdf_tail_v1", "target_qty": target}
    ]

    errors = _audit_signal_execution(activity, [], strategy_config=_ecdf_audit_config())

    assert any("ecdf tail semantics" in error for error in errors)


def test_audit_reports_semantics_mismatch_without_following_row_declared_mode() -> None:
    activity = [
        {
            "event_seq": 1,
            "record_type": "target",
            "target_seq": 1,
            "product": "ag",
            "event_ts": TS,
            "factor_score": 0.8,
            "factor_decision": True,
            "factor_source_ts": TS,
            "factor_age_ms": 0.0,
            "factor_semantics_version": "ecdf_tail_v1",
            "target_qty": 1,
        }
    ]

    errors = _audit_signal_execution(activity, [], strategy_config={"signal_mode": "signed_factor"})

    assert any("does not match strategy config" in error for error in errors)


def test_report_active_rate_and_reference_lines_use_ecdf_thresholds() -> None:
    activity = pd.DataFrame(
        {
            "record_type": ["target"] * 5,
            "factor_score": [-0.9, -0.79, 0.0, 0.79, 0.8],
            "factor_decision": [True] * 5,
        }
    )
    metrics = compute_report_metrics(
        pd.DataFrame(),
        pd.DataFrame(),
        activity,
        initial_cash=1_000_000,
        signal_mode="ecdf_tail",
        short_threshold=-0.8,
        long_threshold=0.8,
    )
    assert metrics["factor_active_rate"] == pytest.approx(0.4)

    market = pd.DataFrame(
        {
            "plot_ts": pd.to_datetime(["2000-01-01 00:00:00", "2000-01-01 00:01:00"]),
            "actual_ts": pd.to_datetime(["2025-04-15 09:00:00", "2025-04-15 09:01:00"]),
            "active_factor": [-0.9, 0.9],
            "front_adjusted_price": [100.0, 101.0],
        }
    )
    hourly = pd.DataFrame(
        {
            "event_ts": pd.to_datetime(["2000-01-01 00:01:00"]),
            "actual_ts": pd.to_datetime(["2025-04-15 09:01:00"]),
            "trading_day": ["2025-04-15"],
            "price": [101.0],
            "volume": [2.0],
        }
    )
    figures = _build_figures(
        {
            "manifest": {
                "factor_name": FACTOR,
                "config": {"strategy": _ecdf_audit_config()},
            },
            "market": market,
            "snapshots": pd.DataFrame(),
            "activity": pd.DataFrame(),
            "accounts": pd.DataFrame(),
            "hourly_market": hourly,
            "hourly_drawdown": pd.DataFrame(),
            "price_basis": "raw",
        },
        metrics,
    )

    assert "short threshold" in figures[0]
    assert "long threshold" in figures[0]
    assert "q10" not in figures[0]
    assert "q90" not in figures[0]

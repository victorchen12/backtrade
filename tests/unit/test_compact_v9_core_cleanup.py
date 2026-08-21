from __future__ import annotations

from datetime import datetime, timedelta, timezone
from dataclasses import replace

import pytest

from backtrade.config.schema import BacktradeConfig
from backtrade.data.replay import MarketReplay
from backtrade.order_match.maker import MakerMatcher
from backtrade.position.single_lot import SingleLotAccount
from backtrade.simulation.events import FillEvent, MarketTick, MatchView, Order, StrategyView
from backtrade.simulation.compact_v9 import _audit_signal_execution
from backtrade.simulation.compact_v9_runner import CompactV9Runner
from backtrade.simulation.state import MatchMode, OrderSide, OrderStatus, OrderType, TimeInForce
from backtrade.strategies.signed_factor import SignedFactorStrategy


TS = datetime(2026, 1, 5, 9, 0, tzinfo=timezone.utc)


def runner_config(*, max_ticks: int = 7) -> BacktradeConfig:
    return BacktradeConfig.model_validate(
        {
            "data": {
                "source": "future_l2",
                "product": "ag",
                "max_ticks": max_ticks,
                "factor_grid_mode": "decision_grid",
                "eof_is_day_end": False,
            },
            "execution": {"latency_ms": 5, "day_end_flatten_window_ms": 0},
            "match": {"mode": "maker"},
            "limit_reference": {"mode": "disabled"},
            "initial_cash": 1_000_000,
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


def decision_tick(
    ts: datetime,
    seq: int,
    score: float,
    *,
    last_price: float = 100.0,
    vol_inc: int = 0,
    direction: OrderSide | None = None,
) -> MarketTick:
    return replace(
        tick(ts, seq),
        last_price=last_price,
        vol_inc=vol_inc,
        amount_inc=last_price * vol_inc * 15.0,
        factors={"active_factor": score},
        factor_decision=True,
        factor_source_ts=ts,
        factor_age_ms=0.0,
        trade_direction=direction,
        trade_direction_source=(
            "ask_touch"
            if direction is OrderSide.BUY
            else "bid_touch"
            if direction is OrderSide.SELL
            else None
        ),
        trade_direction_confidence="high" if direction is not None else None,
        trade_direction_quality="normal" if direction is not None else None,
    )


def tick(ts: datetime, seq: int) -> MarketTick:
    return MarketTick(
        product="ag",
        contract="AG2604",
        tick_ts=ts,
        last_price=100.0,
        bid_prices=(99.0, 98.0, 97.0, 96.0, 95.0),
        bid_qtys=(5, 5, 5, 5, 5),
        ask_prices=(101.0, 102.0, 103.0, 104.0, 105.0),
        ask_qtys=(5, 5, 5, 5, 5),
        trading_day="2026-01-05",
        source_seq=seq,
        factors={"active_factor": 0.0},
    )


def maker_order(side: OrderSide = OrderSide.BUY, price: float = 99.0) -> Order:
    return Order.create(
        contract="AG2604",
        product="ag",
        side=side,
        qty=1,
        limit_price=price,
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.GTC,
        match_mode=MatchMode.MAKER,
        decision_ts=TS,
        reason_code="signal",
    )


def match_view(**kwargs) -> MatchView:
    values = dict(
        product="ag",
        contract="AG2604",
        tick_ts=TS,
        bid_prices=(99.0, 98.0, 97.0, 96.0, 95.0),
        bid_qtys=(5, 5, 5, 5, 5),
        ask_prices=(101.0, 102.0, 103.0, 104.0, 105.0),
        ask_qtys=(5, 5, 5, 5, 5),
        mid=100.0,
        spread=2.0,
        last_price=99.0,
        vol_inc=0,
        trade_direction=OrderSide.SELL,
        trade_direction_source="bid_touch",
        trade_direction_confidence="high",
        trade_direction_quality="normal",
        source_seq=1,
        session_id="day",
        trading_day="2026-01-05",
    )
    values.update(kwargs)
    return MatchView(**values)


def test_zero_window_does_not_release_only_first_tick() -> None:
    rows = list(MarketReplay([tick(TS, 1), tick(TS + timedelta(seconds=1), 2)], closing_window_ms=0))
    assert len(rows) == 2
    assert not any(view.is_day_closing for _, view, _ in rows)


def test_maker_cancels_resting_close_after_five_second_signal_reversion() -> None:
    ticks = [
        decision_tick(TS + timedelta(seconds=5 * index), index + 1, score, **trade)
        for index, (score, trade) in enumerate(
            [
                (1.0, {}),
                (1.0, {}),
                (1.0, {"last_price": 99.0, "vol_inc": 5, "direction": OrderSide.SELL}),
                (-1.0, {}),
                (-1.0, {}),
                (1.0, {}),
                (1.0, {"last_price": 101.0, "vol_inc": 5, "direction": OrderSide.BUY}),
            ]
        )
    ]

    result = CompactV9Runner(runner_config(), ticks).run()

    close_orders = [
        order
        for order in result.orders
        if order.created.decision_ts == TS + timedelta(seconds=15) and order.created.close_qty == 1
    ]
    assert len(close_orders) == 1
    close_order = close_orders[0]
    assert close_order.status is OrderStatus.CANCEL
    assert any(
        change.reason_code == "target_changed_cancel_active_order"
        for change in close_order.status_history
    )
    assert not any(fill.order_id == close_order.order_id for fill in result.fills)


def test_zero_signal_cancels_pending_open_and_holds_actual_flat() -> None:
    ticks = [
        decision_tick(TS + timedelta(seconds=5 * index), index + 1, score, **trade)
        for index, (score, trade) in enumerate(
            [
                (1.0, {}),
                (0.0, {}),
                (0.0, {"last_price": 99.0, "vol_inc": 5, "direction": OrderSide.SELL}),
            ]
        )
    ]

    result = CompactV9Runner(runner_config(max_ticks=3), ticks).run()

    open_order = next(order for order in result.orders if order.created.decision_ts == TS)
    assert open_order.created.open_qty == 1
    assert open_order.status is OrderStatus.CANCEL
    assert not any(fill.order_id == open_order.order_id for fill in result.fills)


def test_opposite_signal_while_flat_replaces_pending_open_order() -> None:
    ticks = [
        decision_tick(TS + timedelta(seconds=5 * index), index + 1, score)
        for index, score in enumerate([1.0, -1.0, -1.0])
    ]

    result = CompactV9Runner(runner_config(max_ticks=3), ticks).run()

    buy_order = next(order for order in result.orders if order.created.decision_ts == TS)
    assert buy_order.status is OrderStatus.CANCEL
    replacement = [
        order
        for order in result.orders
        if order.created.decision_ts == TS + timedelta(seconds=5)
    ]
    assert len(replacement) == 1
    assert replacement[0].created.side is OrderSide.SELL
    assert replacement[0].created.open_qty == 1


def test_unbounded_eof_is_day_end_only_when_explicit() -> None:
    rows = list(MarketReplay([tick(TS, 1), tick(TS + timedelta(seconds=1), 2)], closing_window_ms=0, eof_is_day_end=True))
    assert rows[-1][1].is_last_tick_of_day is True
    bounded = list(MarketReplay([tick(TS, 1), tick(TS + timedelta(seconds=1), 2)], closing_window_ms=0, eof_is_day_end=False))
    assert bounded[-1][1].is_last_tick_of_day is False


@pytest.mark.parametrize(
    "payload",
    [
        {"execution": {"latency_ms": -1}},
        {"execution": {"day_end_flatten_window_ms": -1}},
        {"data": {"max_ticks": 0}},
    ],
)
def test_invalid_limits_are_configuration_errors(payload) -> None:
    with pytest.raises(ValueError):
        BacktradeConfig.model_validate(payload)


def test_signed_factor_strategy_holds_zero_and_flattens_reversal_first() -> None:
    strategy = SignedFactorStrategy()

    def view(score: float) -> StrategyView:
        return StrategyView(
            product="ag",
            contract="AG2604",
            tick_ts=TS,
            mid=100.0,
            factors={"active_factor": score},
            factor_decision=True,
            factor_source_ts=TS,
            factor_age_ms=0.0,
        )

    assert strategy.on_decision(view(0.0), 1).target_qty == 1
    flat = strategy.on_decision(view(-1.0), 1)
    assert (flat.target_qty, flat.reduce_only, flat.factor_semantics_version) == (0, True, "signed_factor_v1")
    assert strategy.on_decision(view(-1.0), 0).target_qty == -1


def test_maker_equality_consumes_queue_before_fill() -> None:
    matcher = MakerMatcher()
    order = maker_order()
    matcher.match(order, match_view())
    fills, events = matcher.match(
        order,
        match_view(tick_ts=TS + timedelta(milliseconds=1), source_seq=2, last_price=99.0, vol_inc=5),
    )
    assert len(fills) == 1
    assert fills[0].price == 99.0
    assert events[0]["reason_code"] == "maker_queue_reached"
    assert events[0]["trade_price"] == 99.0


@pytest.mark.parametrize("field", ["is_stale", "is_anomaly", "side_ambiguous_flag"])
def test_bad_market_observation_freezes_queue(field: str) -> None:
    matcher = MakerMatcher()
    order = maker_order()
    matcher.match(order, match_view())
    bad = match_view(tick_ts=TS + timedelta(milliseconds=1), source_seq=2, bid_qtys=(0, 5, 5, 5, 5), vol_inc=5)
    setattr(bad, field, True)
    fills, events = matcher.match(order, bad)
    assert fills == []
    assert order.queue_ahead == 5
    assert events[0]["event_type"] in {"rebaseline", "progress"}


def test_first_arrival_not_l1_or_crossing_is_rejected() -> None:
    for view in (
        match_view(bid_prices=(98.0, 97.0, 96.0, 95.0, 94.0)),
        match_view(ask_prices=(99.0, 100.0, 101.0, 102.0, 103.0)),
    ):
        order = maker_order()
        fills, events = MakerMatcher().match(order, view)
        assert fills == []
        assert order.status is OrderStatus.REJECTED
        assert events[0]["event_type"] == "rejected"


def test_fill_net_pnl_rows_conserve_cash() -> None:
    fees = type(
        "Fees",
        (),
        {
            "open": type("Fee", (), {"mode": "per_lot", "value": 1.0})(),
            "close": type("Fee", (), {"mode": "per_lot", "value": 2.0})(),
            "close_today": type("Fee", (), {"mode": "per_lot", "value": 3.0})(),
        },
    )()
    rule = type("Rule", (), {"multiplier": 10.0, "fee": fees})()
    account = SingleLotAccount(1000.0)
    opened = account.apply_fill(FillEvent("o", "AG2604", "ag", OrderSide.BUY, TS, 100.0, 1, 1, 0, 0.0, 0.0, "open", trading_day="2026-01-05"), rule)
    closed = account.apply_fill(FillEvent("c", "AG2604", "ag", OrderSide.SELL, TS + timedelta(seconds=1), 101.0, 1, 0, 1, 0.0, 0.0, "close", trading_day="2026-01-06"), rule)
    assert opened.net_pnl == opened.cash_delta == -1.0
    assert closed.net_pnl == closed.cash_delta == 8.0
    assert opened.net_pnl + closed.net_pnl == account.cash - 1000.0


def test_audit_rejects_fill_after_reverted_signed_target():
    activity = [
        {"event_seq": 1, "record_type": "target", "target_seq": 1, "product": "ag", "event_ts": TS, "factor_score": 1.0, "factor_decision": True, "factor_source_ts": TS, "factor_age_ms": 0.0, "target_qty": 1},
        {"event_seq": 2, "record_type": "target", "target_seq": 2, "product": "ag", "event_ts": TS + timedelta(seconds=5), "factor_score": -1.0, "factor_decision": True, "factor_source_ts": TS + timedelta(seconds=5), "factor_age_ms": 0.0, "target_qty": -1},
        {"event_seq": 3, "record_type": "fill", "fill_seq": 1, "product": "ag", "event_ts": TS + timedelta(seconds=6), "boundary_reason": None},
    ]
    accounts = [{"fill_seq": 1, "position_before": 0, "position_after": 1}]

    errors = _audit_signal_execution(activity, accounts)

    assert any("does not move toward latest target" in error for error in errors)

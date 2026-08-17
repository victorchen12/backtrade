from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from backtrade.config.schema import BacktradeConfig
from backtrade.data.replay import MarketReplay
from backtrade.order_match.maker import MakerMatcher
from backtrade.position.single_lot import SingleLotAccount
from backtrade.simulation.events import FillEvent, MarketTick, MatchView, Order, StrategyView
from backtrade.simulation.state import MatchMode, OrderSide, OrderStatus, OrderType, TimeInForce
from backtrade.strategies.ofi import OFISignStrategy


TS = datetime(2026, 1, 5, 9, 0, tzinfo=timezone.utc)


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


def test_ofi_sign_strategy_holds_zero_and_flattens_reversal_first() -> None:
    strategy = OFISignStrategy()

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
    assert (flat.target_qty, flat.reduce_only, flat.factor_semantics_version) == (0, True, "ofi_sign_v1")
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

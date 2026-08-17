from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from backtrade.simulation.events import FillEvent, MatchView, Order, OrderExecutionBit
from backtrade.simulation.state import OrderSide, OrderStatus, TERMINAL_STATUSES


class ProbQueueModel:
    """Deterministic MBP expected-queue estimator; it does not rebuild FIFO."""

    @staticmethod
    def ahead_probability(ahead: float, behind: float) -> float:
        ahead = max(float(ahead), 0.0)
        behind = max(float(behind), 0.0)
        if ahead == 0.0 and behind == 0.0:
            return 0.0
        numerator = math.log1p(ahead)
        denominator = numerator + math.log1p(behind)
        return 0.0 if denominator == 0.0 else numerator / denominator


@dataclass
class _MakerState:
    previous_visible: float
    queue_ahead: float
    last_tick_id: tuple[Any, ...] | None = None


class MakerMatcher:
    """Single-lot post-only maker matcher using conservative MBP evidence."""

    def __init__(self) -> None:
        self._states: dict[str, _MakerState] = {}

    @staticmethod
    def _tick_id(view: MatchView) -> tuple[Any, ...]:
        return (
            view.session_id, view.trading_day, view.tick_ts, view.source_seq,
            view.last_price, view.vol_inc, view.bid_prices, view.bid_qtys,
            view.ask_prices, view.ask_qtys, view.is_stale, view.is_anomaly,
            view.side_ambiguous_flag,
        )

    @staticmethod
    def _same_side_l1(order: Order, view: MatchView) -> tuple[bool, float, int]:
        if order.created.side is OrderSide.BUY:
            price, qty = view.bid_prices[0], view.bid_qtys[0]
        else:
            price, qty = view.ask_prices[0], view.ask_qtys[0]
        same = math.isfinite(float(price)) and price > 0 and price == order.created.limit_price
        return same, float(price), max(int(qty), 0)

    @staticmethod
    def _would_take(order: Order, view: MatchView) -> bool:
        if order.created.side is OrderSide.BUY:
            return view.ask_prices[0] > 0 and view.ask_prices[0] <= float(order.created.limit_price or 0.0)
        return view.bid_prices[0] > 0 and view.bid_prices[0] >= float(order.created.limit_price or 0.0)

    @staticmethod
    def _quality_ok(view: MatchView) -> bool:
        quality = view.trade_direction_quality or view.direction_quality
        return not bool(view.is_stale or view.is_anomaly or view.side_ambiguous_flag) and str(quality or "").lower() not in {
            "low", "invalid", "unknown",
        }

    def _high_confidence_opposite(self, order: Order, view: MatchView) -> bool:
        expected_direction = OrderSide.SELL if order.created.side is OrderSide.BUY else OrderSide.BUY
        expected_source = "bid_touch" if order.created.side is OrderSide.BUY else "ask_touch"
        direction = view.trade_direction
        source = view.trade_direction_source or view.direction_source
        confidence = view.trade_direction_confidence or view.direction_confidence
        return (
            direction is expected_direction
            and str(source or "").lower() == expected_source
            and str(confidence or "").lower() == "high"
            and self._quality_ok(view)
        )

    @staticmethod
    def _reject(order: Order, reason: str, view: MatchView) -> None:
        if order.status is OrderStatus.CREATED:
            order.transition(OrderStatus.REJECTED, reason, view.tick_ts)
        elif order.status is OrderStatus.SUBMITTED:
            order.transition(OrderStatus.REJECTED, reason, view.tick_ts)

    @staticmethod
    def _cancel(order: Order, reason: str, view: MatchView) -> None:
        if order.status is OrderStatus.SUBMITTED:
            order.transition(OrderStatus.ACCEPTED, "auto_accepted_before_cancel", view.tick_ts)
        if order.status in {OrderStatus.ACCEPTED, OrderStatus.QUEUED, OrderStatus.PARTIAL}:
            order.transition(OrderStatus.CANCEL, reason, view.tick_ts)

    def _event(
        self,
        order: Order,
        view: MatchView,
        event_type: str,
        reason: str,
        before: float | None,
        after: float | None,
        *,
        depth_before: float | None = None,
        depth_after: float | None = None,
        same_trade_qty: int = 0,
        observed_opposite_trade_qty: float = 0.0,
        trade_through_qty: float = 0.0,
        probability_ahead: float | None = None,
        queue_behind_before: float | None = None,
        residual_depth_decrease: float = 0.0,
        attributed_trade_qty: float = 0.0,
    ) -> dict[str, Any]:
        opposite_price = view.ask_prices[0] if order.created.side is OrderSide.BUY else view.bid_prices[0]
        opposite_qty = view.ask_qtys[0] if order.created.side is OrderSide.BUY else view.bid_qtys[0]
        return {
            "event_type": event_type,
            "event_ts": view.tick_ts,
            "order_id": order.order_id,
            "product": order.product,
            "contract": order.contract,
            "side": order.created.side.value,
            "price": order.created.limit_price,
            "trade_price": view.last_price,
            "opposite_l1_price": opposite_price,
            "opposite_l1_qty": opposite_qty,
            "queue_ahead_before": before,
            "queue_ahead_after": after,
            "depth_before": depth_before,
            "depth_after": depth_after,
            "same_price_trade_qty": same_trade_qty,
            "observed_opposite_trade_qty": observed_opposite_trade_qty,
            "trade_through_qty": trade_through_qty,
            "probability_ahead": probability_ahead,
            "queue_behind_before": queue_behind_before,
            "residual_depth_decrease": residual_depth_decrease,
            "attributed_same_price_trade_qty": attributed_trade_qty,
            "observed_last_price": view.last_price,
            "direction_source": view.trade_direction_source or view.direction_source,
            "direction_confidence": view.trade_direction_confidence or view.direction_confidence,
            "data_quality": "normal" if self._quality_ok(view) else "invalid",
            "source_seq": view.source_seq,
            "session_id": view.session_id,
            "trading_day": view.trading_day,
            "reason_code": reason,
        }

    def _fill(self, order: Order, view: MatchView, reason: str) -> FillEvent:
        order.ensure_queued(view.tick_ts)
        open_qty, close_qty = order.split_next_fill(1)
        bit = OrderExecutionBit(
            order_id=order.order_id, fill_ts=view.tick_ts, price=float(order.created.limit_price), qty=1,
            open_qty=open_qty, close_qty=close_qty, commission=0.0, pnl=0.0,
            position_snapshot={"net_qty": None}, reason_code=reason,
        )
        order.add_execution_bit(bit)
        return FillEvent(
            order_id=order.order_id, contract=order.contract, product=order.product,
            side=order.created.side, fill_ts=view.tick_ts, price=float(order.created.limit_price),
            qty=1, open_qty=open_qty, close_qty=close_qty, commission=0.0, pnl=0.0,
            reason_code=reason, trading_day=view.trading_day, session_id=view.session_id,
            source_seq=view.source_seq, match_mode="maker", liquidity_source="mbp_expected_queue",
            reduce_only=order.created.reduce_only,
        )

    def retire_order(self, order_id: str) -> None:
        self._states.pop(order_id, None)

    def cancel_event(self, order: Order, reason: str, ts, view: MatchView | None = None) -> dict[str, Any]:
        state = self._states.get(order.order_id)
        before = state.queue_ahead if state is not None else order.queue_ahead
        depth = state.previous_visible if state is not None else order.queue_reference_qty
        self.retire_order(order.order_id)
        if view is not None:
            return self._event(order, view, "cancel", reason, before, None, depth_before=depth)
        return {
            "event_type": "cancel", "event_ts": ts, "order_id": order.order_id,
            "product": order.product, "contract": order.contract, "side": order.created.side.value,
            "price": order.created.limit_price, "trade_price": None, "opposite_l1_price": None,
            "opposite_l1_qty": None, "queue_ahead_before": before, "queue_ahead_after": None,
            "depth_before": depth, "depth_after": None, "same_price_trade_qty": 0,
            "observed_opposite_trade_qty": 0.0, "trade_through_qty": 0.0,
            "probability_ahead": None, "queue_behind_before": None, "residual_depth_decrease": 0.0,
            "attributed_same_price_trade_qty": 0.0, "observed_last_price": None,
            "direction_source": None, "direction_confidence": None, "data_quality": "unknown",
            "source_seq": None, "session_id": None, "trading_day": order.trading_day,
            "reason_code": reason,
        }

    def match(self, order: Order, view: MatchView) -> tuple[list[FillEvent], list[dict[str, Any]]]:
        if order.created.qty != 1:
            raise ValueError("maker orders are single-lot only")
        if order.created.limit_price is None or order.created.limit_price <= 0:
            raise ValueError("maker order requires a positive L1 limit price")
        if order.status in TERMINAL_STATUSES or order.executed.remaining_qty <= 0:
            return [], []
        tick_id = self._tick_id(view)
        state = self._states.get(order.order_id)
        if state is not None and state.last_tick_id == tick_id:
            return [], []
        same_l1, _price, visible = self._same_side_l1(order, view)
        events: list[dict[str, Any]] = []

        if state is None:
            if same_l1 and visible <= 0:
                self._reject(order, "maker_invalid_l1", view)
                return [], [self._event(order, view, "rejected", "maker_invalid_l1", None, None, depth_after=visible)]
            if self._would_take(order, view):
                self._reject(order, "post_only_would_take", view)
                return [], [self._event(order, view, "rejected", "post_only_would_take", None, None)]
            if not same_l1:
                self._reject(order, "maker_not_at_l1", view)
                return [], [self._event(order, view, "rejected", "maker_not_at_l1", None, None)]
            order.ensure_queued(view.tick_ts, "maker_enqueued")
            order.queue_ahead = float(visible)
            order.queue_reference_qty = visible
            order.queue_price_present = True
            self._states[order.order_id] = _MakerState(float(visible), float(visible), tick_id)
            return [], [self._event(order, view, "enqueue", "maker_enqueued", None, float(visible), depth_after=visible)]

        state.last_tick_id = tick_id
        if not same_l1:
            before = state.queue_ahead
            self._cancel(order, "maker_not_at_l1", view)
            self.retire_order(order.order_id)
            return [], [self._event(order, view, "cancel", "maker_not_at_l1", before, None, depth_before=state.previous_visible)]

        current_visible = float(visible)
        if not self._quality_ok(view):
            before = state.queue_ahead
            state.previous_visible = current_visible
            order.queue_reference_qty = visible
            return [], [self._event(order, view, "rebaseline", "maker_invalid_observation", before, before, depth_before=current_visible, depth_after=current_visible)]

        high = self._high_confidence_opposite(order, view)
        observed_trade_qty = max(int(view.vol_inc), 0)
        strict_through = high and observed_trade_qty > 0 and view.last_price is not None and (
            (order.created.side is OrderSide.BUY and view.last_price < order.created.limit_price)
            or (order.created.side is OrderSide.SELL and view.last_price > order.created.limit_price)
        )
        if strict_through:
            before = state.queue_ahead
            fill = self._fill(order, view, "maker_trade_through")
            event = self._event(order, view, "fill", "maker_trade_through", before, 0.0, depth_before=state.previous_visible, depth_after=current_visible, observed_opposite_trade_qty=observed_trade_qty, trade_through_qty=observed_trade_qty)
            self.retire_order(order.order_id)
            return [fill], [event]

        same_trade_qty = observed_trade_qty if high and view.last_price is not None and view.last_price == order.created.limit_price else 0
        previous_visible = state.previous_visible
        queue_before = state.queue_ahead
        queue_behind_before = max(previous_visible - queue_before, 0.0)
        consumed = min(queue_before, float(same_trade_qty))
        queue_after = max(queue_before - consumed, 0.0)
        decrease = max(previous_visible - current_visible, 0.0)
        attributed = min(decrease, float(same_trade_qty))
        residual = max(decrease - attributed, 0.0)
        probability = None
        if residual > 0:
            probability = ProbQueueModel.ahead_probability(queue_after, queue_behind_before)
            queue_after = max(queue_after - residual * probability, 0.0)
        state.previous_visible = current_visible
        state.queue_ahead = queue_after
        order.queue_ahead = queue_after
        order.queue_reference_qty = visible
        order.queue_price_present = True
        event_args = dict(depth_before=previous_visible, depth_after=current_visible, same_trade_qty=same_trade_qty, observed_opposite_trade_qty=observed_trade_qty, probability_ahead=probability, queue_behind_before=queue_behind_before, residual_depth_decrease=residual, attributed_trade_qty=attributed)
        if same_trade_qty >= queue_before and same_trade_qty > 0:
            fill = self._fill(order, view, "maker_queue_reached")
            event = self._event(order, view, "fill", "maker_queue_reached", queue_before, 0.0, **event_args)
            self.retire_order(order.order_id)
            return [fill], [event]
        if queue_after != queue_before or decrease > 0:
            reason = "maker_queue_progress" if same_trade_qty else "maker_conservative_depth_progress"
            return [], [self._event(order, view, "progress", reason, queue_before, queue_after, **event_args)]
        return [], []


__all__ = ["MakerMatcher", "ProbQueueModel"]

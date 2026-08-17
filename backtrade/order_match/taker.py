from __future__ import annotations

import math
from backtrade.simulation.events import FillEvent, MatchView, Order, OrderExecutionBit
from backtrade.simulation.state import OrderSide, TERMINAL_STATUSES


class MissingL1Error(ValueError):
    pass


class TakerMatcher:
    """Strict one-lot taker matcher at the arrival tick's opposite L1."""

    def match(self, order: Order, view: MatchView) -> tuple[list[FillEvent], list[dict]]:
        if order.created.qty != 1:
            raise ValueError("taker orders are single-lot only")
        if order.status in TERMINAL_STATUSES or order.executed.remaining_qty <= 0:
            return [], []
        if order.created.side is OrderSide.BUY:
            price, available = view.ask_prices[0], view.ask_qtys[0]
        else:
            price, available = view.bid_prices[0], view.bid_qtys[0]
        if not math.isfinite(float(price)) or price <= 0 or int(available) <= 0:
            raise MissingL1Error(f"valid opposite L1 is required for taker order {order.order_id}")
        order.ensure_queued(view.tick_ts, "taker_arrived")
        open_qty, close_qty = order.split_next_fill(1)
        bit = OrderExecutionBit(
            order_id=order.order_id,
            fill_ts=view.tick_ts,
            price=float(price),
            qty=1,
            open_qty=open_qty,
            close_qty=close_qty,
            commission=0.0,
            pnl=0.0,
            position_snapshot={"net_qty": None},
            reason_code="taker_l1_fill",
        )
        order.add_execution_bit(bit)
        fill = FillEvent(
            order_id=order.order_id,
            contract=order.contract,
            product=order.product,
            side=order.created.side,
            fill_ts=view.tick_ts,
            price=float(price),
            qty=1,
            open_qty=open_qty,
            close_qty=close_qty,
            commission=0.0,
            pnl=0.0,
            reason_code="taker_l1_fill",
            trading_day=view.trading_day,
            session_id=view.session_id,
            source_seq=view.source_seq,
            match_mode="taker",
            liquidity_source="arrival_l1",
            reduce_only=order.created.reduce_only,
        )
        return [fill], []


__all__ = ["MissingL1Error", "TakerMatcher"]

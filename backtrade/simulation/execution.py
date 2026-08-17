from __future__ import annotations

from math import isfinite

from backtrade.config.schema import BacktradeConfig
from backtrade.simulation.events import Order, PortfolioTarget
from backtrade.simulation.state import MatchMode, OrderSide, OrderType, TimeInForce


def normalize_target_qty(requested_target_qty: int | float | None) -> int:
    if requested_target_qty is None:
        return 0
    value = float(requested_target_qty)
    if not isfinite(value):
        raise ValueError("target quantity must be finite")
    return 1 if value > 0 else -1 if value < 0 else 0


class ExecutionEngine:
    def __init__(self, cfg: BacktradeConfig):
        self.mode = MatchMode(cfg.match.mode)

    def orders_from_target(
        self,
        target: PortfolioTarget,
        current_qty: int,
        best_bid: float,
        best_ask: float,
        *,
        actual_qty: int | None = None,
        trading_day: str | None = None,
        target_seq: int | None = None,
    ) -> list[Order]:
        position_qty = int(current_qty if actual_qty is None else actual_qty)
        if abs(position_qty) > 1:
            raise ValueError("single-lot execution requires current position in {-1, 0, +1}")
        desired = normalize_target_qty(target.target_qty)
        current_sign = normalize_target_qty(position_qty)
        if desired == current_sign:
            return []

        execution_target = 0 if current_sign and desired and current_sign != desired else desired
        delta = execution_target - position_qty
        if delta == 0:
            return []
        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        close_qty = 1 if current_sign and (execution_target == 0 or current_sign != (1 if delta > 0 else -1)) else 0
        open_qty = 1 - close_qty
        reason_code = (
            "reversal_flatten_before_open"
            if current_sign and desired and current_sign != desired
            else target.reason_code
        )
        limit_price = None
        if self.mode is MatchMode.MAKER:
            limit_price = best_bid if side is OrderSide.BUY else best_ask
        order = Order.create(
            contract=target.contract,
            product=target.product,
            side=side,
            qty=1,
            limit_price=limit_price,
            order_type=OrderType.LIMIT,
            time_in_force=TimeInForce.GTC,
            match_mode=self.mode,
            decision_ts=target.decision_ts,
            reason_code=reason_code,
            open_qty=open_qty,
            close_qty=close_qty,
            reduce_only=bool(target.reduce_only or close_qty),
            trading_day=trading_day,
            target_seq=target_seq if target_seq is not None else target.target_seq,
        )
        order.decision_reference_price = limit_price if self.mode is MatchMode.MAKER else (best_ask if side is OrderSide.BUY else best_bid)
        order.latency_exempt_reason = target.latency_exempt_reason
        order.boundary_reason = target.boundary_reason
        return [order]


__all__ = ["ExecutionEngine", "normalize_target_qty"]


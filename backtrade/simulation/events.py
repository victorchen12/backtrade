from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from backtrade.simulation.state import (
    ALLOWED_TRANSITIONS,
    InvalidOrderTransition,
    MatchMode,
    OrderSide,
    OrderStatus,
    OrderType,
    TERMINAL_STATUSES,
    TimeInForce,
)


@dataclass(slots=True)
class MarketTick:
    product: str
    contract: str
    tick_ts: datetime
    last_price: float
    bid_prices: tuple[float, float, float, float, float]
    bid_qtys: tuple[int, int, int, int, int]
    ask_prices: tuple[float, float, float, float, float]
    ask_qtys: tuple[int, int, int, int, int]
    vol_inc: int = 0
    amount_inc: float = 0.0
    factors: dict[str, float] = field(default_factory=dict)
    trading_day: int | str | None = None
    price_limit_up: float | None = None
    price_limit_down: float | None = None
    price_limit_reference_price: float | None = None
    price_limit_reference_source: str | None = None
    price_limit_rule_version: str | None = None
    source_seq: int = 0
    session_id: str | None = None
    cancel_bid_tick: float = 0.0
    cancel_ask_tick: float = 0.0
    cancel_total_tick: float = 0.0
    cancel_imbalance_tick: float = 0.0
    cancel_reliability_score: float = 0.0
    stale_ms: float = 0.0
    cancel_event_flag: int = 0
    quote_change_flag: int = 0
    side_ambiguous_flag: int = 0
    level_shift_flag: int = 0
    is_anomaly: bool = False
    is_stale: bool = False
    trade_direction: OrderSide | None = None
    trade_direction_source: str | None = None
    trade_direction_confidence: str | None = None
    direction_source: str | None = None
    direction_confidence: str | None = None
    trade_direction_quality: str | None = None
    direction_quality: str | None = None
    factor_decision: bool = True
    factor_source_ts: datetime | None = None
    factor_age_ms: float | None = None

    @property
    def bid1(self) -> float:
        return self.bid_prices[0]

    @property
    def ask1(self) -> float:
        return self.ask_prices[0]

    @property
    def mid(self) -> float:
        return (self.bid1 + self.ask1) / 2

    @property
    def spread(self) -> float:
        return self.ask1 - self.bid1


@dataclass(slots=True)
class StrategyView:
    product: str
    contract: str
    tick_ts: datetime
    mid: float
    factors: dict[str, float]
    trading_day: str | None = None
    factor_decision: bool = True
    factor_source_ts: datetime | None = None
    factor_age_ms: float | None = None
    is_day_closing: bool = False
    is_last_tick_of_day: bool = False
    is_last_tick_of_contract: bool = False
    seconds_to_day_end: float | None = None


@dataclass(slots=True)
class MatchView:
    product: str
    contract: str
    tick_ts: datetime
    bid_prices: tuple[float, float, float, float, float]
    bid_qtys: tuple[int, int, int, int, int]
    ask_prices: tuple[float, float, float, float, float]
    ask_qtys: tuple[int, int, int, int, int]
    mid: float
    spread: float
    last_price: float | None = None
    vol_inc: int = 0
    trade_direction: OrderSide | None = None
    source_seq: int = 0
    session_id: str | None = None
    cancel_bid_tick: float = 0.0
    cancel_ask_tick: float = 0.0
    cancel_total_tick: float = 0.0
    cancel_imbalance_tick: float = 0.0
    cancel_reliability_score: float = 0.0
    stale_ms: float = 0.0
    cancel_event_flag: int = 0
    quote_change_flag: int = 0
    side_ambiguous_flag: int = 0
    level_shift_flag: int = 0
    is_anomaly: bool = False
    is_stale: bool = False
    trading_day: str | None = None
    trade_direction_source: str | None = None
    trade_direction_confidence: str | None = None
    direction_source: str | None = None
    direction_confidence: str | None = None
    trade_direction_quality: str | None = None
    direction_quality: str | None = None
    price_limit_up: float | None = None
    price_limit_down: float | None = None
    price_limit_reference_price: float | None = None
    price_limit_reference_source: str | None = None
    price_limit_rule_version: str | None = None


@dataclass(slots=True)
class PortfolioTarget:
    product: str
    contract: str
    decision_ts: datetime
    target_qty: int
    status: str = "created"
    reduce_only: bool = False
    risk_state: str = "normal"
    reason_code: str = "strategy_signal"
    factor_name: str = "ofi_cks_best_level_5s"
    factor_score: float | None = None
    factor_semantics_version: str = "ofi_sign_v1"
    factor_decision: bool = True
    factor_source_ts: datetime | None = None
    factor_age_ms: float | None = None
    position_before: int | None = None
    target_seq: int | None = None
    latency_exempt_reason: str | None = None
    boundary_reason: str | None = None


@dataclass(slots=True)
class OrderCreated:
    decision_ts: datetime
    qty: int
    side: OrderSide
    limit_price: float | None
    order_type: OrderType
    time_in_force: TimeInForce
    match_mode: MatchMode
    open_qty: int
    close_qty: int
    reduce_only: bool
    reason_code: str


@dataclass(slots=True)
class OrderExecuted:
    filled_qty: int = 0
    remaining_qty: int = 0
    avg_price: float = 0.0
    commission: float = 0.0
    pnl: float = 0.0
    open_qty: int = 0
    close_qty: int = 0


@dataclass(slots=True)
class OrderStatusChange:
    from_status: OrderStatus | None
    to_status: OrderStatus
    ts: datetime
    reason_code: str


@dataclass(slots=True)
class OrderExecutionBit:
    order_id: str
    fill_ts: datetime
    price: float
    qty: int
    open_qty: int
    close_qty: int
    commission: float
    pnl: float
    position_snapshot: dict[str, Any]
    reason_code: str
    close_today_qty: int = 0
    execution_bit_seq: int | None = None
    fill_seq: int | None = None
    order_seq: int | None = None
    target_seq: int | None = None


@dataclass(slots=True)
class FillEvent:
    order_id: str
    contract: str
    product: str
    side: OrderSide
    fill_ts: datetime
    price: float
    qty: int
    open_qty: int
    close_qty: int
    commission: float
    pnl: float
    reason_code: str
    close_today_qty: int = 0
    fee_by_open_close_close_today: dict[str, float] = field(default_factory=dict)
    trading_day: str | None = None
    session_id: str | None = None
    source_seq: int | None = None
    match_mode: str | None = None
    fill_seq: int | None = None
    execution_bit_seq: int | None = None
    order_seq: int | None = None
    target_seq: int | None = None
    latency_exempt_reason: str | None = None
    boundary_reason: str | None = None
    contract_multiplier: float = 1.0
    maker_taker_role: str | None = None
    liquidity_source: str | None = None
    reduce_only: bool = False


@dataclass(slots=True)
class BoundaryEvent:
    ts: datetime
    product: str
    contract: str
    event_type: str
    severity: str
    reason_code: str
    order_id: str = ""
    detail: str = ""
    price: float | None = None
    qty: int | None = None
    latency_exempt_reason: str | None = None
    boundary_reason: str | None = None


@dataclass
class Order:
    order_id: str
    product: str
    contract: str
    created: OrderCreated
    executed: OrderExecuted
    status: OrderStatus
    status_history: list[OrderStatusChange]
    execution_bits: list[OrderExecutionBit] = field(default_factory=list)
    arrival_ts: datetime | None = None
    scheduled_arrival_ts: datetime | None = None
    actual_arrival_ts: datetime | None = None
    decision_reference_price: float | None = None
    sequence: int = 0
    queue_ahead: float | None = None
    queue_reference_qty: float | None = None
    queue_price_present: bool | None = None
    latency_exempt_reason: str | None = None
    boundary_reason: str | None = None
    trading_day: str | None = None
    target_seq: int | None = None

    @classmethod
    def create(
        cls,
        contract: str,
        product: str,
        side: OrderSide,
        qty: int,
        limit_price: float | None,
        order_type: OrderType,
        time_in_force: TimeInForce,
        match_mode: MatchMode,
        decision_ts: datetime,
        reason_code: str,
        open_qty: int | None = None,
        close_qty: int | None = None,
        reduce_only: bool = False,
        trading_day: str | None = None,
        target_seq: int | None = None,
    ) -> "Order":
        if qty <= 0:
            raise ValueError("order qty must be positive")
        close_qty = 0 if close_qty is None else int(close_qty)
        open_qty = qty - close_qty if open_qty is None else int(open_qty)
        if open_qty < 0 or close_qty < 0 or open_qty + close_qty != qty:
            raise ValueError("open_qty + close_qty must equal qty")
        created = OrderCreated(
            decision_ts=decision_ts,
            qty=qty,
            side=side,
            limit_price=limit_price,
            order_type=order_type,
            time_in_force=time_in_force,
            match_mode=match_mode,
            open_qty=open_qty,
            close_qty=close_qty,
            reduce_only=reduce_only,
            reason_code=reason_code,
        )
        return cls(
            order_id=str(uuid.uuid4()),
            product=product,
            contract=contract,
            created=created,
            executed=OrderExecuted(remaining_qty=qty),
            status=OrderStatus.CREATED,
            status_history=[OrderStatusChange(None, OrderStatus.CREATED, decision_ts, reason_code)],
            trading_day=str(trading_day) if trading_day is not None else None,
            target_seq=target_seq,
        )

    def transition(self, to_status: OrderStatus, reason_code: str, ts: datetime | None = None) -> None:
        if not reason_code:
            raise ValueError("reason_code is required for every state transition")
        if self.status in TERMINAL_STATUSES:
            raise InvalidOrderTransition(f"{self.status.value} is terminal")
        if to_status not in ALLOWED_TRANSITIONS.get(self.status, set()):
            raise InvalidOrderTransition(f"cannot transition {self.status.value} -> {to_status.value}")
        change_ts = ts or self.created.decision_ts
        self.status_history.append(OrderStatusChange(self.status, to_status, change_ts, reason_code))
        self.status = to_status

    def ensure_queued(self, ts: datetime, reason_code: str = "queued_for_matching") -> None:
        if self.status is OrderStatus.CREATED:
            self.transition(OrderStatus.SUBMITTED, "auto_submitted_for_match", ts)
            self.transition(OrderStatus.ACCEPTED, "auto_accepted_for_match", ts)
            self.transition(OrderStatus.QUEUED, reason_code, ts)
        elif self.status is OrderStatus.SUBMITTED:
            self.transition(OrderStatus.ACCEPTED, "auto_accepted_for_match", ts)
            self.transition(OrderStatus.QUEUED, reason_code, ts)
        elif self.status is OrderStatus.ACCEPTED:
            self.transition(OrderStatus.QUEUED, reason_code, ts)

    def split_next_fill(self, qty: int) -> tuple[int, int]:
        close_left = self.created.close_qty - self.executed.close_qty
        close_qty = min(qty, max(close_left, 0))
        return qty - close_qty, close_qty

    def add_execution_bit(self, bit: OrderExecutionBit) -> None:
        if bit.qty <= 0:
            raise ValueError("execution bit qty must be positive")
        if bit.qty > self.executed.remaining_qty:
            raise ValueError("execution bit exceeds remaining qty")
        total_notional = self.executed.avg_price * self.executed.filled_qty + bit.price * bit.qty
        self.executed.filled_qty += bit.qty
        self.executed.remaining_qty -= bit.qty
        self.executed.avg_price = total_notional / self.executed.filled_qty
        self.executed.commission += bit.commission
        self.executed.pnl += bit.pnl
        self.executed.open_qty += bit.open_qty
        self.executed.close_qty += bit.close_qty
        self.execution_bits.append(bit)
        target = OrderStatus.COMPLETE if self.executed.remaining_qty == 0 else OrderStatus.PARTIAL
        self.transition(target, bit.reason_code, bit.fill_ts)


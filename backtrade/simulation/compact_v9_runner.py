from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from itertools import islice
from pathlib import Path
from typing import Any, Iterable

from backtrade.data.future_l2 import processed_market_path, selected_factor_screen_path
from backtrade.data.replay import MarketReplay
from backtrade.order_match.maker import MakerMatcher
from backtrade.order_match.taker import MissingL1Error, TakerMatcher
from backtrade.position.single_lot import AccountedFill, SingleLotAccount
from backtrade.simulation.compact_v9 import CompactV9ParquetOutput
from backtrade.simulation.events import BoundaryEvent, FillEvent, MatchView, MarketTick, Order, PortfolioTarget
from backtrade.simulation.execution import ExecutionEngine, normalize_target_qty
from backtrade.simulation.state import MatchMode, OrderSide, OrderStatus, OrderType, TimeInForce, TERMINAL_STATUSES
from backtrade.strategies.signed_factor import SignedFactorStrategy
from backtrade.strategies.factors import factor_semantics_version
from backtrade.runtime.manifest import payload_digest




def _file_identity_with_sha(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=False)
    if not resolved.is_file():
        raise FileNotFoundError(f"manifest input file is missing: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    stat = resolved.stat()
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "exists": True,
        "size_bytes": stat.st_size,
        "sha256": digest.hexdigest(),
    }


def _git_provenance(repo_root: Path) -> tuple[str | None, bool, str | None]:
    """Read repository provenance at run time; never trust caller metadata."""

    try:
        revision = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--verify", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain=v1", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None, True, "repository has no usable git metadata"
    if not revision:
        return None, True, "repository has no usable git revision"
    return revision, bool(status.strip()), None


@dataclass
class CompactV9Result:
    orders: list[Order]
    fills: list[FillEvent]
    account_rows: list[dict[str, Any]]
    snapshots: list[dict[str, Any]]
    activity_rows: list[dict[str, Any]]
    maker_events: list[dict[str, Any]]
    boundary_events: list[BoundaryEvent]
    final_snapshot: dict[str, Any]


class CompactV9Runner:
    def __init__(self, cfg, ticks: Iterable[MarketTick], *, strategy=None):
        if cfg.match.mode not in {"maker", "taker"}:
            raise ValueError("compact_v9 requires match.mode=maker or taker")
        self.cfg = cfg
        self.ticks = iter(ticks)
        self.strategy = strategy or SignedFactorStrategy(cfg.strategy.factor_name)
        self.execution = ExecutionEngine(cfg)
        self.account = SingleLotAccount(cfg.initial_cash)
        self.matcher = MakerMatcher() if cfg.match.mode == "maker" else TakerMatcher()
        self.orders: list[Order] = []
        self.fills: list[FillEvent] = []
        self.account_rows: list[dict[str, Any]] = []
        self.snapshots: list[dict[str, Any]] = []
        self.activity_rows: list[dict[str, Any]] = []
        self.maker_events: list[dict[str, Any]] = []
        self.boundary_events: list[BoundaryEvent] = []
        self.pending_orders: list[Order] = []
        self.active_orders: list[Order] = []
        self._next_order_seq = 1
        self._next_fill_seq = 1
        self._next_target_seq = 1
        self._previous_tick: MarketTick | None = None
        self._previous_view: MatchView | None = None
        self._day_key: str | None = None
        self._day_start_equity: float | None = None
        self._flattened_boundaries: set[tuple[str, str, str, datetime]] = set()
        self._logged_order_rows: set[str] = set()
        self._logged_status_changes: set[tuple[str, int]] = set()
        self._seen_days: set[str] = set()
        self._max_events: int | None = None

    def _rule(self, contract: str):
        if contract in self.cfg.contracts:
            return self.cfg.contracts[contract]
        product = "".join(ch for ch in contract if ch.isalpha())
        for key in (product, product.lower(), product.upper()):
            if key in self.cfg.contracts:
                return self.cfg.contracts[key]
        raise ValueError(f"missing contract rule for {contract}")
    @staticmethod
    def _terminal(order: Order) -> bool:
        return order.status in TERMINAL_STATUSES or order.executed.remaining_qty <= 0

    def _active_for_product(self, product: str) -> list[Order]:
        return [
            order
            for order in [*self.pending_orders, *self.active_orders]
            if order.product.lower() == product.lower() and not self._terminal(order)
        ]

    def _active_order_intent(self, product: str) -> int:
        orders = self._active_for_product(product)
        if not orders:
            return 0
        return 1 if orders[0].created.side is OrderSide.BUY else -1

    def _append_target_activity(self, target: PortfolioTarget, tick: MarketTick) -> None:
        actual_position = self.account.net_qty(target.product)
        target.position_before = actual_position
        self.activity_rows.append({
            "source_dataset": "target",
            "record_type": "target",
            "event_ts": target.decision_ts,
            "target_seq": target.target_seq,
            "product": target.product,
            "contract": target.contract,
            "trading_day": str(tick.trading_day) if tick.trading_day is not None else None,
            "target_qty_raw": float(target.target_qty),
            "target_qty": normalize_target_qty(target.target_qty),
            "reason_code": target.reason_code,
            "risk_state": target.risk_state,
            "factor_name": target.factor_name,
            "factor_score": target.factor_score,
            "factor_semantics_version": target.factor_semantics_version,
            "factor_decision": target.factor_decision,
            "factor_source_ts": target.factor_source_ts,
            "factor_age_ms": target.factor_age_ms,
            "arrival_bid1": tick.bid1,
            "arrival_ask1": tick.ask1,
            "source_seq": tick.source_seq,
            "session_id": tick.session_id,
            "position_before": target.position_before,
        })

    def _append_order_activity(
        self,
        order: Order,
        *,
        allowed_statuses: set[OrderStatus] | None = None,
    ) -> None:
        if order.order_id in self._logged_order_rows:
            for row in reversed(self.activity_rows):
                if row.get("record_type") == "order" and row.get("order_id") == order.order_id:
                    row["actual_arrival_ts"] = getattr(order, "actual_arrival_ts", None)
                    row["arrival_bid1"] = getattr(order, "arrival_bid1", None)
                    row["arrival_ask1"] = getattr(order, "arrival_ask1", None)
                    row["source_seq"] = getattr(order, "source_seq", None)
                    row["session_id"] = getattr(order, "session_id", None)
                    break
        if order.order_id not in self._logged_order_rows:
            self.activity_rows.append(
                {
                    "source_dataset": "order",
                    "record_type": "order",
                    "event_ts": order.created.decision_ts,
                    "order_seq": order.sequence,
                    "target_seq": order.target_seq,
                    "order_id": order.order_id,
                    "product": order.product,
                    "contract": order.contract,
                    "side": order.created.side.value,
                    "status": OrderStatus.CREATED.value,
                    "reason_code": order.created.reason_code,
                    "qty": order.created.qty,
                    "maker_taker_role": order.created.match_mode.value,
                    "submitted_limit_price": order.created.limit_price,
                    "decision_reference_price": getattr(order, "decision_reference_price", None),
                    "scheduled_arrival_ts": getattr(order, "scheduled_arrival_ts", None),
                    "actual_arrival_ts": getattr(order, "actual_arrival_ts", None),
                    "boundary_reason": getattr(order, "boundary_reason", None),
                }
            )
            self._logged_order_rows.add(order.order_id)
        for index, change in enumerate(order.status_history[1:], start=1):
            key = (order.order_id, index)
            if key in self._logged_status_changes:
                continue
            if allowed_statuses is not None and change.to_status not in allowed_statuses:
                continue
            self.activity_rows.append(
                {
                    "source_dataset": "order_event",
                    "record_type": "order_event",
                    "event_ts": change.ts,
                    "order_seq": order.sequence,
                    "target_seq": order.target_seq,
                    "order_id": order.order_id,
                    "product": order.product,
                    "contract": order.contract,
                    "side": order.created.side.value,
                    "status": change.to_status.value,
                    "reason_code": change.reason_code,
                    "arrival_bid1": getattr(order, "arrival_bid1", None),
                    "arrival_ask1": getattr(order, "arrival_ask1", None),
                    "scheduled_arrival_ts": getattr(order, "scheduled_arrival_ts", None),
                    "actual_arrival_ts": getattr(order, "actual_arrival_ts", None),
                    "boundary_reason": getattr(order, "boundary_reason", None),
                }
            )
            self._logged_status_changes.add(key)

    def _append_boundary_activity(self, event: BoundaryEvent) -> None:
        self.activity_rows.append(
            {
                "source_dataset": "order_event",
                "record_type": "order_event",
                "event_ts": event.ts,
                "order_id": event.order_id or None,
                "product": event.product,
                "contract": event.contract,
                "status": "boundary",
                "reason_code": event.reason_code,
                "boundary_reason": event.boundary_reason or event.reason_code,
                "submitted_limit_price": event.price,
                "qty": event.qty,
            }
        )

    def _append_fill_activity(self, fill: FillEvent, order: Order, accounted: AccountedFill) -> None:
        bid = getattr(order, "arrival_bid1", None)
        ask = getattr(order, "arrival_ask1", None)
        tick_size = float(self._rule(order.contract).tick_size)
        spread = (ask - bid) if bid is not None and ask is not None else None
        arrival_price = ask if fill.side is OrderSide.BUY else bid
        self.activity_rows.append({
            "source_dataset": "fill",
            "record_type": "fill",
            "event_ts": fill.fill_ts,
            "order_seq": order.sequence,
            "target_seq": order.target_seq,
            "fill_seq": fill.fill_seq,
            "order_id": fill.order_id,
            "product": fill.product,
            "contract": fill.contract,
            "trading_day": fill.trading_day,
            "side": fill.side.value,
            "status": order.status.value,
            "reason_code": fill.reason_code,
            "arrival_bid1": bid,
            "arrival_ask1": ask,
            "arrival_price": arrival_price,
            "fill_price": fill.price,
            "qty": fill.qty,
            "fee": accounted.open_fee + accounted.close_fee,
            "gross_pnl": accounted.gross_pnl,
            "net_pnl": accounted.net_pnl,
            "maker_taker_role": fill.match_mode,
            "direct_spread_ticks": (spread / tick_size) if spread is not None and tick_size else None,
            "direct_spread_bps": (spread / fill.price * 10_000.0) if spread is not None and fill.price else None,
            "liquidity_source": fill.liquidity_source,
            "boundary_reason": fill.boundary_reason,
        })

    @staticmethod
    def _price_limit_reason(order: Order, view: MatchView) -> str | None:
        if order.created.open_qty <= 0:
            return None
        if order.created.match_mode is MatchMode.TAKER:
            price = float(view.ask_prices[0] if order.created.side is OrderSide.BUY else view.bid_prices[0])
        elif order.created.limit_price is not None:
            price = float(order.created.limit_price)
        else:
            return None
        if view.price_limit_up is not None and price > float(view.price_limit_up):
            return "price_limit_up"
        if view.price_limit_down is not None and price < float(view.price_limit_down):
            return "price_limit_down"
        return None

    def _reject_order(self, order: Order, reason: str, ts: datetime) -> None:
        if order.status is OrderStatus.CREATED:
            order.transition(OrderStatus.SUBMITTED, "submitted", ts)
        if order.status is OrderStatus.SUBMITTED:
            order.transition(OrderStatus.REJECTED, reason, ts)
        elif order.status not in TERMINAL_STATUSES:
            raise RuntimeError(f"cannot reject active order {order.order_id} in {order.status.value}")
        self._append_order_activity(order)

    def _account_fill(self, fill: FillEvent, order: Order) -> AccountedFill:
        position_before = self.account.net_qty(fill.product)
        open_position = self.account.position(fill.product)
        fill.fill_seq = self._next_fill_seq
        self._next_fill_seq += 1
        fill.order_seq = order.sequence
        fill.target_seq = order.target_seq
        accounted = self.account.apply_fill(fill, self._rule(fill.contract))
        lineage_position = self.account.position(fill.product) or open_position
        open_fill_seq = open_position.open_fill_seq if open_position is not None else fill.fill_seq if fill.open_qty else None
        before = accounted.account_before
        after = accounted.account_after
        self.account_rows.append({
            "event_ts": fill.fill_ts,
            "fill_seq": fill.fill_seq,
            "order_seq": order.sequence,
            "order_id": fill.order_id,
            "product": fill.product,
            "contract": fill.contract,
            "trading_day": fill.trading_day,
            "side": fill.side.value,
            "open_fill_seq": open_fill_seq,
            "position_before": position_before,
            "position_after": self.account.net_qty(fill.product),
            "cash_before": before["cash"],
            "cash_after": after["cash"],
            "equity_before": before["equity"],
            "equity_after": after["equity"],
            "realized_pnl_before": before["realized_pnl"],
            "realized_pnl_after": after["realized_pnl"],
            "unrealized_pnl_before": before["unrealized_pnl"],
            "unrealized_pnl_after": after["unrealized_pnl"],
            "total_fee_before": before["total_fee"],
            "total_fee_after": after["total_fee"],
            "gross_pnl": accounted.gross_pnl,
            "open_fee": accounted.open_fee,
            "close_fee": accounted.close_fee,
            "net_pnl": accounted.net_pnl,
            "holding_ms": accounted.holding_ms,
            "reason_code": fill.reason_code,
            "open_order_id": lineage_position.open_order_id if lineage_position is not None else None,
            "open_order_seq": lineage_position.open_order_seq if lineage_position is not None else None,
        })
        self.fills.append(fill)
        order.executed.commission += fill.commission
        order.executed.pnl += fill.pnl
        if order.execution_bits:
            bit = order.execution_bits[-1]
            bit.commission = fill.commission
            bit.pnl = fill.pnl
            bit.close_today_qty = fill.close_today_qty
            bit.position_snapshot = self.account.snapshot()
        self._append_fill_activity(fill, order, accounted)
        return accounted

    def _process_order(self, order: Order, view: MatchView, *, forced: bool = False) -> None:
        if not hasattr(order, "_arrival_tick_ts"):
            setattr(order, "_arrival_tick_ts", view.tick_ts)
            order.actual_arrival_ts = view.tick_ts
            setattr(order, "arrival_bid1", view.bid_prices[0])
            setattr(order, "arrival_ask1", view.ask_prices[0])
            setattr(order, "source_seq", view.source_seq)
            setattr(order, "session_id", view.session_id)
        order.trading_day = view.trading_day
        if order.status is OrderStatus.CREATED:
            order.transition(OrderStatus.SUBMITTED, "submitted", view.tick_ts)
        self._append_order_activity(order)
        if order.created.match_mode is MatchMode.TAKER:
            if order.created.side is OrderSide.BUY:
                l1_price, l1_qty = view.ask_prices[0], view.ask_qtys[0]
            else:
                l1_price, l1_qty = view.bid_prices[0], view.bid_qtys[0]
            if l1_price <= 0 or int(l1_qty) <= 0:
                if forced:
                    raise RuntimeError(f"{order.created.reason_code} requires a valid opposite L1")
                order.transition(OrderStatus.REJECTED, "missing_l1", view.tick_ts)
                self._append_order_activity(order)
                return
        limit_reason = self._price_limit_reason(order, view)
        if limit_reason is not None:
            self._reject_order(order, limit_reason, view.tick_ts)
            return
        try:
            fills, events = self.matcher.match(order, view)
        except MissingL1Error:
            if forced:
                raise RuntimeError(f"{order.created.reason_code} requires a valid opposite L1")
            order.transition(OrderStatus.REJECTED, "missing_l1", view.tick_ts)
            fills, events = [], []
        if events:
            self.maker_events.extend(events)
        if fills:
            self._append_order_activity(
                order,
                allowed_statuses={
                    OrderStatus.SUBMITTED, OrderStatus.ACCEPTED, OrderStatus.QUEUED,
                    OrderStatus.REJECTED, OrderStatus.CANCEL,
                },
            )
        for fill in fills:
            if forced:
                fill.reason_code = order.created.reason_code
                fill.boundary_reason = order.boundary_reason or order.created.reason_code
            self._account_fill(fill, order)
        if fills and order.status is OrderStatus.COMPLETE:
            order.transition(OrderStatus.ACCOUNTED, "position_updated", view.tick_ts)
        self._append_order_activity(order)
        if not self._terminal(order) and order not in self.active_orders:
            self.active_orders.append(order)

    def _cancel_order(self, order: Order, reason: str, ts: datetime, *, view: MatchView | None = None) -> None:
        if self._terminal(order):
            return
        if order.status is OrderStatus.CREATED:
            order.transition(OrderStatus.SUBMITTED, "submitted_before_cancel", ts)
        if order.status is OrderStatus.SUBMITTED:
            order.transition(OrderStatus.ACCEPTED, "accepted_before_cancel", ts)
        order.transition(OrderStatus.CANCEL, reason, ts)
        if isinstance(self.matcher, MakerMatcher):
            self.maker_events.append(self.matcher.cancel_event(order, reason, ts, view))
        self._append_order_activity(order)

    def _cancel_product_orders(self, product: str, reason: str, ts: datetime, *, view: MatchView | None = None) -> None:
        for order in self._active_for_product(product):
            self._cancel_order(order, reason, ts, view=view)
        self.pending_orders = [item for item in self.pending_orders if item.product.lower() != product.lower()]
        self.active_orders = [item for item in self.active_orders if item.product.lower() != product.lower()]

    def _submit_target(
        self,
        target: PortfolioTarget,
        tick: MarketTick,
        view: MatchView,
        current: int,
        *,
        reject_open_reason: str | None = None,
    ) -> None:
        target.target_seq = target.target_seq or self._next_target_seq
        self._next_target_seq = max(self._next_target_seq, int(target.target_seq) + 1)
        self._append_target_activity(target, tick)
        desired = normalize_target_qty(target.target_qty)
        strategy_current = current
        wanted = desired - strategy_current
        existing = self._active_for_product(target.product)
        if wanted == 0:
            for order in existing:
                self._cancel_order(order, "target_changed_cancel_active_order", tick.tick_ts, view=view)
            self.pending_orders = [item for item in self.pending_orders if item not in existing]
            self.active_orders = [item for item in self.active_orders if item not in existing]
            return
        wanted_side = OrderSide.BUY if wanted > 0 else OrderSide.SELL
        wanted_open_qty = 1 if strategy_current == 0 else 0
        wanted_close_qty = 1 if strategy_current != 0 else 0
        if existing:
            if (
                len(existing) == 1
                and existing[0].created.side is wanted_side
                and existing[0].created.open_qty == wanted_open_qty
                and existing[0].created.close_qty == wanted_close_qty
                and existing[0].created.reduce_only == bool(wanted_close_qty)
            ):
                return
            for order in existing:
                self._cancel_order(order, "target_changed_cancel_active_order", tick.tick_ts, view=view)
            self.pending_orders = [item for item in self.pending_orders if item not in existing]
            self.active_orders = [item for item in self.active_orders if item not in existing]
        orders = self.execution.orders_from_target(
            target,
            strategy_current,
            view.bid_prices[0],
            view.ask_prices[0],
            actual_qty=current,
            trading_day=view.trading_day,
            target_seq=target.target_seq,
        )
        for order in orders:
            order.sequence = self._next_order_seq
            self._next_order_seq += 1
            order.order_id = f"order-{order.sequence:012d}"
            self.orders.append(order)
            order.scheduled_arrival_ts = tick.tick_ts + timedelta(milliseconds=max(0, int(self.cfg.execution.latency_ms)))
            order.arrival_ts = order.scheduled_arrival_ts
            if reject_open_reason is not None and order.created.open_qty > 0 and current == 0:
                self._reject_order(order, reject_open_reason, tick.tick_ts)
                continue
            self._append_order_activity(order)
            if order.scheduled_arrival_ts <= tick.tick_ts:
                self._process_order(order, view)
            else:
                self.pending_orders.append(order)

    def _risk_reason(self, tick: MarketTick, current: int) -> str | None:
        if self._day_start_equity is None:
            self._day_key = str(tick.trading_day) if tick.trading_day is not None else tick.tick_ts.date().isoformat()
            self._day_start_equity = float(self.account.equity)
        daily_loss = max(0.0, float(self._day_start_equity) - float(self.account.equity))
        position = self.account.position(tick.product)
        if current != 0 and position is not None:
            max_holding_ms = int(self.cfg.risk.max_holding_ms)
            holding_ms = max(0, int((tick.tick_ts - position.open_ts).total_seconds() * 1000))
            if max_holding_ms >= 0 and holding_ms >= max_holding_ms:
                return "max_holding_flatten"
        if bool(self.cfg.risk.stop_on_capital_depleted) and self.account.equity <= float(self.cfg.risk.capital_floor):
            return "capital_depleted_flatten"
        if daily_loss >= float(self.cfg.risk.max_daily_loss):
            return "max_daily_loss_flatten"
        return None

    def _flatten(self, product: str, contract: str, view: MatchView, reason: str, *, day_end_marker: bool = False) -> None:
        current = self.account.net_qty(product)
        boundary_key = (product.lower(), contract, reason, view.tick_ts)
        active = self._active_for_product(product)
        if boundary_key in self._flattened_boundaries and current == 0 and not active:
            return
        self._flattened_boundaries.add(boundary_key)
        self._cancel_product_orders(product, f"{reason}_cancel_active_order", view.tick_ts, view=view)
        boundary = BoundaryEvent(
            view.tick_ts,
            product,
            contract,
            "forced_flatten",
            "info",
            reason,
            detail="day_end_marker" if day_end_marker else "",
            boundary_reason=reason,
        )
        self.boundary_events.append(boundary)
        self._append_boundary_activity(boundary)
        if current == 0:
            return
        side = OrderSide.SELL if current > 0 else OrderSide.BUY
        order = Order.create(
            contract=contract,
            product=product,
            side=side,
            qty=1,
            limit_price=None,
            order_type=OrderType.LIMIT,
            time_in_force=TimeInForce.GTC,
            match_mode=MatchMode.TAKER,
            decision_ts=view.tick_ts,
            reason_code=reason,
            open_qty=0,
            close_qty=1,
            reduce_only=True,
            trading_day=view.trading_day,
        )
        order.sequence = self._next_order_seq
        self._next_order_seq += 1
        order.order_id = f"order-{order.sequence:012d}"
        order.latency_exempt_reason = "forced_liquidation"
        order.boundary_reason = reason
        order.scheduled_arrival_ts = view.tick_ts
        order.arrival_ts = view.tick_ts
        order.decision_reference_price = view.bid_prices[0] if side is OrderSide.SELL else view.ask_prices[0]
        self.orders.append(order)
        self._append_order_activity(order)
        original = self.matcher
        self.matcher = TakerMatcher()
        try:
            self._process_order(order, view, forced=True)
        finally:
            self.matcher = original
        if self.account.net_qty(product) != 0:
            raise RuntimeError(f"{reason} failed for {contract}: residual_qty={self.account.net_qty(product)}")

    def _snapshot_record(self, tick: MarketTick, reason: str) -> dict[str, Any]:
        account_snapshot = self.account.snapshot()
        position = self.account.position(tick.product)
        return {
            "event_ts": tick.tick_ts,
            "product": tick.product,
            "contract": tick.contract,
            "mark_price": tick.mid,
            "snapshot_reason": reason,
            "position_qty": self.account.net_qty(tick.product),
            "position_side": position.side.value if position is not None else "flat",
            "position_open_fill_seq": position.open_fill_seq if position is not None else None,
            "position_open_order_id": position.open_order_id if position is not None else None,
            **{key: account_snapshot[key] for key in ("cash", "equity", "realized_pnl", "unrealized_pnl", "total_fee")},
        }

    def run(self, *, max_events: int | None = None) -> CompactV9Result:
        if max_events is not None and int(max_events) <= 0:
            raise ValueError("max_events must be positive when provided")
        self._max_events = int(max_events) if max_events is not None else None
        bounded = max_events is not None or self.cfg.data.max_ticks is not None
        replay = MarketReplay(
            self.ticks,
            closing_window_ms=int(self.cfg.execution.day_end_flatten_window_ms),
            eof_is_day_end=bool(self.cfg.data.eof_is_day_end),
            bounded=bounded,
            expected_products={self.cfg.data.product},
        )
        iterable = islice(replay, max_events) if max_events is not None else iter(replay)
        for tick, strategy_view, view in iterable:
            self._seen_days.add(
                str(tick.trading_day) if tick.trading_day is not None else tick.tick_ts.date().isoformat()
            )
            if self._previous_tick is not None and self._previous_tick.product.lower() == tick.product.lower() and self._previous_tick.contract != tick.contract:
                previous_product = self._previous_tick.product
                if self.account.net_qty(previous_product) != 0 or self._active_for_product(previous_product):
                    self._flatten(
                        previous_product,
                        self._previous_tick.contract,
                        self._previous_view or view,
                        "contract_roll_flatten",
                        day_end_marker=bool(strategy_view.is_day_closing),
                    )

            day_key = str(tick.trading_day) if tick.trading_day is not None else tick.tick_ts.date().isoformat()
            if day_key != self._day_key:
                self._day_key = day_key
                self._day_start_equity = float(self.account.equity)
            boundary_start = bool(
                strategy_view.is_day_closing
                or strategy_view.is_last_tick_of_contract
                or strategy_view.is_last_tick_of_day
            )
            terminal_boundary = bool(strategy_view.is_last_tick_of_contract or strategy_view.is_last_tick_of_day)

            if boundary_start:
                cancel_reason = (
                    "contract_roll_cancel_active_order"
                    if strategy_view.is_last_tick_of_contract
                    else "day_end_cancel_active_order"
                )
                self._cancel_product_orders(tick.product, cancel_reason, tick.tick_ts, view=view)
                if terminal_boundary:
                    reason = "contract_roll_flatten" if strategy_view.is_last_tick_of_contract else "day_end_flatten"
                    self._flatten(
                        tick.product,
                        tick.contract,
                        view,
                        reason,
                        day_end_marker=bool(strategy_view.is_last_tick_of_contract and strategy_view.is_last_tick_of_day),
                    )
            self.account.mark_to_market(tick.product, tick.mid)
            if not boundary_start:
                current = self.account.net_qty(tick.product)
                risk_reason = self._risk_reason(tick, current)
                if risk_reason is not None:
                    self._cancel_product_orders(tick.product, f"{risk_reason}_cancel_active_order", tick.tick_ts, view=view)
                    if current != 0:
                        self._flatten(tick.product, tick.contract, view, risk_reason)
                    elif strategy_view.factor_decision:
                        target = self.strategy.on_decision(strategy_view, current)
                        self._submit_target(target, tick, view, current, reject_open_reason=risk_reason)
                else:
                    for order in list(self.pending_orders):
                        if order.arrival_ts is not None and order.arrival_ts <= tick.tick_ts:
                            self.pending_orders.remove(order)
                            self._process_order(order, view)
                    for order in list(self.active_orders):
                        if isinstance(self.matcher, MakerMatcher) and not self._terminal(order):
                            self._process_order(order, view)
                    self.active_orders = [order for order in self.active_orders if not self._terminal(order)]
                    current = self.account.net_qty(tick.product)
                    risk_reason = self._risk_reason(tick, current)
                    if risk_reason is not None and current != 0:
                        self._cancel_product_orders(tick.product, f"{risk_reason}_cancel_active_order", tick.tick_ts, view=view)
                        self._flatten(tick.product, tick.contract, view, risk_reason)
                    elif strategy_view.factor_decision:
                        target = self.strategy.on_decision(strategy_view, current)
                        self._submit_target(target, tick, view, current, reject_open_reason=risk_reason)
            snapshot_reason = "contract_roll_flatten" if strategy_view.is_last_tick_of_contract else "day_end_flatten" if strategy_view.is_last_tick_of_day else "day_end_cancel_active_order" if boundary_start else "tick"
            self.snapshots.append(self._snapshot_record(tick, snapshot_reason))
            self._previous_tick = tick
            self._previous_view = view
        if self._previous_tick is not None and self.account.net_qty(self._previous_tick.product) != 0:
            previous_tick = self._previous_tick
            previous_view = self._previous_view or MarketReplay.match_view(previous_tick)
            self._flatten(previous_tick.product, previous_tick.contract, previous_view, "end_of_data_flatten")
            self.account.mark_to_market(previous_tick.product, previous_tick.mid)
            self.snapshots.append(self._snapshot_record(previous_tick, "end_of_data_flatten"))
        final_ts = self._previous_tick.tick_ts if self._previous_tick else datetime.now()
        if self._previous_tick is not None:
            self._cancel_product_orders(self._previous_tick.product, "end_of_data_cancel_active_order", final_ts, view=self._previous_view)
        final = self.account.snapshot()
        if any(value != 0 for value in final.get("net_qty", {}).values()):
            raise RuntimeError(f"end_of_data_flatten failed: {final['net_qty']}")
        for event_seq, row in enumerate(self.activity_rows, start=1):
            row["event_seq"] = event_seq
            if not row.get("record_type"):
                row["record_type"] = row.get("source_dataset")
        return CompactV9Result(self.orders, self.fills, self.account_rows, self.snapshots, self.activity_rows, self.maker_events, self.boundary_events, final)

    def _manifest_payload(self, result: CompactV9Result, *, output_root: str | Path | None = None) -> dict[str, Any]:
        def identity(path: str | Path | None) -> dict[str, Any] | None:
            if path is None:
                return None
            return _file_identity_with_sha(path)

        identities: dict[str, Any] = {}
        identities["market"] = identity(processed_market_path(self.cfg))
        factor_path = selected_factor_screen_path(self.cfg)
        identities["factor"] = identity(factor_path)
        identities["factor_manifest"] = identity(factor_path.with_name("manifest.json"))
        for index, contract_file in enumerate(self.cfg.contract_files):
            identities[f"contract_file_{index}"] = identity(contract_file)
        if self.cfg.limit_reference.snapshot_path is not None:
            identities["price_limit_snapshot"] = _file_identity_with_sha(self.cfg.limit_reference.snapshot_path)
        config = self.cfg.model_dump(mode="json")
        source_files = [
            Path(__file__),
            Path(__file__).with_name("compact_v9.py"),
            Path(__file__).parents[1] / "order_match" / "maker.py",
            Path(__file__).parents[1] / "order_match" / "taker.py",
            Path(__file__).parents[1] / "simulation" / "execution.py",
            Path(__file__).parents[1] / "simulation" / "events.py",
            Path(__file__).parents[1] / "position" / "single_lot.py",
            Path(__file__).parents[1] / "strategies" / "signed_factor.py",
            Path(__file__).parents[1] / "strategies" / "factors.py",
            Path(__file__).parents[1] / "data" / "future_l2.py",
            Path(__file__).parents[1] / "data" / "tabular.py",
            Path(__file__).parents[1] / "data" / "market_quality.py",
            Path(__file__).parents[1] / "data" / "replay.py",
        ]
        repo_root = Path(__file__).resolve().parents[2]
        source_hashes = {
            str(path.relative_to(repo_root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_files
            if path.is_file()
        }
        git_revision, dirty, provenance_reason = _git_provenance(repo_root)
        source_provenance = {"git_revision": git_revision, "dirty": dirty, "source_files_sha256": source_hashes}
        if provenance_reason is not None:
            source_provenance["reason"] = provenance_reason
        days = sorted(self._seen_days)
        bounded = bool(self.cfg.data.max_ticks is not None or self._max_events is not None)
        max_events = self._max_events
        return {
        # [README-6] manifest 记录 writer 实际收到的输出目录，而不是配置默认目录。
            "engine_version": "compact_v9_core",
            "output_root": str(Path(output_root).expanduser().resolve()) if output_root is not None else None,
            "strategy_name": "signed_factor",
            "factor_name": self.cfg.strategy.factor_name,
            "factor_semantics_version": factor_semantics_version(self.cfg.strategy.factor_name),
            "match_mode": self.cfg.match.mode,
            "config": config,
            "config_digest": payload_digest(config),
            "input_identities": identities,
            "source_provenance": source_provenance,
            "day": days[0] if len(days) == 1 else days,
            "runtime": {
                "bounded": bounded,
                "eof_boundary": "end_of_data" if bounded else ("known_day_end" if self.cfg.data.eof_is_day_end else "end_of_data"),
                "eof_is_day_end": bool(self.cfg.data.eof_is_day_end and not bounded),
                "max_events": max_events,
                "max_ticks": self.cfg.data.max_ticks,
            },
            "latency": {"configured_ms": int(self.cfg.execution.latency_ms)},
            "price_limit_approximation": {
                "mode": self.cfg.limit_reference.mode,
                "prev_day_vwap_proxy": self.cfg.limit_reference.mode == "prev_day_vwap_proxy",
            },
            "maker_model_version": "mbp_prob_queue_v2_strict_through" if self.cfg.match.mode == "maker" else None,
            "mbp_estimation": self.cfg.match.mode == "maker",
            "fifo_reconstruction": False,
            "single_lot": True,
            "position_limit": 1,
            "margin_enabled": False,
            "final_position_zero": all(value == 0 for value in result.final_snapshot.get("net_qty", {}).values()),
        }

    def write(self, output_root) -> dict[str, Any]:
        result = CompactV9Result(self.orders, self.fills, self.account_rows, self.snapshots, self.activity_rows, self.maker_events, self.boundary_events, self.account.snapshot())
        sink = CompactV9ParquetOutput(output_root, maker_enabled=self.cfg.match.mode == "maker")
        sink.write_many("activity", result.activity_rows)
        sink.write_many("account", result.account_rows)
        sink.write_many("snapshot", result.snapshots)
        if self.cfg.match.mode == "maker":
            sink.write_many("maker_event", result.maker_events)
        manifest = self._manifest_payload(result, output_root=output_root)
        return sink.close(manifest=manifest)


__all__ = ["CompactV9Result", "CompactV9Runner"]

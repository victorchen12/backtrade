from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from backtrade.simulation.events import FillEvent
from backtrade.simulation.state import OrderSide


def _fee(
    rule: Any,
    price: float,
    multiplier: float,
    qty: int = 1,
    *,
    contract: str | None = None,
) -> float:
    value = float(rule.value_for(contract) if hasattr(rule, "value_for") else rule.value)
    if rule.mode == "per_lot":
        return value * qty
    notional = price * multiplier * qty
    if rule.mode == "rate":
        return notional * value
    if rule.mode == "bps":
        return notional * value / 10_000.0
    raise ValueError(f"unsupported fee mode: {rule.mode}")


@dataclass(slots=True)
class OpenPosition:
    product: str
    contract: str
    side: OrderSide
    open_price: float
    open_ts: datetime
    open_trading_day: str
    open_order_id: str
    open_order_seq: int | None
    open_fill_seq: int | None
    open_fee: float
    multiplier: float

    @property
    def sign(self) -> int:
        return 1 if self.side is OrderSide.BUY else -1


@dataclass(slots=True)
class AccountedFill:
    fill: FillEvent
    open_fee: float
    close_fee: float
    gross_pnl: float
    net_pnl: float
    holding_ms: int | None
    account_before: dict[str, float]
    cash_delta: float
    account_after: dict[str, float]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.fill, name)


class SingleLotAccount:
    def __init__(self, cash: float):
        self.cash = float(cash)
        self.equity = float(cash)
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.total_fee = 0.0
        self.positions: dict[str, OpenPosition] = {}
        self._last_marks: dict[str, float] = {}

    def _values(self) -> dict[str, float]:
        return {
            "cash": self.cash,
            "equity": self.equity,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "total_fee": self.total_fee,
        }

    def net_qty(self, product: str) -> int:
        position = self.positions.get(product)
        return 0 if position is None else position.sign

    def position(self, product: str) -> OpenPosition | None:
        return self.positions.get(product)

    def apply_fill(self, fill: FillEvent, rule: Any) -> AccountedFill:
        if fill.qty != 1:
            raise ValueError("single-lot accounting accepts exactly one fill")
        if fill.open_qty + fill.close_qty != 1:
            raise ValueError("fill open_qty + close_qty must equal one")
        if fill.price <= 0:
            raise ValueError("fill price must be positive")
        product = fill.product
        before = self._values()
        current = self.positions.get(product)
        multiplier = float(rule.multiplier)
        trading_day = str(fill.trading_day or fill.fill_ts.date().isoformat())
        open_fee = 0.0
        close_fee = 0.0
        gross = 0.0
        holding_ms: int | None = None

        if current is None:
            if fill.reduce_only:
                raise ValueError("reduce-only fill cannot open from flat")
            if fill.close_qty != 0 or fill.open_qty != 1:
                raise ValueError("opening fill must be marked open_qty=1")
            open_fee = _fee(rule.fee.open, fill.price, multiplier, contract=fill.contract)
            self.cash -= open_fee
            self.total_fee += open_fee
            self.positions[product] = OpenPosition(
                product=product,
                contract=fill.contract,
                side=fill.side,
                open_price=float(fill.price),
                open_ts=fill.fill_ts,
                open_trading_day=trading_day,
                open_order_id=fill.order_id,
                open_order_seq=fill.order_seq,
                open_fill_seq=fill.fill_seq,
                open_fee=open_fee,
                multiplier=multiplier,
            )
        else:
            if fill.side is current.side:
                raise ValueError("same-direction repeat opening is forbidden")
            if fill.contract != current.contract:
                raise ValueError("closing fill contract differs from open position contract")
            if multiplier != current.multiplier:
                raise ValueError("closing contract multiplier differs from opening multiplier")
            if fill.close_qty != 1 or fill.open_qty != 0:
                raise ValueError("closing fill must be marked close_qty=1")
            gross = current.sign * (float(fill.price) - current.open_price) * current.multiplier
            close_rule = rule.fee.close_today if current.open_trading_day == trading_day else rule.fee.close
            close_fee = _fee(close_rule, fill.price, current.multiplier, contract=current.contract)
            holding_ms = max(0, int((fill.fill_ts - current.open_ts).total_seconds() * 1000))
            self.cash += gross - close_fee
            self.realized_pnl += gross
            self.total_fee += close_fee
            self.positions.pop(product)

        self._recompute_unrealized()
        fill.commission = open_fee + close_fee
        fill.pnl = gross
        fill.open_qty = 1 if current is None else 0
        fill.close_qty = 0 if current is None else 1
        fill.close_today_qty = 1 if current is not None and current.open_trading_day == trading_day else 0
        fill.fee_by_open_close_close_today = {
            "open": open_fee,
            "close": close_fee if fill.close_today_qty == 0 else 0.0,
            "close_today": close_fee if fill.close_today_qty == 1 else 0.0,
        }
        after = self._values()
        return AccountedFill(
            fill=fill,
            open_fee=open_fee,
            close_fee=close_fee,
            gross_pnl=gross,
            net_pnl=-open_fee if current is None else gross - close_fee,
            cash_delta=-open_fee if current is None else gross - close_fee,
            holding_ms=holding_ms,
            account_before=before,
            account_after=after,
        )

    def mark_to_market(self, product: str, price: float) -> None:
        if price > 0:
            self._last_marks[product] = float(price)
        self._recompute_unrealized()

    def _recompute_unrealized(self) -> None:
        unrealized = 0.0
        for name, position in self.positions.items():
            mark = self._last_marks.get(name)
            if mark is not None:
                unrealized += position.sign * (mark - position.open_price) * position.multiplier
        self.unrealized_pnl = unrealized
        self.equity = self.cash + unrealized

    def snapshot(self) -> dict[str, Any]:
        positions = {
            product: {
                "side": position.side.value,
                "open_price": position.open_price,
                "open_ts": position.open_ts,
                "open_trading_day": position.open_trading_day,
                "open_order_id": position.open_order_id,
                "open_order_seq": position.open_order_seq,
                "open_fill_seq": position.open_fill_seq,
                "open_fee": position.open_fee,
            }
            for product, position in self.positions.items()
        }
        return {**self._values(), "net_qty": {product: self.net_qty(product) for product in sorted(set(positions) | set(self._last_marks))}, "positions": positions}


__all__ = ["AccountedFill", "OpenPosition", "SingleLotAccount"]

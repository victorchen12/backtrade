from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Iterator

from backtrade.simulation.events import MarketTick, MatchView, StrategyView


@dataclass
class ReplayClock:
    current_ts: datetime | None = None

    def update(self, ts: datetime) -> None:
        self.current_ts = ts


class MarketReplay:
    """Single-product replay with explicit end-of-day evidence."""

    def __init__(
        self,
        ticks: Iterable[MarketTick],
        closing_window_ms: int = 5000,
        *,
        eof_is_day_end: bool = False,
        bounded: bool = False,
        expected_products: set[str] | None = None,
    ):
        if int(closing_window_ms) < 0:
            raise ValueError("closing_window_ms must be non-negative")
        self.ticks = iter(ticks)
        self.clock = ReplayClock()
        self.closing_window = timedelta(milliseconds=int(closing_window_ms))
        self.eof_is_day_end = bool(eof_is_day_end and not bounded)
        declared_products = {str(product).lower() for product in (expected_products or set())}
        if len(declared_products) > 1:
            raise ValueError("compact_v9 supports exactly one expected product")
        self.expected_product = next(iter(declared_products), None)

    def __iter__(self) -> Iterator[tuple[MarketTick, StrategyView, MatchView]]:
        observed_product: str | None = self.expected_product
        buffer: deque[MarketTick] = deque()
        exhausted = False

        def append_next() -> MarketTick | None:
            nonlocal exhausted, observed_product
            try:
                candidate = next(self.ticks)
            except StopIteration:
                exhausted = True
                return None
            product_key = candidate.product.lower()
            if observed_product is None:
                observed_product = product_key
            if product_key != observed_product:
                raise ValueError("compact_v9 supports a single product; interleaved products are not supported")
            buffer.append(candidate)
            return candidate

        append_next()
        previous_order_key: tuple[datetime, int] | None = None
        active_contract_by_product: dict[str, str] = {}
        closed_contracts_by_product: dict[str, set[str]] = {}
        buffered_day: object | None = None
        buffered_day_count = 0
        buffered_day_end: MarketTick | None = None
        buffered_day_end_known = False
        while buffer:
            tick = buffer[0]
            day_key = self._day_key(tick)
            if buffered_day != day_key:
                buffered_day = day_key
                buffered_day_count = 1
                buffered_day_end = tick
                buffered_day_end_known = False
            while not exhausted and not buffered_day_end_known:
                if not self.eof_is_day_end and self.closing_window > timedelta(0) and buffer[-1].tick_ts - tick.tick_ts >= self.closing_window:
                    break
                candidate = append_next()
                if candidate is None:
                    if self.eof_is_day_end:
                        buffered_day_end_known = True
                    break
                if self._day_key(candidate) != day_key:
                    buffered_day_end_known = True
                    break
                buffered_day_count += 1
                buffered_day_end = candidate
                if not self.eof_is_day_end and self.closing_window == timedelta(0):
                    break

            next_tick = buffer[1] if len(buffer) > 1 else None
            day_end_tick = buffered_day_end
            day_end_known = buffered_day_end_known
            is_last_tick_of_day = day_end_known and buffered_day_count == 1
            is_day_closing = day_end_known and tick.tick_ts >= day_end_tick.tick_ts - self.closing_window
            seconds_to_day_end = (
                max(0.0, (day_end_tick.tick_ts - tick.tick_ts).total_seconds()) if day_end_known else None
            )
            is_last_tick_of_contract = next_tick is not None and (
                next_tick.product.lower() != tick.product.lower() or next_tick.contract != tick.contract
            )
            product_key = tick.product.lower()
            active_contract = active_contract_by_product.get(product_key)
            closed_contracts = closed_contracts_by_product.setdefault(product_key, set())
            if tick.contract in closed_contracts:
                raise ValueError(f"contract sequence for product {tick.product} reopens closed contract {tick.contract}")
            if active_contract is not None and active_contract != tick.contract:
                closed_contracts.add(active_contract)
            active_contract_by_product[product_key] = tick.contract
            order_key = (tick.tick_ts, tick.source_seq)
            if previous_order_key is not None and (
                tick.tick_ts < previous_order_key[0] or tick.source_seq <= previous_order_key[1]
            ):
                raise ValueError(f"market ticks out of order: {order_key} follows {previous_order_key}")
            self.clock.update(tick.tick_ts)
            yield tick, self.strategy_view(
                tick,
                is_day_closing,
                is_last_tick_of_day,
                is_last_tick_of_contract,
                seconds_to_day_end,
            ), self.match_view(tick)
            previous_order_key = order_key
            buffer.popleft()
            buffered_day_count -= 1
            if buffered_day_count == 0:
                buffered_day = None
                buffered_day_end = None
                buffered_day_end_known = False
            if not buffer and not exhausted:
                append_next()

    @staticmethod
    def _day_key(tick: MarketTick | None) -> object | None:
        if tick is None:
            return None
        return tick.trading_day if tick.trading_day is not None else tick.tick_ts.date()

    @staticmethod
    def strategy_view(
        tick: MarketTick,
        is_day_closing: bool = False,
        is_last_tick_of_day: bool = False,
        is_last_tick_of_contract: bool = False,
        seconds_to_day_end: float | None = None,
    ) -> StrategyView:
        factors = dict(tick.factors)
        factors["is_day_end_flatten"] = 1.0 if is_day_closing else 0.0
        return StrategyView(
            product=tick.product,
            contract=tick.contract,
            tick_ts=tick.tick_ts,
            mid=tick.mid,
            factors=factors,
            trading_day=str(tick.trading_day) if tick.trading_day is not None else None,
            factor_decision=tick.factor_decision,
            factor_source_ts=tick.factor_source_ts,
            factor_age_ms=tick.factor_age_ms,
            is_day_closing=is_day_closing,
            is_last_tick_of_day=is_last_tick_of_day,
            is_last_tick_of_contract=is_last_tick_of_contract,
            seconds_to_day_end=seconds_to_day_end,
        )

    @staticmethod
    def match_view(tick: MarketTick) -> MatchView:
        return MatchView(
            product=tick.product,
            contract=tick.contract,
            tick_ts=tick.tick_ts,
            bid_prices=tick.bid_prices,
            bid_qtys=tick.bid_qtys,
            ask_prices=tick.ask_prices,
            ask_qtys=tick.ask_qtys,
            mid=tick.mid,
            spread=tick.spread,
            last_price=tick.last_price,
            vol_inc=tick.vol_inc,
            trade_direction=tick.trade_direction,
            source_seq=tick.source_seq,
            session_id=tick.session_id,
            cancel_bid_tick=tick.cancel_bid_tick,
            cancel_ask_tick=tick.cancel_ask_tick,
            cancel_total_tick=tick.cancel_total_tick,
            cancel_imbalance_tick=tick.cancel_imbalance_tick,
            cancel_reliability_score=tick.cancel_reliability_score,
            stale_ms=tick.stale_ms,
            cancel_event_flag=tick.cancel_event_flag,
            quote_change_flag=tick.quote_change_flag,
            side_ambiguous_flag=tick.side_ambiguous_flag,
            level_shift_flag=tick.level_shift_flag,
            is_anomaly=tick.is_anomaly,
            is_stale=tick.is_stale,
            trading_day=str(tick.trading_day) if tick.trading_day is not None else None,
            trade_direction_source=tick.trade_direction_source,
            trade_direction_confidence=tick.trade_direction_confidence,
            direction_source=tick.direction_source,
            direction_confidence=tick.direction_confidence,
            trade_direction_quality=tick.trade_direction_quality,
            direction_quality=tick.direction_quality,
            price_limit_up=tick.price_limit_up,
            price_limit_down=tick.price_limit_down,
            price_limit_reference_price=tick.price_limit_reference_price,
            price_limit_reference_source=tick.price_limit_reference_source,
            price_limit_rule_version=tick.price_limit_rule_version,
        )

from __future__ import annotations

from enum import Enum


class OrderStatus(str, Enum):
    CREATED = "created"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    QUEUED = "queued"
    PARTIAL = "partial"
    COMPLETE = "complete"
    REJECTED = "rejected"
    CANCEL = "cancel"
    ACCOUNTED = "accounted"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    LIMIT = "limit"


class MatchMode(str, Enum):
    MAKER = "maker"
    TAKER = "taker"


class TimeInForce(str, Enum):
    GTC = "gtc"


class InvalidOrderTransition(ValueError):
    pass


TERMINAL_STATUSES = {OrderStatus.REJECTED, OrderStatus.CANCEL, OrderStatus.ACCOUNTED}

ALLOWED_TRANSITIONS = {
    OrderStatus.CREATED: {OrderStatus.SUBMITTED, OrderStatus.REJECTED},
    OrderStatus.SUBMITTED: {OrderStatus.ACCEPTED, OrderStatus.REJECTED},
    OrderStatus.ACCEPTED: {OrderStatus.QUEUED, OrderStatus.CANCEL},
    OrderStatus.QUEUED: {OrderStatus.PARTIAL, OrderStatus.COMPLETE, OrderStatus.CANCEL},
    OrderStatus.PARTIAL: {OrderStatus.PARTIAL, OrderStatus.COMPLETE, OrderStatus.CANCEL},
    OrderStatus.COMPLETE: {OrderStatus.ACCOUNTED},
}


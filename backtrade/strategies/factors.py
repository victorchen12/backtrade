from __future__ import annotations

import math
import re

L1_IMBALANCE_NAME = "l1_imbalance"
SIGNED_FACTOR_SEMANTICS_VERSION = "signed_factor_v1"
ECDF_TAIL_SEMANTICS_VERSION = "ecdf_tail_v1"
SUPPORTED_FACTOR_SEMANTICS = frozenset(
    {SIGNED_FACTOR_SEMANTICS_VERSION, ECDF_TAIL_SEMANTICS_VERSION}
)
FACTOR_SEMANTICS = {
    L1_IMBALANCE_NAME: SIGNED_FACTOR_SEMANTICS_VERSION,
}
_FACTOR_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
_RESERVED_FACTOR_NAMES = frozenset(
    {
        "tick_ts",
        "product",
        "trading_day",
        "session_id",
        "underlying_secu_cd",
        "active_factor",
        "factor_source_ts",
        "factor_decision",
        "factor_age_ms",
        "part",
        "split_id",
        "source_seq",
    }
)


def validate_factor_name(factor_name: str) -> str:
    if not isinstance(factor_name, str):
        raise ValueError("factor name must contain only letters, digits, dot, underscore, or hyphen")
    if factor_name in _RESERVED_FACTOR_NAMES:
        raise ValueError(f"factor name is reserved: {factor_name}")
    if (
        not factor_name
        or factor_name in {".", ".."}
        or _FACTOR_NAME_PATTERN.fullmatch(factor_name) is None
    ):
        raise ValueError("factor name must contain only letters, digits, dot, underscore, or hyphen")
    return factor_name


def factor_semantics_version(factor_name: str, signal_mode: str = "signed_factor") -> str:
    factor_name = validate_factor_name(factor_name)
    if signal_mode == "ecdf_tail":
        return ECDF_TAIL_SEMANTICS_VERSION
    if signal_mode != "signed_factor":
        raise ValueError(f"unsupported signal mode: {signal_mode}")
    return FACTOR_SEMANTICS.get(factor_name, SIGNED_FACTOR_SEMANTICS_VERSION)


def compute_l1_imbalance(bid1_qty: float, ask1_qty: float) -> float:
    bid = float(bid1_qty)
    ask = float(ask1_qty)
    if not math.isfinite(bid) or not math.isfinite(ask):
        raise ValueError("L1 quantities must be finite")
    if bid < 0 or ask < 0:
        raise ValueError("L1 quantities must be non-negative")
    total = bid + ask
    return 0.0 if total == 0 else (bid - ask) / total


__all__ = [
    "FACTOR_SEMANTICS",
    "ECDF_TAIL_SEMANTICS_VERSION",
    "L1_IMBALANCE_NAME",
    "SIGNED_FACTOR_SEMANTICS_VERSION",
    "SUPPORTED_FACTOR_SEMANTICS",
    "compute_l1_imbalance",
    "factor_semantics_version",
    "validate_factor_name",
]

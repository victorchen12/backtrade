from __future__ import annotations

import math

L1_IMBALANCE_NAME = "l1_imbalance"
SUPPORTED_FACTOR_NAMES = frozenset({L1_IMBALANCE_NAME})
FACTOR_SEMANTICS = {
    L1_IMBALANCE_NAME: "signed_factor_v1",
}


def factor_semantics_version(factor_name: str) -> str:
    try:
        return FACTOR_SEMANTICS[factor_name]
    except KeyError as exc:
        raise ValueError(f"unsupported factor: {factor_name}") from exc


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
    "L1_IMBALANCE_NAME",
    "SUPPORTED_FACTOR_NAMES",
    "compute_l1_imbalance",
    "factor_semantics_version",
]

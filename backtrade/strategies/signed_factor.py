from __future__ import annotations

import math

from backtrade.simulation.events import PortfolioTarget, StrategyView
from backtrade.strategies.factors import (
    L1_IMBALANCE_NAME,
    factor_semantics_version,
    validate_factor_name,
)


class SignedFactorStrategy:
    """将已完成的有符号因子值映射为单手目标仓位。"""

    required_features = frozenset({"active_factor"})

    def __init__(self, factor_name: str = L1_IMBALANCE_NAME) -> None:
        self.factor_name = validate_factor_name(factor_name)
        self.factor_semantics_version = factor_semantics_version(self.factor_name)
        self.reason_prefix = "l1" if self.factor_name == L1_IMBALANCE_NAME else "factor"

    def on_decision(self, view: StrategyView, current_position: int) -> PortfolioTarget:
        if current_position not in {-1, 0, 1}:
            raise ValueError("signed factor strategy requires current position in {-1, 0, +1}")
        score = view.factors.get("active_factor")
        if not view.factor_decision:
            return self._target(view, current_position, None, f"{self.reason_prefix}_factor_wait")
        if score is None or not math.isfinite(float(score)):
            raise ValueError("signed factor strategy requires a finite active_factor on decision ticks")
        score = float(score)
        desired = 1 if score > 0 else -1 if score < 0 else current_position
        if score == 0:
            reason = f"{self.reason_prefix}_hold_zero"
        elif current_position and desired != current_position:
            return self._target(view, 0, score, f"{self.reason_prefix}_flat_for_reversal", reduce_only=True)
        elif desired > 0:
            reason = f"{self.reason_prefix}_long" if current_position == 0 else f"{self.reason_prefix}_hold_long"
        elif desired < 0:
            reason = f"{self.reason_prefix}_short" if current_position == 0 else f"{self.reason_prefix}_hold_short"
        else:
            reason = f"{self.reason_prefix}_flat"
        return self._target(view, desired, score, reason)

    def _target(
        self,
        view: StrategyView,
        target_qty: int,
        score: float | None,
        reason_code: str,
        *,
        reduce_only: bool = False,
    ) -> PortfolioTarget:
        return PortfolioTarget(
            product=view.product,
            contract=view.contract,
            decision_ts=view.tick_ts,
            target_qty=int(target_qty),
            reduce_only=reduce_only,
            reason_code=reason_code,
            factor_name=self.factor_name,
            factor_score=score,
            factor_semantics_version=self.factor_semantics_version,
            factor_decision=view.factor_decision,
            factor_source_ts=view.factor_source_ts,
            factor_age_ms=view.factor_age_ms,
            position_before=None,
        )


__all__ = ["SignedFactorStrategy"]

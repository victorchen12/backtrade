from __future__ import annotations

import math

from backtrade.simulation.events import PortfolioTarget, StrategyView


class OFISignStrategy:
    """Map a completed five-second OFI value to a single-lot signed target."""

    required_features = frozenset({"active_factor"})
    factor_semantics_version = "ofi_sign_v1"

    def __init__(self, factor_name: str = "ofi_cks_best_level_5s") -> None:
        if factor_name != "ofi_cks_best_level_5s":
            raise ValueError("compact_v9 supports only ofi_cks_best_level_5s")
        self.factor_name = factor_name

    def on_decision(self, view: StrategyView, current_position: int) -> PortfolioTarget:
        if current_position not in {-1, 0, 1}:
            raise ValueError("OFI sign strategy requires current position in {-1, 0, +1}")
        score = view.factors.get("active_factor")
        if not view.factor_decision:
            return self._target(view, current_position, None, "ofi_factor_wait")
        if score is None or not math.isfinite(float(score)):
            raise ValueError("OFI sign strategy requires a finite active_factor on decision ticks")
        score = float(score)
        desired = 1 if score > 0 else -1 if score < 0 else current_position
        if score == 0:
            reason = "ofi_hold_zero"
        elif current_position and desired != current_position:
            return self._target(view, 0, score, "ofi_flat_for_reversal", reduce_only=True)
        elif desired > 0:
            reason = "ofi_long" if current_position == 0 else "ofi_hold_long"
        elif desired < 0:
            reason = "ofi_short" if current_position == 0 else "ofi_hold_short"
        else:
            reason = "ofi_flat"
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


__all__ = ["OFISignStrategy"]

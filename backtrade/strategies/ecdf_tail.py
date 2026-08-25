from __future__ import annotations

import math

from backtrade.simulation.events import PortfolioTarget, StrategyView
from backtrade.strategies.factors import ECDF_TAIL_SEMANTICS_VERSION, validate_factor_name


class EcdfTailStrategy:
    """Map calibrated ECDF tails to flat or single-lot targets."""

    required_features = frozenset({"active_factor"})

    def __init__(self, factor_name: str, *, short_threshold: float, long_threshold: float) -> None:
        self.factor_name = validate_factor_name(factor_name)
        self.short_threshold = float(short_threshold)
        self.long_threshold = float(long_threshold)
        if not math.isfinite(self.short_threshold) or not math.isfinite(self.long_threshold):
            raise ValueError("ecdf tail thresholds must be finite")
        if self.short_threshold >= self.long_threshold:
            raise ValueError("short_threshold must be less than long_threshold")
        self.factor_semantics_version = ECDF_TAIL_SEMANTICS_VERSION

    def on_decision(self, view: StrategyView, current_position: int) -> PortfolioTarget:
        if current_position not in {-1, 0, 1}:
            raise ValueError("ecdf tail strategy requires current position in {-1, 0, +1}")
        score = view.factors.get("active_factor")
        if not view.factor_decision:
            return self._target(view, current_position, None, "ecdf_tail_factor_wait")
        if score is None or not math.isfinite(float(score)):
            raise ValueError("ecdf tail strategy requires a finite active_factor on decision ticks")
        score = float(score)
        desired = 1 if score >= self.long_threshold else -1 if score <= self.short_threshold else 0
        if current_position != 0 and desired != current_position:
            reason = "ecdf_tail_flat_for_reversal" if desired != 0 else "ecdf_tail_flat_for_neutral"
            return self._target(view, 0, score, reason, reduce_only=True)
        if desired > 0:
            reason = "ecdf_tail_long" if current_position == 0 else "ecdf_tail_hold_long"
        elif desired < 0:
            reason = "ecdf_tail_short" if current_position == 0 else "ecdf_tail_hold_short"
        else:
            reason = "ecdf_tail_flat"
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


__all__ = ["EcdfTailStrategy"]

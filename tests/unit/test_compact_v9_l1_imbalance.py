from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from backtrade.cli import _derive_factor, _prepare_input
from backtrade.config.schema import StrategyConfig
from backtrade.simulation.events import StrategyView
from backtrade.strategies.factors import L1_IMBALANCE_NAME, compute_l1_imbalance
from backtrade.strategies.signed_factor import SignedFactorStrategy


def test_l1_imbalance_uses_only_best_level_quantities() -> None:
    assert compute_l1_imbalance(75, 25) == pytest.approx(0.5)
    assert compute_l1_imbalance(25, 75) == pytest.approx(-0.5)
    assert compute_l1_imbalance(0, 0) == 0.0


def test_l1_imbalance_rejects_invalid_quantities() -> None:
    with pytest.raises(ValueError, match="finite"):
        compute_l1_imbalance(float("nan"), 1)
    with pytest.raises(ValueError, match="non-negative"):
        compute_l1_imbalance(-1, 1)


def test_strategy_config_accepts_l1_imbalance() -> None:
    cfg = StrategyConfig(factor_name=L1_IMBALANCE_NAME, factor_column=L1_IMBALANCE_NAME)
    assert cfg.factor_name == L1_IMBALANCE_NAME


def test_strategy_config_accepts_user_selected_factor_name() -> None:
    cfg = StrategyConfig(factor_name="my_ofi", factor_column="my_ofi")
    assert cfg.factor_name == "my_ofi"


def test_strategy_config_rejects_unsafe_factor_name() -> None:
    with pytest.raises(ValueError, match="factor name"):
        StrategyConfig(factor_name="../my_ofi", factor_column="../my_ofi")


def test_strategy_config_rejects_reserved_factor_name() -> None:
    with pytest.raises(ValueError, match="reserved"):
        StrategyConfig(factor_name="active_factor", factor_column="active_factor")


def test_l1_strategy_preserves_signed_target_semantics() -> None:
    strategy = SignedFactorStrategy(L1_IMBALANCE_NAME)
    view = StrategyView(
        product="ag",
        contract="AG2601",
        tick_ts=datetime(2026, 1, 5, 9),
        mid=100.0,
        factors={"active_factor": -0.25},
        factor_decision=True,
        factor_source_ts=datetime(2026, 1, 5, 9),
        factor_age_ms=0.0,
    )
    target = strategy.on_decision(view, current_position=0)
    assert (target.target_qty, target.factor_name, target.factor_score) == (-1, L1_IMBALANCE_NAME, -0.25)
    assert target.factor_semantics_version == "signed_factor_v1"


def test_signed_factor_strategy_accepts_user_selected_factor_name() -> None:
    strategy = SignedFactorStrategy("my_ofi")
    view = StrategyView(
        product="ag",
        contract="AG2601",
        tick_ts=datetime(2026, 1, 5, 9),
        mid=100.0,
        factors={"active_factor": 0.25},
        factor_decision=True,
        factor_source_ts=datetime(2026, 1, 5, 9),
        factor_age_ms=0.0,
    )
    target = strategy.on_decision(view, current_position=0)
    assert target.factor_name == "my_ofi"
    assert target.target_qty == 1


def test_prepare_input_accepts_explicit_l1_factor_column(tmp_path) -> None:
    market_path = tmp_path / "market.parquet"
    factor_path = tmp_path / "l1.parquet"
    _write_minimal_market(market_path)
    pd.DataFrame(
        {"tick_ts": [pd.Timestamp("2026-01-05 09:00:00")], L1_IMBALANCE_NAME: [0.25]}
    ).to_parquet(factor_path, index=False)
    result = _prepare_input(
        None,
        "ag",
        str(market_path),
        str(factor_path),
        factor_column=L1_IMBALANCE_NAME,
    )
    assert result["factor"] == str(factor_path)


def test_prepare_input_accepts_user_selected_factor_column(tmp_path) -> None:
    market_path = tmp_path / "market.parquet"
    factor_path = tmp_path / "my_ofi.parquet"
    _write_minimal_market(market_path)
    pd.DataFrame(
        {"tick_ts": [pd.Timestamp("2026-01-05 09:00:00")], "my_ofi": [0.25]}
    ).to_parquet(factor_path, index=False)
    result = _prepare_input(
        None,
        "ag",
        str(market_path),
        str(factor_path),
        factor_column="my_ofi",
    )
    assert result["factor_column"] == "my_ofi"


def test_custom_factor_manifest_is_accepted_by_runtime_validation(tmp_path) -> None:
    from backtrade.config.schema import BacktradeConfig
    from backtrade.data.future_l2 import _validate_factor_manifest

    market_path = tmp_path / "market.parquet"
    factor_path = tmp_path / "my_ofi.parquet"
    _write_minimal_market(market_path)
    pd.DataFrame(
        {"tick_ts": [pd.Timestamp("2026-01-05 09:00:00")], "my_ofi": [0.25]}
    ).to_parquet(factor_path, index=False)
    _prepare_input(
        None,
        "ag",
        str(market_path),
        str(factor_path),
        factor_column="my_ofi",
    )
    cfg = BacktradeConfig(
        initial_cash=1000.0,
        data={
            "product": "ag",
            "market_path": market_path,
            "factor_path": factor_path,
            "eof_is_day_end": True,
        },
        strategy={"factor_name": "my_ofi", "factor_column": "my_ofi"},
    )
    manifest = _validate_factor_manifest(cfg, factor_path)
    assert manifest["schema_version"] == "my_ofi_minimal_v1"


def _write_minimal_market(path) -> None:
    row = {
        "trading_day": "2026-01-05",
        "session_id": "day",
        "tick_ts": pd.Timestamp("2026-01-05 09:00:00"),
        "underlying_secu_cd": "AG2601",
        "last_prc": 100.0,
        "vol_inc": 1,
        "amt_inc": 100.0,
    }
    for level in range(1, 6):
        row[f"bid{level}_prc"] = 100.0 - level
        row[f"ask{level}_prc"] = 100.0 + level
        row[f"bid{level}_qty"] = 10
        row[f"ask{level}_qty"] = 10
    pd.DataFrame([row]).to_parquet(path, index=False)


def test_full_context_factor_is_exposed_as_active_factor() -> None:
    from backtrade.data.future_l2 import merge_market_and_factors

    common = {
        "product": "ag",
        "trading_day": "2026-01-05",
        "session_id": "day",
        "underlying_secu_cd": "AG2601",
        "tick_ts": pd.Timestamp("2026-01-05 09:00:00"),
    }
    market = pd.DataFrame(
        [{**common, "last_prc": 100.0, "bid1_prc": 99.0, "ask1_prc": 101.0}]
    )
    factors = pd.DataFrame([{**common, "l1_imbalance": 0.25}])
    aligned = merge_market_and_factors(
        market,
        factors,
        factor_name=L1_IMBALANCE_NAME,
    )
    assert aligned.loc[0, "active_factor"] == pytest.approx(0.25)


def test_custom_factor_name_flows_through_decision_grid() -> None:
    from backtrade.data.future_l2 import merge_market_and_factors

    common = {
        "product": "ag",
        "trading_day": "2026-01-05",
        "session_id": "day",
        "underlying_secu_cd": "AG2601",
        "tick_ts": pd.Timestamp("2026-01-05 09:00:00"),
    }
    market = pd.DataFrame(
        [{**common, "last_prc": 100.0, "bid1_prc": 99.0, "ask1_prc": 101.0}]
    )
    factors = pd.DataFrame([{**common, "my_ofi": -0.5}])
    aligned = merge_market_and_factors(market, factors, factor_name="my_ofi")
    assert aligned.loc[0, "active_factor"] == pytest.approx(-0.5)
    assert bool(aligned.loc[0, "factor_decision"])


def test_derive_factor_writes_recomputable_l1_values(tmp_path) -> None:
    market_path = tmp_path / "market.parquet"
    factor_path = tmp_path / "derived.parquet"
    _write_minimal_market(market_path)
    result = _derive_factor(str(market_path), str(factor_path), "ag", L1_IMBALANCE_NAME)
    assert result["rows"] == 1
    derived = pd.read_parquet(factor_path)
    assert derived.loc[0, L1_IMBALANCE_NAME] == pytest.approx(0.0)

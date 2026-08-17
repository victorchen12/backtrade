from datetime import datetime, timezone

import pandas as pd
import pytest

from backtrade.config.loader import load_config
from backtrade.config.schema import BacktradeConfig
from backtrade.data.future_l2 import JOIN_KEYS, merge_market_and_factors
from backtrade.simulation.compact_v9 import _maker_fill_rows
from backtrade.simulation.compact_v9_runner import CompactV9Result, CompactV9Runner


TS = datetime(2026, 1, 5, 9, 0, tzinfo=timezone.utc)


def _market(ts):
    return pd.DataFrame(
        [
            {
                "product": "ag",
                "trading_day": "2026-01-05",
                "session_id": "day",
                "tick_ts": ts,
                "underlying_secu_cd": "AG2604",
                "last_prc": 100.0,
            }
        ]
    )


def _factor(ts, score):
    return pd.DataFrame(
        [
            {
                "product": "ag",
                "trading_day": "2026-01-05",
                "session_id": "day",
                "tick_ts": ts,
                "underlying_secu_cd": "AG2604",
                "ofi_cks_best_level_5s": score,
                "active_factor": score,
            }
        ]
    )


def test_decision_grid_is_causal_and_only_exact_source_ticks_decide():
    frame = merge_market_and_factors(
        pd.concat([_market(TS), _market(TS.replace(microsecond=10_000))], ignore_index=True),
        _factor(TS, 2.0),
    )
    assert bool(frame.loc[0, "factor_decision"])
    assert frame.loc[0, "factor_source_ts"] == TS
    assert not bool(frame.loc[1, "factor_decision"])
    assert frame.loc[1, "active_factor"] == 2.0
    assert frame.loc[1, "factor_age_ms"] > 0


def test_old_position_filter_and_tolerance_fields_are_rejected():
    with pytest.raises(ValueError):
        BacktradeConfig.model_validate(
            {
                "initial_cash": 1000,
                "data": {"product": "ag", "eof_is_day_end": True},
                "strategy": {"position_filter": {"mode": "external_directional"}},
                "limit_reference": {"tolerance_ticks": 2},
            }
        )


def test_manifest_uses_resolved_default_input_paths(tmp_path):
    market = tmp_path / "pre_data" / "continuous_main_tick" / "ag_con_tick.parquet"
    factor = tmp_path / "factor_data" / "ag_5s" / "ofi_factor_values_keyed.parquet"
    market.parent.mkdir(parents=True)
    factor.parent.mkdir(parents=True)
    market.write_bytes(b"market")
    factor.write_bytes(b"factor")
    (factor.parent / "manifest.json").write_text("{}", encoding="utf-8")
    cfg = BacktradeConfig.model_validate(
        {
            "initial_cash": 1000,
            "paths": {"future_l2_data_root": str(tmp_path)},
            "data": {"product": "ag", "eof_is_day_end": True},
        }
    )
    runner = CompactV9Runner(cfg, [])
    result = CompactV9Result([], [], [], [], [], [], [], {"net_qty": {}})
    manifest = runner._manifest_payload(result)
    assert {"market", "factor", "factor_manifest"} <= set(manifest["input_identities"])
    assert manifest["input_identities"]["market"]["resolved_path"] == str(market.resolve())
    assert manifest["input_identities"]["factor"]["resolved_path"] == str(factor.resolve())


def test_maker_audit_excludes_forced_taker_flatten_fills():
    fills = [
        {"order_id": "maker-1", "maker_taker_role": "maker"},
        {"order_id": "forced-1", "maker_taker_role": "taker", "boundary_reason": "day_end_flatten"},
    ]
    assert [row["order_id"] for row in _maker_fill_rows(fills)] == ["maker-1"]


def test_loader_resolves_relative_contract_files_from_config_directory(tmp_path):
    contract_path = tmp_path / "contracts.yaml"
    config_path = tmp_path / "run.yaml"
    contract_path.write_text("contracts: {}\n", encoding="utf-8")
    config_path.write_text(
        "contract_files:\n  - contracts.yaml\n"
        "initial_cash: 1000\n"
        "data:\n  product: ag\n  eof_is_day_end: true\n",
        encoding="utf-8",
    )
    cfg = load_config(config_path)
    assert cfg.contract_files == [contract_path.resolve()]

from __future__ import annotations

from datetime import datetime

import hashlib
import json
import pandas as pd
import pytest

from backtrade.config.schema import BacktradeConfig
from backtrade.data.future_l2 import _validate_factor_manifest, enrich_factor_keys
from backtrade.data.market_quality import derive_market_quality_flags


TS = datetime(2026, 1, 5, 9, 0)


def _market_frame() -> pd.DataFrame:
    row = {
        "product": "ag",
        "trading_day": "2026-01-05",
        "session_id": "day",
        "tick_ts": TS,
        "underlying_secu_cd": "AG2604",
        "last_prc": 100.5,
        "vol_inc": 1.0,
        "amt_inc": 1507.5,
    }
    for level in range(1, 6):
        row[f"bid{level}_prc"] = 100.0 - level + 1
        row[f"ask{level}_prc"] = 101.0 + level - 1
        row[f"bid{level}_qty"] = 10.0
        row[f"ask{level}_qty"] = 12.0
    return pd.DataFrame([row])


def test_minimal_factor_values_inherit_market_context():
    factors = pd.DataFrame({"tick_ts": [TS], "l1_imbalance": [2.5]})

    result = enrich_factor_keys(_market_frame(), factors)

    assert result.loc[0, "product"] == "ag"
    assert result.loc[0, "trading_day"] == "2026-01-05"
    assert result.loc[0, "session_id"] == "day"
    assert result.loc[0, "underlying_secu_cd"] == "AG2604"


def test_minimal_factor_values_reject_duplicate_or_unknown_ticks():
    market = _market_frame()
    duplicate = pd.DataFrame({"tick_ts": [TS, TS], "l1_imbalance": [1.0, 2.0]})
    unknown = pd.DataFrame({"tick_ts": [TS.replace(minute=1)], "l1_imbalance": [1.0]})

    with pytest.raises(ValueError, match="duplicate factor tick_ts"):
        enrich_factor_keys(market, duplicate)
    with pytest.raises(ValueError, match="does not match a market tick"):
        enrich_factor_keys(market, unknown)


def test_market_quality_is_derived_from_raw_book():
    frame = _market_frame()
    result = derive_market_quality_flags(frame, tick_size=1.0)

    assert bool(result.loc[0, "side_ambiguous_flag"])
    assert not bool(result.loc[0, "is_anomaly"])

    crossed = frame.copy()
    crossed.loc[0, "ask2_prc"] = 99.0
    result = derive_market_quality_flags(crossed, tick_size=1.0)
    assert bool(result.loc[0, "is_anomaly"])


def test_existing_quality_flags_can_only_tighten_raw_quality():
    frame = _market_frame()
    frame["is_anomaly"] = False
    frame["side_ambiguous_flag"] = False
    frame.loc[0, "is_anomaly"] = True

    result = derive_market_quality_flags(frame, tick_size=1.0)

    assert bool(result.loc[0, "is_anomaly"])


def _manifest_config(market_path, factor_path, product="ag") -> BacktradeConfig:
    return BacktradeConfig.model_validate(
        {
            "initial_cash": 1000,
            "data": {
                "product": product,
                "market_path": str(market_path),
                "factor_path": str(factor_path),
                "max_ticks": 1,
                "eof_is_day_end": False,
            },
        }
    )


def _write_minimal_manifest(root, *, market_path, factor_path, schema_version, product="ag", include_market_hash=True):
    payload = {
        "schema_version": schema_version,
        "factor_columns": ["l1_imbalance"],
        "factor_values_path": str(factor_path),
        "factor_values_sha256": hashlib.sha256(factor_path.read_bytes()).hexdigest(),
    }
    if product is not None:
        payload["product"] = product
    if market_path is not None:
        payload["market_path"] = str(market_path)
    if include_market_hash:
        payload["market_sha256"] = hashlib.sha256(market_path.read_bytes()).hexdigest()
    (factor_path.parent / "manifest.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_minimal_manifest_binds_market_hash_and_product(tmp_path):
    market_a = tmp_path / "market_a.parquet"
    market_b = tmp_path / "market_b.parquet"
    factor = tmp_path / "l1_imbalance.parquet"
    market_a.write_bytes(b"market-a")
    market_b.write_bytes(b"market-b")
    factor.write_bytes(b"factor")
    _write_minimal_manifest(
        tmp_path,
        market_path=market_a,
        factor_path=factor,
        schema_version="l1_imbalance_minimal_v1",
    )

    _validate_factor_manifest(_manifest_config(market_a, factor), factor)
    with pytest.raises(ValueError, match="market.*hash"):
        _validate_factor_manifest(_manifest_config(market_b, factor), factor)

    _write_minimal_manifest(
        tmp_path,
        market_path=market_a,
        factor_path=factor,
        schema_version="l1_imbalance_minimal_v1",
        product="au",
    )
    with pytest.raises(ValueError, match="product"):
        _validate_factor_manifest(_manifest_config(market_a, factor, product="ag"), factor)


def test_legacy_manifest_without_market_hash_requires_exact_market_path(tmp_path):
    market_a = tmp_path / "market_a.parquet"
    market_b = tmp_path / "market_b.parquet"
    factor = tmp_path / "l1_imbalance.parquet"
    market_a.write_bytes(b"market-a")
    market_b.write_bytes(b"market-b")
    factor.write_bytes(b"factor")
    _write_minimal_manifest(
        tmp_path,
        market_path=market_a,
        factor_path=factor,
        schema_version="l1_imbalance_v1",
        product=None,
        include_market_hash=False,
    )

    _validate_factor_manifest(_manifest_config(market_a, factor), factor)
    with pytest.raises(ValueError, match="market.*path"):
        _validate_factor_manifest(_manifest_config(market_b, factor), factor)

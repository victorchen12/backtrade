from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from backtrade.cli import _prepare_input, build_parser
from backtrade.config.schema import BacktradeConfig
from backtrade.data.future_l2 import _iter_market_frames, _validate_factor_manifest, load_factor_frame_for_split


FACTOR_A = "v2p3_factor_a"
FACTOR_B = "v2p3_factor_b"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_market(path: Path, *, last_price: float = 100.0) -> None:
    row = {
        "product": "ag",
        "trading_day": "2025-04-15",
        "session_id": "day",
        "tick_ts": pd.Timestamp("2025-04-15 09:00:00"),
        "underlying_secu_cd": "AG2506",
        "last_prc": last_price,
        "vol_inc": 1,
        "amt_inc": last_price * 15.0,
    }
    for level in range(1, 6):
        row[f"bid{level}_prc"] = last_price - level
        row[f"ask{level}_prc"] = last_price + level
        row[f"bid{level}_qty"] = 10
        row[f"ask{level}_qty"] = 10
    pd.DataFrame([row]).to_parquet(path, index=False)


def _write_factors(path: Path, *, factor_b: float = -0.9) -> None:
    pd.DataFrame(
        {
            "split_id": ["015"],
            "part": ["test"],
            "product": ["ag"],
            "trading_day": ["2025-04-15"],
            "session_id": ["day"],
            "tick_ts": [pd.Timestamp("2025-04-15 09:00:00")],
            "underlying_secu_cd": ["AG2506"],
            FACTOR_A: [0.9],
            FACTOR_B: [factor_b],
        }
    ).to_parquet(path, index=False)


def _write_source_manifest(path: Path, factor_path: Path, *, factor_hash: str | None = None) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "rolling_source_v1",
                "calibrated_factor_file": str(factor_path),
                "calibrated_factor_file_sha256": factor_hash or _sha256(factor_path),
                "factor_columns": [FACTOR_A, FACTOR_B],
                "calibrated_factor_columns": [
                    "split_id",
                    "part",
                    "product",
                    "trading_day",
                    "session_id",
                    "tick_ts",
                    "underlying_secu_cd",
                    FACTOR_A,
                    FACTOR_B,
                ],
            }
        ),
        encoding="utf-8",
    )


def _prepare_bundle(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    market_path = tmp_path / "market.parquet"
    factor_path = tmp_path / "factors.parquet"
    source_manifest = tmp_path / "rolling_manifest.json"
    bundle_manifest = tmp_path / "input" / "factor_bundle_manifest.json"
    bundle_manifest.parent.mkdir()
    _write_market(market_path)
    _write_factors(factor_path)
    _write_source_manifest(source_manifest, factor_path)
    original_hash = _sha256(factor_path)

    result = _prepare_input(
        None,
        "ag",
        str(market_path),
        str(factor_path),
        factor_columns=[FACTOR_A, FACTOR_B],
        manifest_path_arg=str(bundle_manifest),
        source_manifest_path_arg=str(source_manifest),
    )

    assert result["factor_columns"] == [FACTOR_A, FACTOR_B]
    assert _sha256(factor_path) == original_hash
    return market_path, factor_path, source_manifest, bundle_manifest


def _bundle_config(
    market_path: Path,
    factor_path: Path,
    bundle_manifest: Path,
    *,
    product: str = "ag",
) -> BacktradeConfig:
    return BacktradeConfig.model_validate(
        {
            "initial_cash": 1_000_000,
            "data": {
                "product": product,
                "market_path": str(market_path),
                "factor_path": str(factor_path),
                "factor_manifest_path": str(bundle_manifest),
                "split_id": "015",
                "parts": ["test"],
                "max_ticks": 1,
                "eof_is_day_end": False,
            },
            "strategy": {"factor_name": FACTOR_A, "factor_column": FACTOR_A},
        }
    )


def test_prepare_input_writes_factor_bundle_manifest_without_rewriting_source_data(tmp_path: Path) -> None:
    market_path, factor_path, source_manifest, bundle_manifest = _prepare_bundle(tmp_path)

    payload = json.loads(bundle_manifest.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "factor_bundle_v1"
    assert payload["factor_columns"] == [FACTOR_A, FACTOR_B]
    assert payload["factor_values_path"] == str(factor_path)
    assert payload["factor_values_sha256"] == _sha256(factor_path)
    assert payload["market_path"] == str(market_path)
    assert payload["market_sha256"] == _sha256(market_path)
    assert payload["source_manifest_path"] == str(source_manifest)
    assert payload["source_manifest_sha256"] == _sha256(source_manifest)


def test_bundle_manifest_uses_explicit_path_and_reads_only_selected_factor(tmp_path: Path) -> None:
    market_path, factor_path, _, bundle_manifest = _prepare_bundle(tmp_path)
    cfg = _bundle_config(market_path, factor_path, bundle_manifest)

    manifest = _validate_factor_manifest(cfg, factor_path)
    frame = load_factor_frame_for_split(cfg)

    assert manifest["schema_version"] == "factor_bundle_v1"
    assert FACTOR_A in frame.columns
    assert FACTOR_B not in frame.columns
    assert frame[FACTOR_A].tolist() == [0.9]


def test_bundle_manifest_rejects_factor_hash_market_binding_and_product_mismatches(tmp_path: Path) -> None:
    market_path, factor_path, _, bundle_manifest = _prepare_bundle(tmp_path)
    cfg = _bundle_config(market_path, factor_path, bundle_manifest)
    factor_path.write_bytes(factor_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="factor.*hash"):
        _validate_factor_manifest(cfg, factor_path)

    other_root = tmp_path / "other"
    other_root.mkdir()
    other_market = other_root / "market.parquet"
    other_factor = other_root / "factors.parquet"
    other_source = other_root / "rolling_manifest.json"
    other_bundle = other_root / "bundle.json"
    _write_market(other_market, last_price=101.0)
    _write_factors(other_factor)
    _write_source_manifest(other_source, other_factor)
    _prepare_input(
        None,
        "ag",
        str(other_market),
        str(other_factor),
        factor_columns=[FACTOR_A, FACTOR_B],
        manifest_path_arg=str(other_bundle),
        source_manifest_path_arg=str(other_source),
    )
    with pytest.raises(ValueError, match="market.*path|market.*hash"):
        _validate_factor_manifest(_bundle_config(market_path, other_factor, other_bundle), other_factor)
    with pytest.raises(ValueError, match="product"):
        _validate_factor_manifest(_bundle_config(other_market, other_factor, other_bundle, product="au"), other_factor)


def test_prepare_input_rejects_nonfinite_bundle_columns_and_source_hash_mismatch(tmp_path: Path) -> None:
    market_path = tmp_path / "market.parquet"
    factor_path = tmp_path / "factors.parquet"
    source_manifest = tmp_path / "rolling_manifest.json"
    _write_market(market_path)
    _write_factors(factor_path, factor_b=float("nan"))
    _write_source_manifest(source_manifest, factor_path)
    with pytest.raises(ValueError, match=FACTOR_B):
        _prepare_input(
            None,
            "ag",
            str(market_path),
            str(factor_path),
            factor_columns=[FACTOR_A, FACTOR_B],
            manifest_path_arg=str(tmp_path / "bundle.json"),
            source_manifest_path_arg=str(source_manifest),
        )

    _write_factors(factor_path)
    _write_source_manifest(source_manifest, factor_path, factor_hash="0" * 64)
    with pytest.raises(ValueError, match="source manifest.*hash"):
        _prepare_input(
            None,
            "ag",
            str(market_path),
            str(factor_path),
            factor_columns=[FACTOR_A, FACTOR_B],
            manifest_path_arg=str(tmp_path / "bundle.json"),
            source_manifest_path_arg=str(source_manifest),
        )


def test_prepare_input_parser_accepts_bundle_options() -> None:
    args = build_parser().parse_args(
        [
            "prepare-input",
            "--product",
            "ag",
            "--factor-columns",
            FACTOR_A,
            FACTOR_B,
            "--manifest-path",
            "bundle.json",
            "--source-manifest-path",
            "rolling_manifest.json",
        ]
    )

    assert args.factor_columns == [FACTOR_A, FACTOR_B]
    assert args.manifest_path == "bundle.json"
    assert args.source_manifest_path == "rolling_manifest.json"


def test_market_reader_yields_bounded_batches_with_source_order(tmp_path: Path) -> None:
    market_path = tmp_path / "market.parquet"
    rows = []
    for offset in range(3):
        row_path = tmp_path / f"row_{offset}.parquet"
        _write_market(row_path, last_price=100.0 + offset)
        row = pd.read_parquet(row_path).iloc[0].copy()
        row["tick_ts"] = pd.Timestamp("2025-04-15 09:00:00") + pd.Timedelta(seconds=offset)
        rows.append(row)
    pd.DataFrame(rows).to_parquet(market_path, index=False, row_group_size=1)

    stream = _iter_market_frames(market_path, {"2025-04-15"}, product="ag", tick_size=1.0, batch_size=1)
    first = next(stream)
    second = next(stream)

    assert len(first) == 1
    assert len(second) == 1
    assert first["source_seq"].tolist() == [0]
    assert second["source_seq"].tolist() == [1]

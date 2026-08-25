from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from backtrade.cli import _prepare_input, build_parser
from backtrade.config.schema import BacktradeConfig, DataSourceConfig
from backtrade.data.future_l2 import (
    _iter_market_frames,
    _book_quantity,
    _validate_factor_manifest,
    iter_future_l2_ticks,
    load_factor_frame_for_split,
)


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
            "contracts": {
                "ag": {
                    "code": "AG",
                    "exchange": "SHFE",
                    "multiplier": 15.0,
                    "tick_size": 1.0,
                    "fee": {
                        "open": {"mode": "rate", "value": 0.00005},
                        "close": {"mode": "rate", "value": 0.00005},
                        "close_today": {"mode": "rate", "value": 0.00005},
                    },
                    "price_limit": {"mode": "percent", "value": 0.2},
                }
            },
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

def test_data_source_rejects_ambiguous_or_empty_split_selection() -> None:
    with pytest.raises(ValueError, match="split_id"):
        DataSourceConfig(
            product="ag", split_id="001", split_ids=["002"], max_ticks=1
        )
    with pytest.raises(ValueError, match="split_ids"):
        DataSourceConfig(product="ag", split_ids=[], max_ticks=1)
    with pytest.raises(ValueError, match="ascending"):
        DataSourceConfig(product="ag", split_ids=["002", "001"], max_ticks=1)
    with pytest.raises(ValueError, match="duplicate"):
        DataSourceConfig(product="ag", split_ids=["1", "001"], max_ticks=1)
    normalized = DataSourceConfig(product="ag", split_ids=["1", "002"], max_ticks=1)
    assert normalized.split_ids == ["001", "002"]

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


def test_full_split_stream_selects_one_split_at_a_time_without_resetting_order(
    tmp_path: Path, monkeypatch
) -> None:
    market_path = tmp_path / "market.parquet"
    factor_path = tmp_path / "factors.parquet"
    bundle_path = tmp_path / "bundle.json"
    source_path = tmp_path / "source.json"
    market_rows = []
    factor_rows = []
    for index, day in enumerate(("2025-04-15", "2025-04-16"), start=1):
        row = {
            "product": "ag",
            "trading_day": day,
            "session_id": "day",
            "tick_ts": pd.Timestamp(f"{day} 09:00:00"),
            "underlying_secu_cd": "AG2506",
            "last_prc": 100.0 + index,
            "vol_inc": 1,
            "amt_inc": (100.0 + index) * 15.0,
        }
        for level in range(1, 6):
            row[f"bid{level}_prc"] = row["last_prc"] - level
            row[f"ask{level}_prc"] = row["last_prc"] + level
            row[f"bid{level}_qty"] = 10
            row[f"ask{level}_qty"] = 10
        market_rows.append(row)
        factor_rows.append(
            {
                "split_id": f"{index:03d}",
                "part": "test",
                "product": "ag",
                "trading_day": day,
                "session_id": "day",
                "tick_ts": row["tick_ts"],
                "underlying_secu_cd": "AG2506",
                FACTOR_A: 0.9 if index == 1 else -0.9,
                FACTOR_B: 0.1,
            }
        )
    pd.DataFrame(market_rows).to_parquet(market_path, index=False)
    pd.DataFrame(factor_rows).to_parquet(factor_path, index=False)
    _write_source_manifest(source_path, factor_path)
    _prepare_input(
        None,
        "ag",
        str(market_path),
        str(factor_path),
        factor_columns=[FACTOR_A, FACTOR_B],
        manifest_path_arg=str(bundle_path),
        source_manifest_path_arg=str(source_path),
    )
    template = _bundle_config(market_path, factor_path, bundle_path)
    cfg = template.model_copy(
        update={
            "data": template.data.model_copy(
                update={"split_id": None, "split_ids": ["001", "002"], "max_ticks": None, "eof_is_day_end": True}
            )
        }
    )
    seen_splits: list[str | None] = []
    from backtrade.data import future_l2

    original_loader = future_l2.load_factor_frame_for_split

    def recording_loader(config, *args, **kwargs):
        seen_splits.append(config.data.split_id)
        return original_loader(config, *args, **kwargs)


    monkeypatch.setattr(future_l2, "load_factor_frame_for_split", recording_loader)
    ticks = list(iter_future_l2_ticks(cfg, batch_size=1))

    assert seen_splits == ["001", "002"]
    assert [tick.trading_day for tick in ticks] == ["2025-04-15", "2025-04-16"]
    assert [tick.factors["active_factor"] for tick in ticks] == [0.9, -0.9]


def test_multi_split_stream_rejects_factor_input_without_split_id(tmp_path: Path) -> None:
    market_path, factor_path, source_manifest, bundle_manifest = _prepare_bundle(tmp_path)
    factors = pd.read_parquet(factor_path).drop(columns=["split_id"])
    factors.to_parquet(factor_path, index=False)
    source_manifest.unlink()
    _write_source_manifest(source_manifest, factor_path)
    bundle_manifest.unlink()
    _prepare_input(
        None,
        "ag",
        str(market_path),
        str(factor_path),
        factor_columns=[FACTOR_A, FACTOR_B],
        manifest_path_arg=str(bundle_manifest),
        source_manifest_path_arg=str(source_manifest),
    )
    template = _bundle_config(market_path, factor_path, bundle_manifest)
    cfg = template.model_copy(
        update={
            "data": template.data.model_copy(
                update={"split_id": None, "split_ids": ["015"], "max_ticks": None, "eof_is_day_end": True}
            )
        }
    )
    with pytest.raises(ValueError, match="split_id"):
        next(iter_future_l2_ticks(cfg, batch_size=1))


def test_single_split_minimal_factor_input_keeps_legacy_selection(tmp_path: Path) -> None:
    market_path, factor_path, source_manifest, bundle_manifest = _prepare_bundle(tmp_path)
    pd.read_parquet(factor_path).drop(columns=["split_id"]).to_parquet(factor_path, index=False)
    source_manifest.unlink()
    _write_source_manifest(source_manifest, factor_path)
    bundle_manifest.unlink()
    _prepare_input(
        None, "ag", str(market_path), str(factor_path),
        factor_columns=[FACTOR_A, FACTOR_B],
        manifest_path_arg=str(bundle_manifest),
        source_manifest_path_arg=str(source_manifest),
    )
    cfg = _bundle_config(market_path, factor_path, bundle_manifest)
    tick = next(iter_future_l2_ticks(cfg, batch_size=1))
    assert tick.trading_day == "2025-04-15"

def _replay_tick(contract: str, trading_day: str, seq: int):
    from backtrade.simulation.events import MarketTick

    return MarketTick(
        product="ag",
        contract=contract,
        tick_ts=pd.Timestamp("2026-01-05 09:00:00") + pd.Timedelta(seconds=seq),
        last_price=100.0,
        bid_prices=(99.0, 98.0, 97.0, 96.0, 95.0),
        bid_qtys=(5, 5, 5, 5, 5),
        ask_prices=(101.0, 102.0, 103.0, 104.0, 105.0),
        ask_qtys=(5, 5, 5, 5, 5),
        trading_day=trading_day,
        source_seq=seq,
    )

def test_replay_allows_contract_reopen_at_new_trading_day() -> None:
    from backtrade.data.replay import MarketReplay

    rows = list(MarketReplay([
        _replay_tick("AG2408", "2026-01-05", 1),
        _replay_tick("AG2502", "2026-01-06", 2),
        _replay_tick("AG2408", "2026-01-07", 3),
    ], closing_window_ms=0))

    assert len(rows) == 3
    assert rows[0][1].is_last_tick_of_contract is True
    assert rows[1][1].is_last_tick_of_contract is True

def test_replay_rejects_contract_reopen_within_one_trading_day() -> None:
    from backtrade.data.replay import MarketReplay

    with pytest.raises(ValueError, match="reopens closed contract"):
        list(MarketReplay([
            _replay_tick("AG2408", "2026-01-05", 1),
            _replay_tick("AG2502", "2026-01-05", 2),
            _replay_tick("AG2408", "2026-01-05", 3),
        ], closing_window_ms=0))

def test_book_quantity_preserves_anomaly_and_rejects_invalid_clean_values() -> None:
    assert _book_quantity(float("nan"), field="bid1_qty") == 0
    assert _book_quantity(None, field="ask1_qty") == 0
    with pytest.raises(ValueError, match="non-negative integer"):
        _book_quantity(1.5, field="bid1_qty")
    with pytest.raises(ValueError, match="non-negative integer"):
        _book_quantity(-1, field="ask1_qty")

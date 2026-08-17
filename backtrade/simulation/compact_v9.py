from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq


COMPACT_V9_SCHEMA_VERSION = "compact_v9"
TIMESTAMP = pa.timestamp("us")

ACTIVITY_SCHEMA = pa.schema(
    [
        ("event_seq", pa.int64()), ("source_dataset", pa.string()), ("record_type", pa.string()),
        ("event_ts", TIMESTAMP), ("target_seq", pa.int64()), ("order_seq", pa.int64()), ("fill_seq", pa.int64()),
        ("order_id", pa.string()), ("product", pa.string()), ("contract", pa.string()), ("trading_day", pa.string()),
        ("side", pa.string()), ("target_qty_raw", pa.float64()), ("target_qty", pa.int64()),
        ("status", pa.string()), ("reason_code", pa.string()), ("risk_state", pa.string()),
        ("factor_name", pa.string()), ("factor_score", pa.float64()), ("factor_semantics_version", pa.string()),
        ("factor_decision", pa.bool_()), ("factor_source_ts", TIMESTAMP), ("factor_age_ms", pa.float64()),
        ("arrival_bid1", pa.float64()), ("arrival_ask1", pa.float64()), ("arrival_price", pa.float64()),
        ("fill_price", pa.float64()), ("qty", pa.int64()), ("fee", pa.float64()), ("gross_pnl", pa.float64()),
        ("net_pnl", pa.float64()), ("maker_taker_role", pa.string()), ("direct_spread_ticks", pa.float64()),
        ("direct_spread_bps", pa.float64()), ("liquidity_source", pa.string()), ("boundary_reason", pa.string()),
    ]
)

ACCOUNT_SCHEMA = pa.schema(
    [
        ("account_event_seq", pa.int64()), ("event_ts", TIMESTAMP), ("fill_seq", pa.int64()),
        ("order_seq", pa.int64()), ("order_id", pa.string()), ("product", pa.string()), ("contract", pa.string()),
        ("trading_day", pa.string()), ("side", pa.string()), ("open_fill_seq", pa.int64()),
        ("open_order_seq", pa.int64()), ("open_order_id", pa.string()), ("position_before", pa.int64()),
        ("position_after", pa.int64()), ("cash_before", pa.float64()), ("cash_after", pa.float64()),
        ("equity_before", pa.float64()), ("equity_after", pa.float64()), ("realized_pnl_before", pa.float64()),
        ("realized_pnl_after", pa.float64()), ("unrealized_pnl_before", pa.float64()),
        ("unrealized_pnl_after", pa.float64()), ("total_fee_before", pa.float64()), ("total_fee_after", pa.float64()),
        ("gross_pnl", pa.float64()), ("open_fee", pa.float64()), ("close_fee", pa.float64()),
        ("net_pnl", pa.float64()), ("holding_ms", pa.int64()), ("reason_code", pa.string()),
    ]
)

STATE_SCHEMA = pa.schema(
    [
        ("snapshot_seq", pa.int64()), ("event_ts", TIMESTAMP), ("cash", pa.float64()), ("equity", pa.float64()),
        ("realized_pnl", pa.float64()), ("unrealized_pnl", pa.float64()), ("total_fee", pa.float64()),
        ("position_qty", pa.int64()), ("product", pa.string()), ("contract", pa.string()), ("mark_price", pa.float64()),
    ]
)

MAKER_SCHEMA = pa.schema(
    [
        ("maker_event_seq", pa.int64()), ("event_ts", TIMESTAMP), ("order_id", pa.string()),
        ("product", pa.string()), ("contract", pa.string()), ("side", pa.string()), ("price", pa.float64()),
        ("trade_price", pa.float64()), ("event_type", pa.string()), ("reason_code", pa.string()),
        ("queue_ahead_before", pa.float64()), ("queue_ahead_after", pa.float64()), ("depth_before", pa.float64()),
        ("depth_after", pa.float64()), ("opposite_l1_price", pa.float64()), ("opposite_l1_qty", pa.float64()),
        ("same_price_trade_qty", pa.int64()), ("probability_ahead", pa.float64()), ("direction_source", pa.string()),
        ("direction_confidence", pa.string()), ("data_quality", pa.string()), ("source_seq", pa.int64()),
        ("session_id", pa.string()), ("trading_day", pa.string()),
    ]
)

SCHEMAS = {"activity_ledger": ACTIVITY_SCHEMA, "account_ledger": ACCOUNT_SCHEMA, "state_snapshots": STATE_SCHEMA, "maker_events": MAKER_SCHEMA}


def compact_v9_filenames(*, maker_enabled: bool) -> set[str]:
    names = {"activity_ledger.parquet", "account_ledger.parquet", "state_snapshots.parquet", "manifest.json"}
    if maker_enabled:
        names.add("maker_events.parquet")
    return names


def _scalar(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, default=str)
    return value


def _record(schema: pa.Schema, value: dict[str, Any]) -> dict[str, Any]:
    return {field.name: _scalar(value.get(field.name)) for field in schema}


class CompactV9ParquetOutput:
    compact_schema_version = COMPACT_V9_SCHEMA_VERSION
    compact = True
    streaming = True

    def __init__(self, output_root: str | Path, *, maker_enabled: bool = False, batch_size: int = 4096):
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.maker_enabled = bool(maker_enabled)
        self.batch_size = int(batch_size)
        self.buffers = {name: [] for name in ("activity_ledger", "account_ledger", "state_snapshots")}
        if self.maker_enabled:
            self.buffers["maker_events"] = []
        self.writers: dict[str, pq.ParquetWriter] = {}
        self._event_seq = self._account_seq = self._snapshot_seq = self._maker_seq = 1

    def write(self, dataset: str, record: dict[str, Any]) -> None:
        mapping = {"activity": ("activity_ledger", None), "target": ("activity_ledger", "target"), "order": ("activity_ledger", "order"), "order_event": ("activity_ledger", "order_event"), "fill": ("activity_ledger", "fill"), "account": ("account_ledger", None), "snapshot": ("state_snapshots", None), "maker_event": ("maker_events", None)}
        if dataset not in mapping:
            raise KeyError(f"unknown compact_v9 source dataset: {dataset}")
        physical, record_type = mapping[dataset]
        if physical == "maker_events" and not self.maker_enabled:
            raise ValueError("maker events are only valid for maker runs")
        item = dict(record)
        if physical == "maker_events" and item.get("event_type") == "noop":
            raise ValueError("maker_events do not record no-op ticks")
        if physical == "activity_ledger":
            item.setdefault("record_type", record_type or item.get("source_dataset"))
            item.setdefault("source_dataset", item.get("record_type"))
            item.setdefault("event_seq", self._event_seq)
            self._event_seq = max(self._event_seq, int(item["event_seq"]) + 1)
        elif physical == "account_ledger":
            item.setdefault("account_event_seq", self._account_seq)
            self._account_seq = max(self._account_seq, int(item["account_event_seq"]) + 1)
        elif physical == "state_snapshots":
            item.setdefault("snapshot_seq", self._snapshot_seq)
            self._snapshot_seq = max(self._snapshot_seq, int(item["snapshot_seq"]) + 1)
        else:
            item.setdefault("maker_event_seq", self._maker_seq)
            self._maker_seq = max(self._maker_seq, int(item["maker_event_seq"]) + 1)
        self.buffers[physical].append(_record(SCHEMAS[physical], item))
        if len(self.buffers[physical]) >= self.batch_size:
            self.flush(physical)

    def write_many(self, dataset: str, records: Iterable[dict[str, Any]]) -> None:
        for record in records:
            self.write(dataset, record)

    def flush(self, dataset: str) -> None:
        records = self.buffers[dataset]
        if not records:
            return
        writer = self.writers.get(dataset)
        if writer is None:
            writer = pq.ParquetWriter(self.output_root / f"{dataset}.parquet", SCHEMAS[dataset], compression="zstd")
            self.writers[dataset] = writer
        writer.write_table(pa.Table.from_pylist(records, schema=SCHEMAS[dataset]))
        records.clear()

    def close(self, *, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
        for dataset in self.buffers:
            self.flush(dataset)
        for writer in self.writers.values():
            writer.close()
        for dataset, schema in SCHEMAS.items():
            if dataset == "maker_events" and not self.maker_enabled:
                continue
            path = self.output_root / f"{dataset}.parquet"
            if not path.exists():
                pq.write_table(pa.Table.from_pylist([], schema=schema), path, compression="zstd")
        files = compact_v9_filenames(maker_enabled=self.maker_enabled) - {"manifest.json"}
        file_hashes = {name: hashlib.sha256((self.output_root / name).read_bytes()).hexdigest() for name in sorted(files)}
        payload = dict(manifest or {})
        payload.update({
            "artifact_schema_version": COMPACT_V9_SCHEMA_VERSION,
            "schema_version": COMPACT_V9_SCHEMA_VERSION,
            "single_lot": True,
            "position_limit": 1,
            "margin_enabled": False,
            "maker_model_version": "mbp_prob_queue_v2_strict_through" if self.maker_enabled else None,
            "mbp_estimation": self.maker_enabled,
            "fifo_reconstruction": False,
            "data_files": sorted(compact_v9_filenames(maker_enabled=self.maker_enabled)),
            "file_hashes": file_hashes,
        })
        (self.output_root / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
        return payload


def _read_table(root: Path, name: str, schema: pa.Schema) -> pa.Table:
    table = pq.read_table(root / f"{name}.parquet")
    if table.schema.names != schema.names or any(table.schema.field(i).type != schema.field(i).type for i in range(len(schema))):
        raise ValueError(f"{name}: parquet schema does not match compact_v9")
    return table


def read_compact_v9(output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("artifact_schema_version") != COMPACT_V9_SCHEMA_VERSION:
        raise ValueError("legacy artifact schema is not accepted")
    mode = manifest.get("match_mode")
    if mode not in {"maker", "taker"}:
        raise ValueError("compact_v9 manifest must declare match_mode=maker or taker")
    expected = compact_v9_filenames(maker_enabled=mode == "maker")
    actual = {path.name for path in root.iterdir()}
    if actual != expected:
        raise ValueError(f"compact_v9 file set mismatch: expected={sorted(expected)} actual={sorted(actual)}")
    hashes = manifest.get("file_hashes")
    if not isinstance(hashes, dict) or set(hashes) != expected - {"manifest.json"} or any(not isinstance(value, str) for value in hashes.values()):
        raise ValueError("compact_v9 manifest must declare exact file_hashes")
    for filename, expected_hash in hashes.items():
        if hashlib.sha256((root / filename).read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"compact_v9 output hash mismatch: {filename}")
    if set(manifest.get("data_files", [])) != expected:
        raise ValueError("compact_v9 manifest data_files disagrees with inventory")
    if manifest.get("single_lot") is not True or manifest.get("position_limit") != 1 or manifest.get("margin_enabled") is not False:
        raise ValueError("compact_v9 requires single_lot=true, position_limit=1, margin_enabled=false")
    config = manifest.get("config")
    digest = manifest.get("config_digest")
    if config is not None:
        encoded = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode()
        if digest != hashlib.sha256(encoded).hexdigest():
            raise ValueError("compact_v9 config_digest does not match config")
    identities = manifest.get("input_identities")
    if not isinstance(identities, dict) or not identities:
        raise ValueError("compact_v9 input_identities must be a non-empty mapping of file identities")
    for name, identity in identities.items():
        if not isinstance(name, str) or not isinstance(identity, dict):
            raise ValueError("compact_v9 input_identities must map names to identity objects")
        if identity.get("exists") is not True:
            raise ValueError(f"compact_v9 input identity is missing: {name}")
        resolved = Path(str(identity.get("resolved_path", "")))
        if not resolved.is_file():
            raise ValueError(f"compact_v9 input identity path is missing: {name}")
        expected_size = identity.get("size_bytes")
        expected_hash = identity.get("sha256")
        if not isinstance(expected_size, int) or expected_size != resolved.stat().st_size:
            raise ValueError(f"compact_v9 input size mismatch: {name}")
        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if not isinstance(expected_hash, str) or digest.hexdigest() != expected_hash:
            raise ValueError(f"compact_v9 input hash mismatch: {name}")
    _read_table(root, "activity_ledger", ACTIVITY_SCHEMA)
    _read_table(root, "account_ledger", ACCOUNT_SCHEMA)
    _read_table(root, "state_snapshots", STATE_SCHEMA)
    if mode == "maker":
        if manifest.get("maker_model_version") != "mbp_prob_queue_v2_strict_through" or manifest.get("mbp_estimation") is not True or manifest.get("fifo_reconstruction") is not False:
            raise ValueError("maker compact_v9 manifest must declare strict MBP estimation and no FIFO reconstruction")
        _read_table(root, "maker_events", MAKER_SCHEMA)
    return manifest


def _maker_fill_rows(fills: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in fills if row.get("maker_taker_role") == "maker"]


def audit_compact_v9(output_root: str | Path, *, require_fills: bool = False, require_final_flat: bool = False) -> dict[str, Any]:
    root = Path(output_root)
    try:
        manifest = read_compact_v9(root)
    except Exception as exc:
        return {"status": "fail", "passed": False, "errors": [str(exc)], "artifact_schema_version": None, "require_fills_blocked": False}
    errors: list[str] = []
    activity = _read_table(root, "activity_ledger", ACTIVITY_SCHEMA).to_pylist()
    accounts = _read_table(root, "account_ledger", ACCOUNT_SCHEMA).to_pylist()
    snapshots = _read_table(root, "state_snapshots", STATE_SCHEMA).to_pylist()
    maker = _read_table(root, "maker_events", MAKER_SCHEMA).to_pylist() if manifest["match_mode"] == "maker" else []
    fills = [row for row in activity if row.get("record_type") == "fill"]
    orders = [row for row in activity if row.get("record_type") == "order"]
    targets = [row for row in activity if row.get("record_type") == "target"]
    if [row.get("event_seq") for row in activity] != list(range(1, len(activity) + 1)):
        errors.append("activity event_seq is not monotonic")
    if [row.get("account_event_seq") for row in accounts] != list(range(1, len(accounts) + 1)):
        errors.append("account sequence is not monotonic")
    if [row.get("snapshot_seq") for row in snapshots] != list(range(1, len(snapshots) + 1)):
        errors.append("snapshot sequence is not monotonic")
    if maker and [row.get("maker_event_seq") for row in maker] != list(range(1, len(maker) + 1)):
        errors.append("maker event sequence is not monotonic")
    if any(row.get("factor_semantics_version") not in {None, "ofi_sign_v1"} for row in targets):
        errors.append("target factor semantics is not ofi_sign_v1")
    if any(row.get("target_qty") not in {None, -1, 0, 1} for row in targets):
        errors.append("target quantity is outside -1/0/+1")
    fill_seqs = [row.get("fill_seq") for row in fills]
    if len(fill_seqs) != len(set(fill_seqs)) or any(value is None for value in fill_seqs):
        errors.append("fill sequence is missing or duplicated")
    if sorted(fill_seqs) != sorted(row.get("fill_seq") for row in accounts):
        errors.append("account ledger is not one-to-one with fills")
    if any(row.get("qty") != 1 for row in fills + orders):
        errors.append("non-single-lot order or fill found")

    def close(left: Any, right: Any) -> bool:
        try:
            return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-6)
        except (TypeError, ValueError):
            return False

    for row in accounts:
        before = int(row.get("position_before") or 0)
        after = int(row.get("position_after") or 0)
        side = row.get("side")
        expected = 1 if side == "buy" else -1 if side == "sell" else None
        if expected is None or after - before != expected or abs(after) > 1:
            errors.append(f"invalid position transition for fill {row.get('fill_seq')}")
        opening = before == 0
        expected_net = -float(row.get("open_fee") or 0.0) if opening else float(row.get("gross_pnl") or 0.0) - float(row.get("close_fee") or 0.0)
        if not close(row.get("net_pnl"), expected_net) or not close(float(row.get("cash_after") or 0.0) - float(row.get("cash_before") or 0.0), expected_net):
            errors.append(f"cash/net_pnl mismatch for fill {row.get('fill_seq')}")
        if not close(float(row.get("total_fee_after") or 0.0) - float(row.get("total_fee_before") or 0.0), float(row.get("open_fee") or 0.0) + float(row.get("close_fee") or 0.0)):
            errors.append(f"fee transition mismatch for fill {row.get('fill_seq')}")
    if accounts:
        initial_cash = float((manifest.get("config") or {}).get("initial_cash", accounts[0].get("cash_before", 0.0)))
        final_cash = float(accounts[-1].get("cash_after") or 0.0)
        final_realized = float(accounts[-1].get("realized_pnl_after") or 0.0)
        final_fee = float(accounts[-1].get("total_fee_after") or 0.0)
        net_sum = sum(float(row.get("net_pnl") or 0.0) for row in accounts)
        if not close(net_sum, final_cash - initial_cash) or not close(net_sum, final_realized - final_fee):
            errors.append("cash/PnL conservation failed")
    order_map = {row.get("order_id"): row for row in orders}
    latency_ms = float((manifest.get("latency") or {}).get("configured_ms", 0.0))
    for row in fills:
        order = order_map.get(row.get("order_id"))
        if order is None:
            errors.append(f"fill references unknown order {row.get('order_id')}")
            continue
        boundary = row.get("boundary_reason") or order.get("boundary_reason")
        decision_ts = order.get("event_ts")
        event_ts = row.get("event_ts")
        if boundary is None and decision_ts is not None and event_ts is not None:
            if event_ts < decision_ts + timedelta(milliseconds=latency_ms):
                errors.append(f"fill precedes decision plus latency for {row.get('order_id')}")
        if row.get("maker_taker_role") == "taker":
            expected = row.get("arrival_ask1") if row.get("side") == "buy" else row.get("arrival_bid1")
            if expected is None or not close(row.get("fill_price"), expected):
                errors.append(f"taker fill price differs from arrival L1 for {row.get('order_id')}")
    if manifest["match_mode"] == "maker":
        maker_fills = [row for row in maker if row.get("event_type") == "fill"]
        activity_maker_fills = _maker_fill_rows(fills)
        if {row.get("order_id") for row in maker_fills} != {row.get("order_id") for row in activity_maker_fills}:
            errors.append("maker fill events and activity maker fills disagree")
        for row in maker:
            order = order_map.get(row.get("order_id"))
            if order is None:
                continue
            if order.get("boundary_reason") is None and row.get("event_ts") is not None and order.get("event_ts") is not None:
                if row["event_ts"] < order["event_ts"] + timedelta(milliseconds=latency_ms):
                    errors.append(f"maker event precedes decision plus latency for {row.get('order_id')}")
        for row in maker_fills:
            price = row.get("price")
            trade_price = row.get("trade_price")
            side = row.get("side")
            queue = float(row.get("queue_ahead_before") or 0.0)
            same = float(row.get("same_price_trade_qty") or 0.0)
            equality = close(trade_price, price) and same >= queue and same > 0
            through = trade_price is not None and price is not None and ((side == "buy" and float(trade_price) < float(price)) or (side == "sell" and float(trade_price) > float(price)))
            if row.get("data_quality") != "normal" or not (equality or through):
                errors.append(f"maker fill lacks strict queue/trade evidence for {row.get('order_id')}")
    final_flat = bool(snapshots) and snapshots[-1].get("position_qty") == 0
    if require_fills and not fills:
        errors.append("no fills present")
    if require_final_flat and not final_flat:
        errors.append("final position is not flat")
    status = "fail" if errors else "pass"
    return {"status": status, "passed": status == "pass", "errors": errors, "artifact_schema_version": manifest.get("artifact_schema_version"), "orders_count": len(orders), "fills_count": len(fills), "account_event_count": len(accounts), "final_flat": final_flat, "require_fills_blocked": bool(require_fills and not fills)}


__all__ = ["ACCOUNT_SCHEMA", "ACTIVITY_SCHEMA", "COMPACT_V9_SCHEMA_VERSION", "MAKER_SCHEMA", "STATE_SCHEMA", "CompactV9ParquetOutput", "audit_compact_v9", "compact_v9_filenames", "read_compact_v9"]

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from backtrade.data.future_l2 import (
    JOIN_KEYS,
    MARKET_COLUMNS,
    _validate_factor_manifest,
    processed_market_path,
    selected_factor_screen_path,
)


def validate_config(cfg) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    inputs: dict[str, Any] = {}

    if cfg.data.source != "future_l2":
        errors.append("only data.source=future_l2 is supported")
    if cfg.match.mode not in {"maker", "taker"}:
        errors.append("only match.mode=maker or taker is supported")
    if cfg.strategy.factor_name != "ofi_cks_best_level_5s" or cfg.strategy.factor_column != "ofi_cks_best_level_5s":
        errors.append("only canonical ofi_cks_best_level_5s is supported")
    if cfg.limit_reference.mode == "prev_day_vwap_proxy":
        warnings.append("prev_day_vwap_proxy is an explicit approximation, not an official settlement price")
    if cfg.data.max_ticks is None and not cfg.data.eof_is_day_end:
        errors.append("unbounded runs must declare data.eof_is_day_end=true")

    for name, path in (
        ("market", processed_market_path(cfg)),
        ("factor", selected_factor_screen_path(cfg)),
    ):
        path = Path(path)
        entry: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
        inputs[name] = entry
        if not path.is_file():
            errors.append(f"{name} input does not exist: {path}")
            continue
        try:
            schema_names = set(pq.ParquetFile(path).schema.names)
            entry["columns"] = sorted(schema_names)
            required = set(MARKET_COLUMNS if name == "market" else [*JOIN_KEYS, "part", "split_id", "active_factor", "ofi_cks_best_level_5s"])
            missing = sorted(required - schema_names)
            if missing:
                errors.append(f"{name} input is missing columns: {missing}")
            if name == "factor":
                _validate_factor_manifest(cfg, path)
        except Exception as exc:
            errors.append(f"{name} input validation failed: {exc}")

    try:
        cfg.require_contract_for_real_run()
    except ValueError as exc:
        errors.append(str(exc))

    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "inputs": inputs,
        "source": cfg.data.source,
        "product": cfg.data.product,
        "strategy_mode": "ofi_sign",
        "factor_name": cfg.strategy.factor_name,
    }


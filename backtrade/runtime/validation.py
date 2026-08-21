from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from backtrade.strategies.factors import validate_factor_name
from backtrade.data.future_l2 import (
    JOIN_KEYS,
    MARKET_COLUMNS,
    _validate_factor_manifest,
    processed_market_path,
    selected_factor_screen_path,
)


def validate_config(cfg) -> dict[str, Any]:
    # [README-8] validate 只做运行前结构、文件、manifest 和契约检查；通过后才进入回放。
    errors: list[str] = []
    warnings: list[str] = []
    inputs: dict[str, Any] = {}

    if cfg.data.source != "future_l2":
        errors.append("only data.source=future_l2 is supported")
    if cfg.match.mode not in {"maker", "taker"}:
        errors.append("only match.mode=maker or taker is supported")
    try:
        validate_factor_name(cfg.strategy.factor_name)
    except ValueError as exc:
        errors.append(str(exc))
    if cfg.strategy.factor_column != cfg.strategy.factor_name:
        errors.append(f"factor configuration names must match: {cfg.strategy.factor_name}/{cfg.strategy.factor_column}")
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
            required = set(MARKET_COLUMNS if name == "market" else ["tick_ts", cfg.strategy.factor_column])
            missing = sorted(required - schema_names)
            if missing:
                errors.append(f"{name} input is missing columns: {missing}")
            if name == "factor":
                factor_context = set(JOIN_KEYS) - {"tick_ts"}
                present_context = factor_context & schema_names
                if present_context and present_context != factor_context:
                    errors.append("factor input must contain complete JOIN_KEYS or only tick_ts")
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
        "strategy_mode": "signed_factor",
        "factor_name": cfg.strategy.factor_name,
    }


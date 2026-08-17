from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from backtrade.config.schema import BacktradeConfig


PROJECT_WRITE_ROOT = Path("/home/cws/QUANT/Backtrade")
OUTPUT_WRITE_ROOT = Path("/data1/cws/backtrade")
FORBIDDEN_WRITE_ROOT = Path("/mnt/nvme")


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _validate_write_paths(cfg: BacktradeConfig) -> None:
    write_paths = {
        "project_root": cfg.paths.project_root,
        "output_root": cfg.paths.output_root,
    }
    for name, path in write_paths.items():
        if _is_relative_to(path, FORBIDDEN_WRITE_ROOT):
            raise ValueError(f"{name} cannot point under /mnt/nvme: {path}")

    if not _is_relative_to(cfg.paths.project_root, PROJECT_WRITE_ROOT):
        raise ValueError(f"project_root must stay under {PROJECT_WRITE_ROOT}: {cfg.paths.project_root}")
    if not _is_relative_to(cfg.paths.output_root, OUTPUT_WRITE_ROOT):
        raise ValueError(f"output_root must stay under {OUTPUT_WRITE_ROOT}: {cfg.paths.output_root}")


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path, profile_path: str | Path | None = None) -> BacktradeConfig:
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if profile_path is not None:
        with Path(profile_path).expanduser().open("r", encoding="utf-8") as fh:
            profile = yaml.safe_load(fh) or {}
        data = _deep_merge(data, profile)
    data = _expand_env(data)
    contract_files = data.get("contract_files", [])
    resolved_contract_files: list[str] = []
    merged_contracts: dict[str, Any] = {}
    for item in contract_files:
        contract_path = Path(item)
        if not contract_path.is_absolute():
            contract_path = cfg_path.parent / contract_path
        contract_path = contract_path.expanduser().resolve()
        with contract_path.open("r", encoding="utf-8") as fh:
            contract_data = yaml.safe_load(fh) or {}
        resolved_contract_files.append(str(contract_path))
        merged_contracts.update(_expand_env(contract_data).get("contracts", {}))
    if resolved_contract_files:
        data["contract_files"] = resolved_contract_files
    merged_contracts.update(data.get("contracts", {}))
    if merged_contracts:
        data["contracts"] = merged_contracts
    cfg = BacktradeConfig.model_validate(data)
    _validate_write_paths(cfg)
    return cfg



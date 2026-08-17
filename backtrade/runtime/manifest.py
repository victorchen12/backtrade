from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def payload_digest(payload: Any) -> str:
    encoded = json.dumps(_jsonable(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def make_run_id(config_digest: str, *, label: str = "run", timestamp: datetime | None = None) -> str:
    instant = (timestamp or datetime.now(timezone.utc)).astimezone(timezone.utc)
    safe_label = "-".join(part for part in label.strip().split() if part) or "run"
    return f"{safe_label}-{instant:%Y%m%dT%H%M%SZ}-{config_digest[:12]}"


def file_identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=False)
    if not resolved.is_file():
        return {"path": str(path), "resolved_path": str(resolved), "exists": False}
    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    stat = resolved.stat()
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "exists": True,
        "kind": "file",
        "size_bytes": stat.st_size,
        "sha256": digest,
    }


def build_run_manifest(cfg, *, run_id: str, output_root: str | Path, input_manifest: dict[str, Any] | None = None, artifact_schema_version: str = "compact_v9") -> dict[str, Any]:
    config = _jsonable(cfg.model_dump(mode="json"))
    return {
        "artifact_schema_version": artifact_schema_version,
        "run_id": run_id,
        "output_root": str(Path(output_root).expanduser().resolve()),
        "config": config,
        "config_digest": payload_digest(config),
        "input_manifest": _jsonable(input_manifest or {}),
        "match_mode": cfg.match.mode,
        "factor_semantics_version": "ofi_sign_v1",
    }


def write_run_manifest(output_root: str | Path, manifest: dict[str, Any]) -> Path:
    root = Path(output_root)
    path = root / "run_manifest.json"
    path.write_text(json.dumps(_jsonable(manifest), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


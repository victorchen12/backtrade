from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from backtrade.config.schema import BacktradeConfig
from backtrade.data.future_l2 import iter_future_l2_ticks
from backtrade.simulation.compact_v9 import audit_compact_v9
from backtrade.simulation.compact_v9_runner import CompactV9Runner
from backtrade.runtime.manifest import payload_digest
from backtrade.strategies.factors import SUPPORTED_FACTOR_NAMES


def _allowed_reset_root(path: Path) -> bool:
    resolved = path.expanduser().resolve(strict=False)
    # [README-6] reset 只允许清理批准根；普通运行用新的 output_root，不会覆盖已有产物。
    return any(
        resolved == root or root in resolved.parents
        for root in (Path.cwd().resolve(), Path("/tmp").resolve())
    )


def run_from_config(
    cfg: BacktradeConfig,
    output_root: str | Path | None = None,
    reset_output: bool = False,
    max_events: int | None = None,
    run_id: str | None = None,
    input_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if cfg.data.source != "future_l2":
        raise ValueError("compact_v9 requires data.source=future_l2")
    if cfg.strategy.factor_name not in SUPPORTED_FACTOR_NAMES or cfg.strategy.factor_column != cfg.strategy.factor_name:
        raise ValueError(f"compact_v9 does not support factor {cfg.strategy.factor_name}")
    if max_events is not None and int(max_events) <= 0:
        raise ValueError("max_events must be positive when provided")
    cfg.require_contract_for_real_run()
    # [README-6] 输出目录只接受本次新目录；reset 仅允许清理批准根下的旧目录。
    out = Path(output_root) if output_root is not None else cfg.paths.output_root
    if reset_output and out.exists():
        if not _allowed_reset_root(out):
            raise ValueError(f"refusing to reset output outside approved roots: {out}")
        shutil.rmtree(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"compact_v9 output root is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    limit = int(max_events) if max_events is not None else cfg.data.max_ticks
    ticks = iter_future_l2_ticks(cfg, max_events=limit)
    runner = CompactV9Runner(cfg, ticks)
    result = runner.run(max_events=max_events)
    manifest = runner.write(out)
    audit = audit_compact_v9(out, require_final_flat=True)
    summary = {
        "artifact_schema_version": "compact_v9",
        "match_mode": cfg.match.mode,
        "orders": len(result.orders),
        "fills": len(result.fills),
        "account_events": len(result.account_rows),
        "maker_events": len(result.maker_events),
        "final_snapshot": result.final_snapshot,
        "manifest": manifest,
        "audit": audit,
        "config_digest": payload_digest(cfg.model_dump(mode="json")),
        "input_manifest_ignored": input_manifest is not None,
    }
    return summary


__all__ = ["run_from_config"]


from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from backtrade.reporting.data import load_report_data
from backtrade.reporting.html import render_report_html
from backtrade.reporting.metrics import compute_report_metrics


def _safe_name(value: str) -> str:
    name = str(value).strip()
    if not name or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError(f"invalid report name: {value!r}")
    return name


def generate_backtest_report(
    run_root: str | Path,
    result_root: str | Path,
    *,
    factor_name: str | None = None,
    mode: str | None = None,
) -> dict:
    run_path = Path(run_root).expanduser().resolve()
    report_base = Path(result_root).expanduser().resolve()
    try:
        report_base.relative_to(run_path)
    except ValueError:
        pass
    else:
        raise ValueError(f"report root must be separate from run root: {report_base}")
    data = load_report_data(run_path)
    manifest = data["manifest"]
    factor = _safe_name(factor_name or manifest.get("factor_name") or "factor")
    match_mode = _safe_name(mode or manifest.get("match_mode") or "result")
    root = report_base / factor / match_mode
    # [README-6] 报告写入 result_view_root/因子名/撮合模式；目录非空时拒绝覆盖。
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"report directory is not empty and cannot be overwritten: {root}")
    root.mkdir(parents=True, exist_ok=True)
    strategy_config = ((manifest.get("config") or {}).get("strategy") or {})
    metrics = compute_report_metrics(
        data["accounts"],
        data["snapshots"],
        data["activity"],
        initial_cash=float((manifest.get("config") or {}).get("initial_cash", 0.0)),
        signal_mode=str(strategy_config.get("signal_mode") or "signed_factor"),
        short_threshold=strategy_config.get("short_threshold"),
        long_threshold=strategy_config.get("long_threshold"),
        snapshot_stats=data.get("snapshot_stats"),
    )
    html = render_report_html(data, metrics)
    html_path = root / "report.html"
    metrics_path = root / "metrics.json"
    manifest_path = root / "manifest.json"
    html_path.write_text(html, encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")
    report_manifest = {
        "report_schema_version": "compact_v9_report_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_path),
        "factor_name": factor,
        "match_mode": match_mode,
        "config_digest": manifest.get("config_digest"),
        "input_identities": manifest.get("input_identities", {}),
        "source_provenance": manifest.get("source_provenance", {}),
        "files": {
            "report.html": {"sha256": hashlib.sha256(html_path.read_bytes()).hexdigest(), "size_bytes": html_path.stat().st_size},
            "metrics.json": {"sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(), "size_bytes": metrics_path.stat().st_size},
        },
    }
    manifest_path.write_text(json.dumps(report_manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return {"root": str(root), "report": str(html_path), "metrics": str(metrics_path), "manifest": str(manifest_path), "factor_name": factor, "match_mode": match_mode, "metrics_summary": metrics}


__all__ = ["generate_backtest_report"]

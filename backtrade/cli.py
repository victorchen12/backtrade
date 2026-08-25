from __future__ import annotations

import argparse
import json
import subprocess
import sys
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtrade.config.loader import load_config
from backtrade.config.loader import validate_result_view_root
from backtrade.data.future_l2 import MARKET_COLUMNS
from backtrade.data.tabular import read_table, resolve_table_candidate, table_columns, table_row_count, write_table
from backtrade.reporting import generate_backtest_report
from backtrade.run import run_from_config
from backtrade.runtime.manifest import make_run_id, payload_digest
from backtrade.runtime.validation import validate_config
from backtrade.strategies.factors import (
    L1_IMBALANCE_NAME,
    compute_l1_imbalance,
    validate_factor_name,
)
from backtrade.simulation.compact_v9 import audit_compact_v9, read_compact_v9


def build_parser() -> argparse.ArgumentParser:
    # [README-7] CLI 顺序为 prepare-input、validate、run、inspect；路径参数见 README 第 1、2、6 节。
    parser = argparse.ArgumentParser(prog="backtrade", description="期货 L2 快照与时间序列因子的确定性回测模拟器")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "run"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--profile")
        command.add_argument("--input-root", help="输入目录；包含 market 和 YAML 中配置的因子文件（支持 Parquet、CSV/CSV.GZ、Feather）")
        command.add_argument("--market-path", help="自定义市场表格文件；覆盖 input-root 下的 market 文件")
        command.add_argument("--factor-path", help="自定义因子表格文件；覆盖 input-root 下的因子文件")
        command.add_argument("--trading-days", nargs="+", help="覆盖 YAML 中的交易日列表")
        command.add_argument("--result-view-root", help="自定义报告根目录；非空目录拒绝覆盖")
    run = commands.choices["run"]
    run.add_argument("--output-root", help="本次运行产物目录；覆盖 YAML 的 paths.output_root")
    run.add_argument("--label", help="运行标识，用于生成默认目录名")
    run.add_argument("--max-events", type=int, help="最多回放的事件数；设置后 EOF 为 end_of_data")
    prepare = commands.add_parser("prepare-input")
    prepare.add_argument("--root", help="默认输入目录；为空时由两个显式文件路径推导 manifest 目录")
    prepare.add_argument("--market-path", help="自定义市场表格文件路径")
    prepare.add_argument("--factor-path", help="自定义因子表格文件路径；manifest 写在同目录")
    prepare.add_argument("--product", required=True, help="商品代码")
    prepare.add_argument("--factor-column", default=L1_IMBALANCE_NAME, help="因子表格中的列名；必须与 YAML strategy.factor_column 一致")
    prepare.add_argument("--factor-columns", nargs="+", help="宽表中的全部因子列；启用 factor_bundle_v1")
    prepare.add_argument("--manifest-path", help="manifest 输出路径；为空时写入因子文件相邻 manifest.json")
    prepare.add_argument("--source-manifest-path", help="bundle 对应的来源 rolling manifest")
    derive = commands.add_parser("derive-factor")
    derive.add_argument("--market-path", required=True)
    derive.add_argument("--factor-path", required=True)
    derive.add_argument("--product", required=True)
    derive.add_argument("--factor-column", choices=[L1_IMBALANCE_NAME], default=L1_IMBALANCE_NAME)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--run", required=True)
    check = commands.add_parser("check")
    check.add_argument("--pytest-args", nargs="*", default=[])
    return parser


def _print(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str))


def _load_and_validate(
    config_path: str,
    profile_path: str | None,
    input_root: str | None = None,
    trading_days: list[str] | None = None,
    result_view_root: str | None = None,
    market_path: str | None = None,
    factor_path: str | None = None,
):
    cfg = load_config(config_path, profile_path=profile_path)
    data_updates: dict[str, Path] = {}
    if input_root:
        root = Path(input_root).expanduser().resolve()
        data_updates.update({"market_path": resolve_table_candidate(root, "market"), "factor_path": resolve_table_candidate(root, cfg.strategy.factor_column)})
    # [README-2] 显式文件路径可覆盖 input-root 的约定文件名，适合已有数据目录。
    if market_path:
        data_updates["market_path"] = Path(market_path).expanduser().resolve()
    if factor_path:
        data_updates["factor_path"] = Path(factor_path).expanduser().resolve()
    if data_updates:
        cfg = cfg.model_copy(update={"data": cfg.data.model_copy(update=data_updates)})
    if trading_days is not None:
        cfg = cfg.model_copy(update={"data": cfg.data.model_copy(update={"trading_days": list(trading_days)})})
    if result_view_root:
        root = validate_result_view_root(result_view_root)
        cfg = cfg.model_copy(update={"paths": cfg.paths.model_copy(update={"result_view_root": root})})
    return cfg, validate_config(cfg)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_input(
    root_arg: str | None,
    product: str,
    market_path_arg: str | None = None,
    factor_path_arg: str | None = None,
    *,
    factor_column: str = L1_IMBALANCE_NAME,
    factor_columns: list[str] | None = None,
    manifest_path_arg: str | None = None,
    source_manifest_path_arg: str | None = None,
) -> dict[str, Any]:
    # [README-1] 输入登记只检查身份和表结构，不改写原始表格文件。
    bundle_mode = factor_columns is not None
    selected_columns = [validate_factor_name(value) for value in factor_columns] if bundle_mode else [validate_factor_name(factor_column)]
    if not selected_columns or len(selected_columns) != len(set(selected_columns)):
        raise ValueError("factor columns must be non-empty and unique")
    if source_manifest_path_arg is not None and not bundle_mode:
        raise ValueError("--source-manifest-path requires --factor-columns")
    root = Path(root_arg).expanduser().resolve() if root_arg else None
    market_path = Path(market_path_arg).expanduser().resolve() if market_path_arg else resolve_table_candidate(root, "market") if root else None
    factor_path = Path(factor_path_arg).expanduser().resolve() if factor_path_arg else resolve_table_candidate(root, selected_columns[0]) if root and not bundle_mode else None
    if market_path is None or factor_path is None:
        raise ValueError("prepare-input requires --root or both explicit file paths")
    if not market_path.is_file() or not factor_path.is_file():
        raise FileNotFoundError(f"input file missing: market={market_path}, factor={factor_path}")
    market_names = set(table_columns(market_path))
    missing_market = sorted(set(MARKET_COLUMNS) - market_names)
    if missing_market:
        raise ValueError(f"market input is missing required columns: {missing_market}")
    factor_names = set(table_columns(factor_path))
    required_factor = {"tick_ts", *selected_columns}
    if required_factor - factor_names:
        raise ValueError(f"factor input is missing required columns: {sorted(required_factor - factor_names)}")
    factor_read_columns = ["tick_ts", *selected_columns]
    factor_context = ["product", "trading_day", "session_id", "underlying_secu_cd"]
    present_context = set(factor_context) & factor_names
    if present_context and present_context != set(factor_context):
        raise ValueError("factor input must contain complete context columns or only tick_ts")
    if bundle_mode and present_context != set(factor_context):
        raise ValueError("factor bundle input requires complete context columns")
    if present_context == set(factor_context):
        factor_read_columns = [*factor_context, *factor_read_columns]
    factor_read_columns = [column for column in ["split_id", "part", *factor_read_columns] if column in factor_names]
    factors = read_table(factor_path, columns=factor_read_columns)
    factors["tick_ts"] = pd.to_datetime(factors["tick_ts"])
    key_columns = [column for column in ["split_id", "part", *factor_context, "tick_ts"] if column in factors.columns]
    if factors["tick_ts"].isna().any() or factors.duplicated(key_columns).any():
        raise ValueError("factor input must have unique, non-null factor keys")
    if bundle_mode and (factors["product"].astype(str).str.lower() != str(product).lower()).any():
        raise ValueError("factor bundle product does not match configured product")
    for column in selected_columns:
        values = pd.to_numeric(factors[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{column} must contain finite values")
    factor_hash = _sha256(factor_path)
    source_manifest_path: Path | None = None
    source_manifest_hash: str | None = None
    source_schema_version: str | None = None
    if bundle_mode:
        if source_manifest_path_arg is None:
            raise ValueError("factor bundle requires --source-manifest-path")
        source_manifest_path = Path(source_manifest_path_arg).expanduser().resolve()
        if not source_manifest_path.is_file():
            raise FileNotFoundError(f"source manifest is missing: {source_manifest_path}")
        try:
            source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"source manifest cannot be read: {source_manifest_path}") from exc
        if source_manifest.get("calibrated_factor_file_sha256") != factor_hash:
            raise ValueError("source manifest factor hash does not match factor input")
        if source_manifest.get("factor_columns") != selected_columns:
            raise ValueError("source manifest factor columns do not match requested bundle")
        calibrated_columns = source_manifest.get("calibrated_factor_columns")
        if not isinstance(calibrated_columns, list) or not set(selected_columns).issubset(calibrated_columns):
            raise ValueError("source manifest calibrated columns do not cover requested bundle")
        source_manifest_hash = _sha256(source_manifest_path)
        source_schema_version = source_manifest.get("schema_version")
    manifest = {
        "schema_version": "factor_bundle_v1" if bundle_mode else f"{selected_columns[0]}_minimal_v1",
        "product": str(product).lower(),
        "market_path": str(market_path),
        "market_sha256": _sha256(market_path),
        "factor_values_path": str(factor_path),
        "factor_values_sha256": factor_hash,
        "factor_columns": selected_columns,
        "input_columns": {"market_required": MARKET_COLUMNS, "factor_required": ["tick_ts", *selected_columns]},
    }
    if bundle_mode:
        manifest.update(
            {
                "source_manifest_path": str(source_manifest_path),
                "source_manifest_sha256": source_manifest_hash,
                "source_manifest_schema_version": source_schema_version,
            }
        )
    # [README-1] manifest 与因子输入同目录，并绑定两份输入哈希。
    manifest_path = Path(manifest_path_arg).expanduser().resolve() if manifest_path_arg else factor_path.with_name("manifest.json")
    if manifest_path.exists():
        raise FileExistsError(f"input manifest already exists: {manifest_path}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "root": str(root or factor_path.parent),
        "manifest": str(manifest_path),
        "market": str(market_path),
        "factor": str(factor_path),
        "factor_column": selected_columns[0],
        "factor_columns": selected_columns,
        "market_rows": table_row_count(market_path),
        "factor_rows": int(len(factors)),
    }


def _derive_factor(
    market_path_arg: str,
    factor_path_arg: str,
    product: str,
    factor_column: str,
) -> dict[str, Any]:
    # [README-1] 只用当前买一和卖一数量生成文档中的 L1 指标。
    if factor_column != L1_IMBALANCE_NAME:
        raise ValueError(f"only {L1_IMBALANCE_NAME} can be derived from L2 snapshots")
    market_path = Path(market_path_arg).expanduser().resolve()
    factor_path = Path(factor_path_arg).expanduser().resolve()
    if not market_path.is_file():
        raise FileNotFoundError(f"market input is missing: {market_path}")
    if factor_path.exists():
        raise FileExistsError(f"factor input already exists: {factor_path}")
    names = set(table_columns(market_path))
    missing = sorted(set(MARKET_COLUMNS) - names)
    if missing:
        raise ValueError(f"market input is missing required columns: {missing}")
    columns = [*MARKET_COLUMNS, "product"]
    frame = read_table(market_path, columns=[column for column in columns if column in names])
    if "product" not in frame:
        frame["product"] = str(product).lower()
    frame["product"] = frame["product"].fillna(product).astype(str).str.lower()
    if (frame["product"] != str(product).lower()).any():
        raise ValueError("market product does not match configured product")
    values = [
        compute_l1_imbalance(bid, ask)
        for bid, ask in zip(frame["bid1_qty"], frame["ask1_qty"], strict=True)
    ]
    factor = frame[["product", "trading_day", "session_id", "tick_ts", "underlying_secu_cd"]].copy()
    factor[factor_column] = values
    write_table(factor, factor_path)
    return {"market": str(market_path), "factor": str(factor_path), "factor_column": factor_column, "rows": int(len(factor))}


def _new_output_root(cfg, explicit: str | None, label: str | None) -> tuple[Path, str]:
    digest = payload_digest(cfg.model_dump(mode="json"))
    run_id = make_run_id(digest, label=label or cfg.data.product)
    root = Path(explicit).expanduser().resolve() if explicit else cfg.paths.output_root / "runs" / run_id
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"output root is not empty and cannot be overwritten: {root}")
    return root, run_id


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare-input":
        try:
            _print(
                _prepare_input(
                    args.root,
                    args.product,
                    args.market_path,
                    args.factor_path,
                    factor_column=args.factor_column,
                    factor_columns=args.factor_columns,
                    manifest_path_arg=args.manifest_path,
                    source_manifest_path_arg=args.source_manifest_path,
                )
            )
            return 0
        except Exception as exc:
            _print({"passed": False, "error": str(exc)})
            return 2
    if args.command == "derive-factor":
        try:
            _print(_derive_factor(args.market_path, args.factor_path, args.product, args.factor_column))
            return 0
        except Exception as exc:
            _print({"passed": False, "error": str(exc)})
            return 2
    if args.command == "validate":
        _, report = _load_and_validate(args.config, args.profile, args.input_root, args.trading_days, args.result_view_root, args.market_path, args.factor_path)
        _print(report)
        return 0 if report["passed"] else 2
    if args.command == "run":
        cfg, report = _load_and_validate(args.config, args.profile, args.input_root, args.trading_days, args.result_view_root, args.market_path, args.factor_path)
        if not report["passed"]:
            _print(report)
            return 2
        output_root, run_id = _new_output_root(cfg, args.output_root, args.label)
        summary = run_from_config(cfg, output_root=output_root, max_events=args.max_events, run_id=run_id, input_manifest={"validation": report})
        try:
            summary["report"] = generate_backtest_report(output_root, cfg.paths.result_view_root, factor_name=cfg.strategy.factor_name, mode=cfg.match.mode)
        except Exception as exc:
            summary["report"] = {"passed": False, "error": str(exc)}
            summary["audit"]["passed"] = False
        _print(summary)
        return 0 if summary.get("audit", {}).get("passed", False) else 3
    if args.command == "inspect":
        root = Path(args.run).expanduser().resolve()
        try:
            manifest = read_compact_v9(root)
            audit = audit_compact_v9(root, require_final_flat=True)
            _print({"run": str(root), "manifest": manifest, "audit": audit})
            return 0 if audit["passed"] else 3
        except Exception as exc:
            _print({"run": str(root), "passed": False, "error": str(exc)})
            return 2
    return subprocess.run([sys.executable, "-m", "pytest", "-q", *args.pytest_args], check=False).returncode


__all__ = ["build_parser", "main"]

if __name__ == "__main__":
    raise SystemExit(main())


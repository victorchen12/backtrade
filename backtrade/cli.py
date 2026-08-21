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
import pyarrow.parquet as pq

from backtrade.config.loader import load_config
from backtrade.config.loader import validate_result_view_root
from backtrade.data.future_l2 import MARKET_COLUMNS
from backtrade.reporting import generate_backtest_report
from backtrade.run import run_from_config
from backtrade.runtime.manifest import make_run_id, payload_digest
from backtrade.runtime.validation import validate_config
from backtrade.strategies.factors import (
    L1_IMBALANCE_NAME,
    SUPPORTED_FACTOR_NAMES,
    compute_l1_imbalance,
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
        command.add_argument("--input-root", help="输入目录；包含 market.parquet 和 l1_imbalance.parquet")
        command.add_argument("--market-path", help="自定义市场 parquet；覆盖 input-root/market.parquet")
        command.add_argument("--factor-path", help="自定义因子 parquet；覆盖 input-root/l1_imbalance.parquet")
        command.add_argument("--trading-days", nargs="+", help="覆盖 YAML 中的交易日列表")
        command.add_argument("--result-view-root", help="自定义报告根目录；非空目录拒绝覆盖")
    run = commands.choices["run"]
    run.add_argument("--output-root", help="本次运行产物目录；覆盖 YAML 的 paths.output_root")
    run.add_argument("--label", help="运行标识，用于生成默认目录名")
    run.add_argument("--max-events", type=int, help="最多回放的事件数；设置后 EOF 为 end_of_data")
    prepare = commands.add_parser("prepare-input")
    prepare.add_argument("--root", help="默认输入目录；为空时由两个显式文件路径推导 manifest 目录")
    prepare.add_argument("--market-path", help="自定义市场 parquet 路径")
    prepare.add_argument("--factor-path", help="自定义因子 parquet 路径；manifest 写在同目录")
    prepare.add_argument("--product", required=True, help="商品代码")
    prepare.add_argument("--factor-column", choices=sorted(SUPPORTED_FACTOR_NAMES), default=L1_IMBALANCE_NAME)
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
        data_updates.update({"market_path": root / "market.parquet", "factor_path": root / "l1_imbalance.parquet"})
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
) -> dict[str, Any]:
    # [README-1] 输入登记只检查身份和表结构，不改写原始 parquet。
    if factor_column not in SUPPORTED_FACTOR_NAMES:
        raise ValueError(f"unsupported factor column: {factor_column}")
    root = Path(root_arg).expanduser().resolve() if root_arg else None
    market_path = Path(market_path_arg).expanduser().resolve() if market_path_arg else root / "market.parquet" if root else None
    factor_path = Path(factor_path_arg).expanduser().resolve() if factor_path_arg else root / "l1_imbalance.parquet" if root else None
    if market_path is None or factor_path is None:
        raise ValueError("prepare-input requires --root or both explicit file paths")
    if not market_path.is_file() or not factor_path.is_file():
        raise FileNotFoundError(f"input file missing: market={market_path}, factor={factor_path}")
    market_names = set(pq.ParquetFile(market_path).schema.names)
    missing_market = sorted(set(MARKET_COLUMNS) - market_names)
    if missing_market:
        raise ValueError(f"market parquet is missing required columns: {missing_market}")
    factor_names = set(pq.ParquetFile(factor_path).schema.names)
    required_factor = {"tick_ts", factor_column}
    if required_factor - factor_names:
        raise ValueError(f"factor parquet requires tick_ts and {factor_column}")
    factor_read_columns = ["tick_ts", factor_column]
    factor_context = ["product", "trading_day", "session_id", "underlying_secu_cd"]
    if set(factor_context).issubset(factor_names):
        factor_read_columns = [*factor_context, *factor_read_columns]
    factors = pd.read_parquet(factor_path, columns=factor_read_columns)
    factors["tick_ts"] = pd.to_datetime(factors["tick_ts"])
    key_columns = [*factor_context, "tick_ts"] if set(factor_context).issubset(factors.columns) else ["tick_ts"]
    if factors["tick_ts"].isna().any() or factors.duplicated(key_columns).any():
        raise ValueError("factor parquet must have unique, non-null factor keys")
    if not factors[factor_column].map(lambda value: pd.notna(value) and np.isfinite(float(value))).all():
        raise ValueError(f"{factor_column} must contain finite values")
    manifest = {
        "schema_version": f"{factor_column}_minimal_v1",
        "product": str(product).lower(),
        "market_path": str(market_path),
        "market_sha256": _sha256(market_path),
        "factor_values_path": str(factor_path),
        "factor_values_sha256": _sha256(factor_path),
        "factor_columns": [factor_column],
        "input_columns": {"market_required": MARKET_COLUMNS, "factor_required": ["tick_ts", factor_column]},
    }
    # [README-1] manifest 与因子 parquet 同目录，并绑定两份输入哈希。
    manifest_path = factor_path.with_name("manifest.json")
    if manifest_path.exists():
        raise FileExistsError(f"input manifest already exists: {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "root": str(root or factor_path.parent),
        "manifest": str(manifest_path),
        "market": str(market_path),
        "factor": str(factor_path),
        "factor_column": factor_column,
        "market_rows": int(pq.ParquetFile(market_path).metadata.num_rows),
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
        raise FileNotFoundError(f"market parquet is missing: {market_path}")
    if factor_path.exists():
        raise FileExistsError(f"factor parquet already exists: {factor_path}")
    names = set(pq.ParquetFile(market_path).schema.names)
    missing = sorted(set(MARKET_COLUMNS) - names)
    if missing:
        raise ValueError(f"market parquet is missing required columns: {missing}")
    columns = [*MARKET_COLUMNS, "product"]
    frame = pd.read_parquet(market_path, columns=[column for column in columns if column in names])
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
    factor.to_parquet(factor_path, index=False)
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
            _print(_prepare_input(args.root, args.product, args.market_path, args.factor_path, factor_column=args.factor_column))
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


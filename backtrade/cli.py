from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from backtrade.config.loader import load_config
from backtrade.run import run_from_config
from backtrade.runtime.manifest import make_run_id, payload_digest
from backtrade.runtime.validation import validate_config
from backtrade.simulation.compact_v9 import audit_compact_v9, read_compact_v9


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="backtrade", description="Deterministic futures L2 backtest simulator")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "run"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--profile")
    run = commands.choices["run"]
    run.add_argument("--output-root")
    run.add_argument("--label")
    run.add_argument("--max-events", type=int)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--run", required=True)
    check = commands.add_parser("check")
    check.add_argument("--pytest-args", nargs="*", default=[])
    return parser


def _print(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str))


def _load_and_validate(config_path: str, profile_path: str | None):
    cfg = load_config(config_path, profile_path=profile_path)
    return cfg, validate_config(cfg)


def _new_output_root(cfg, explicit: str | None, label: str | None) -> tuple[Path, str]:
    digest = payload_digest(cfg.model_dump(mode="json"))
    run_id = make_run_id(digest, label=label or cfg.data.product)
    root = Path(explicit).expanduser().resolve() if explicit else cfg.paths.output_root / "runs" / run_id
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"output root is not empty and cannot be overwritten: {root}")
    return root, run_id


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        _, report = _load_and_validate(args.config, args.profile)
        _print(report)
        return 0 if report["passed"] else 2
    if args.command == "run":
        cfg, report = _load_and_validate(args.config, args.profile)
        if not report["passed"]:
            _print(report)
            return 2
        output_root, run_id = _new_output_root(cfg, args.output_root, args.label)
        summary = run_from_config(cfg, output_root=output_root, max_events=args.max_events, run_id=run_id, input_manifest={"validation": report})
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


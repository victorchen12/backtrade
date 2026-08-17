#!/usr/bin/env python3
"""Build a validated prev_day_vwap_proxy reference snapshot for a compact_v9 run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import pandas as pd

from backtrade.config.loader import load_config
from backtrade.data.limit_reference import (
    _rule_version,
    load_prev_day_vwap_limit_references,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--factor-path", type=Path, required=True)
    parser.add_argument("--market-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if not args.factor_path.is_file() or not args.market_path.is_file():
        raise FileNotFoundError("factor and market paths must exist")
    factor_keys = pd.read_parquet(
        args.factor_path,
        columns=["trading_day", "underlying_secu_cd"],
    )
    targets = (
        factor_keys.assign(
            trading_day=factor_keys["trading_day"].astype(str),
            contract=factor_keys["underlying_secu_cd"].astype(str).str.upper(),
        )[["trading_day", "contract"]]
        .drop_duplicates()
        .sort_values(["trading_day", "contract"], kind="mergesort")
        .reset_index(drop=True)
    )
    references = load_prev_day_vwap_limit_references(
        args.market_path,
        cfg.contracts,
        targets["trading_day"].tolist(),
        cfg.limit_reference.shfe_new_rule_effective_date,
    )

    rows = []
    for row in targets.itertuples(index=False):
        key = (str(row.trading_day), str(row.contract))
        reference = references.get(key)
        if reference is None:
            rows.append(
                {
                    "trading_day": key[0],
                    "contract": key[1],
                    "reference_price": None,
                    "limit_rate": None,
                    "limit_up": None,
                    "limit_down": None,
                    "source": "missing",
                    "rule_version": _rule_version(key[0], cfg.limit_reference.shfe_new_rule_effective_date),
                }
            )
        else:
            rows.append(
                {
                    "trading_day": reference.trading_day,
                    "contract": reference.contract,
                    "reference_price": reference.reference_price,
                    "limit_rate": reference.limit_rate,
                    "limit_up": reference.limit_up,
                    "limit_down": reference.limit_down,
                    "source": reference.source,
                    "rule_version": reference.rule_version,
                }
            )
    snapshot = pd.DataFrame(rows)
    counts = snapshot["source"].value_counts().to_dict()
    manifest = {
        "schema_version": "backtrade_price_limit_reference_snapshot_v1",
        "source": "prev_day_vwap_proxy",
        "approximate_limit_reference": True,
        "target_key_count": int(len(snapshot)),
        "source_counts": {str(key): int(value) for key, value in counts.items()},
        "factor_path": str(args.factor_path),
        "market_path": str(args.market_path),
        "factor_sha256": _sha256(args.factor_path),
        "market_sha256": _sha256(args.market_path),
        "rule_version_effective_date": str(cfg.limit_reference.shfe_new_rule_effective_date),
        "columns": list(snapshot.columns),
    }

    output_root = args.output_root
    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"refusing to overwrite non-empty output: {output_root}")
        shutil.rmtree(output_root)
    staging = output_root.parent / f".{output_root.name}.staging.{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        snapshot.to_parquet(staging / "price_limit_references.parquet", index=False, compression="zstd")
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        output_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, output_root)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

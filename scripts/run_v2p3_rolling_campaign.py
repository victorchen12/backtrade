from __future__ import annotations

import argparse
import json
import multiprocessing
import queue
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""} and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from backtrade.config.loader import load_config  # noqa: E402
from backtrade.config.schema import BacktradeConfig, StrategyConfig  # noqa: E402
from backtrade.reporting import generate_backtest_report  # noqa: E402
from backtrade.run import run_from_config  # noqa: E402
from backtrade.runtime.validation import validate_config  # noqa: E402


BASE_CONFIG_PATH = PROJECT_ROOT / "configs" / "l1_imbalance_single_day_taker.yaml"
MARKET_PATH = Path("/data1/cws/future_l2/v2p3_dataset_24_26/raw_input/ag_con_tick.parquet")
FACTOR_PATH = Path(
    "/data1/cws/future_l2/v2p3_dataset_24_26_rolling_7f_v2/factors/"
    "ag_v2p3_rolling_24_26_factor_values_calibrated_5s_keyed.parquet"
)
FACTOR_MANIFEST_PATH = Path(
    "/data1/cws/backtrade/v2p3_rolling_24_26/input/factor_bundle_manifest.json"
)
SAMPLE_OUTPUT_BASE = Path("/data1/cws/backtrade/v2p3_rolling_24_26/sample_split_015")
SAMPLE_RESULT_BASE = Path("/home/cws/QUANT/Backtrade/result_view/_sample_split_015")
FULL_OUTPUT_BASE = Path("/data1/cws/backtrade/v2p3_rolling_24_26/full")
FULL_RESULT_BASE = Path("/home/cws/QUANT/Backtrade/result_view")

FACTOR_COLUMNS = (
    "v2p3_m2_lgbm_last_ret_log_mdd_5s",
    "v2p3_m2_lgbm_last_trend_cascade_focal_5s",
    "v2p3_m2_lgbm_opp_cover_diff_focal_5s",
    "v2p3_m1_lgbm_opp_long_short_q75_diff_5s",
    "v2p3_m2_lgbm_last_path_mean_huber_5s",
    "v2p3_m1_lgbm_last_full_cascade_wbce_5s",
)
MATCH_MODES = ("maker", "taker")
EXPECTED_TEST_SPLITS = frozenset(f"{index:03d}" for index in range(32))


@dataclass(frozen=True)
class CampaignJob:
    factor: str
    mode: str
    config: BacktradeConfig
    output_root: Path
    report_root: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the fixed v2p3 rolling six-factor campaign")
    parser.add_argument("--scope", choices=["sample", "full"], required=True)
    return parser


def _validate_base_config(config: BacktradeConfig) -> None:
    rule = config.contract_rule("ag")
    fixed_values = {
        "initial_cash": (float(config.initial_cash), 1_000_000.0),
        "latency_ms": (int(config.execution.latency_ms), 5),
        "day_end_flatten_window_ms": (int(config.execution.day_end_flatten_window_ms), 5_000),
        "multiplier": (float(rule.multiplier), 15.0),
        "tick_size": (float(rule.tick_size), 1.0),
    }
    mismatches = [name for name, (actual, expected) in fixed_values.items() if actual != expected]
    if mismatches:
        raise ValueError(f"campaign base config changes fixed values: {mismatches}")
    if config.limit_reference.mode != "disabled":
        raise ValueError("campaign requires limit_reference.mode=disabled")


def build_jobs(
    base_config: BacktradeConfig,
    *,
    split_id: str | None,
    output_base: Path,
    result_base: Path,
    market_path: Path,
    factor_path: Path,
    factor_manifest_path: Path,
) -> list[CampaignJob]:
    _validate_base_config(base_config)
    output_base = Path(output_base).expanduser().resolve()
    result_base = Path(result_base).expanduser().resolve()
    market_path = Path(market_path).expanduser().resolve()
    factor_path = Path(factor_path).expanduser().resolve()
    factor_manifest_path = Path(factor_manifest_path).expanduser().resolve()
    jobs: list[CampaignJob] = []
    for factor in FACTOR_COLUMNS:
        for mode in MATCH_MODES:
            output_root = output_base / factor / mode
            report_root = result_base / factor / mode
            data = base_config.data.model_copy(
                update={
                    "product": "ag",
                    "split_id": split_id,
                    "max_ticks": None,
                    "market_path": market_path,
                    "factor_path": factor_path,
                    "factor_manifest_path": factor_manifest_path,
                    "parts": ["test"],
                    "trading_days": None,
                    "factor_grid_mode": "decision_grid",
                    "eof_is_day_end": True,
                }
            )
            strategy = StrategyConfig(
                factor_name=factor,
                factor_column=factor,
                signal_mode="ecdf_tail",
                short_threshold=-0.8,
                long_threshold=0.8,
            )
            paths = base_config.paths.model_copy(
                update={
                    "project_root": PROJECT_ROOT,
                    "output_root": output_root,
                    "result_view_root": result_base,
                }
            )
            config = base_config.model_copy(
                update={
                    "paths": paths,
                    "data": data,
                    "strategy": strategy,
                    "match": base_config.match.model_copy(update={"mode": mode}),
                }
            )
            jobs.append(CampaignJob(factor, mode, config, output_root, report_root))
    return jobs


def _nonempty(path: Path) -> bool:
    if not path.exists():
        return False
    if not path.is_dir():
        return True
    return next(path.iterdir(), None) is not None


def preflight_targets(jobs: list[CampaignJob]) -> None:
    outputs = [job.output_root for job in jobs]
    reports = [job.report_root for job in jobs]
    if len(outputs) != len(set(outputs)) or len(reports) != len(set(reports)):
        raise ValueError("campaign target directories are not unique")
    for job in jobs:
        try:
            job.report_root.resolve().relative_to(job.output_root.resolve())
        except ValueError:
            pass
        else:
            raise ValueError(f"report target is inside run target: {job.report_root}")
        for path in (job.output_root, job.report_root):
            if _nonempty(path):
                raise FileExistsError(f"campaign target is not empty and cannot be overwritten: {path}")


def validate_campaign_input_coverage(jobs: list[CampaignJob]) -> set[str]:
    if not jobs:
        raise ValueError("campaign has no jobs")
    first = jobs[0].config.data
    selection = (
        Path(first.market_path).resolve(),
        Path(first.factor_path).resolve(),
        first.split_id,
        tuple(first.parts),
    )
    for job in jobs[1:]:
        data = job.config.data
        candidate = (
            Path(data.market_path).resolve(),
            Path(data.factor_path).resolve(),
            data.split_id,
            tuple(data.parts),
        )
        if candidate != selection:
            raise ValueError("campaign jobs do not share one input selection")

    factor_path = selection[1]
    market_path = selection[0]
    factor_schema = pq.ParquetFile(factor_path).schema_arrow
    required_columns = {"part", "trading_day"}
    if first.split_id is not None:
        required_columns.add("split_id")
    missing_columns = sorted(required_columns - set(factor_schema.names))
    if missing_columns:
        raise ValueError(f"campaign factor input is missing columns: {missing_columns}")
    factor_filters: list[tuple[str, str, object]] = [("part", "in", list(first.parts))]
    if first.split_id is not None:
        split_type = factor_schema.field("split_id").type
        if pa.types.is_integer(split_type):
            split_value: object = int(first.split_id)
        elif pa.types.is_string(split_type) or pa.types.is_large_string(split_type):
            split_value = str(first.split_id).zfill(3)
        else:
            raise ValueError(f"campaign factor split_id type is unsupported: {split_type}")
        factor_filters.append(("split_id", "=", split_value))
    factor_days = pq.read_table(
        factor_path,
        columns=["trading_day"],
        filters=factor_filters,
    ).column("trading_day")
    if factor_days.null_count:
        raise ValueError("campaign factor selection contains null trading_day")
    expected = {str(value) for value in factor_days.unique().to_pylist()}
    if not expected:
        raise ValueError("campaign factor selection has no trading days")

    market_days = pq.read_table(
        market_path,
        columns=["trading_day"],
        filters=[("trading_day", "in", sorted(expected))],
    ).column("trading_day")
    actual = {str(value) for value in market_days.unique().to_pylist() if value is not None}
    missing = sorted(expected - actual)
    if missing:
        raise ValueError(f"campaign market coverage is missing trading days: {missing}")
    return expected


def validate_full_split_coverage(factor_path: Path) -> set[str]:
    factor_path = Path(factor_path).expanduser().resolve()
    if not factor_path.is_file():
        raise FileNotFoundError(f"campaign factor input is missing: {factor_path}")
    parquet = pq.ParquetFile(factor_path)
    required = {"split_id", "part"}
    missing_columns = sorted(required - set(parquet.schema.names))
    if missing_columns:
        raise ValueError(f"full campaign factor input is missing columns: {missing_columns}")
    split_values = pq.read_table(
        factor_path,
        columns=["split_id"],
        filters=[("part", "=", "test")],
    ).column("split_id")
    if split_values.null_count:
        raise ValueError("full campaign test rows contain null split_id")
    actual = {str(value).zfill(3) for value in split_values.unique().to_pylist()}
    missing = sorted(EXPECTED_TEST_SPLITS - actual)
    unexpected = sorted(actual - EXPECTED_TEST_SPLITS)
    if missing or unexpected:
        raise ValueError(
            "full campaign test split coverage is invalid: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return actual


def _run_job_child(job: CampaignJob, validation: dict, result_queue) -> None:
    try:
        summary = run_from_config(
            job.config,
            output_root=job.output_root,
            input_manifest={"validation": validation},
        )
        if not summary.get("audit", {}).get("passed"):
            raise RuntimeError(
                f"audit failed for {job.factor}/{job.mode}: "
                f"{summary.get('audit', {}).get('errors', [])}"
            )
        report = generate_backtest_report(
            job.output_root,
            job.config.paths.result_view_root,
            factor_name=job.factor,
            mode=job.mode,
        )
        result_queue.put(
            (
                "ok",
                {
                    "factor": job.factor,
                    "mode": job.mode,
                    "output_root": str(job.output_root),
                    "report_root": str(job.report_root),
                    "summary": summary,
                    "report": report,
                },
            )
        )
    except BaseException as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))


def _execute_job_isolated(job: CampaignJob, validation: dict) -> dict:
    methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("fork" if "fork" in methods else methods[0])
    result_queue = context.Queue()
    process = context.Process(target=_run_job_child, args=(job, validation, result_queue))
    process.start()
    process.join()
    try:
        status, payload = result_queue.get(timeout=5)
    except queue.Empty as exc:
        raise RuntimeError(
            f"isolated campaign job exited without a result: {job.factor}/{job.mode} "
            f"(exitcode={process.exitcode})"
        ) from exc
    finally:
        result_queue.close()
    if status != "ok":
        raise RuntimeError(payload)
    if process.exitcode != 0:
        raise RuntimeError(f"isolated campaign job exited with code {process.exitcode}: {job.factor}/{job.mode}")
    return payload


def execute_campaign(jobs: list[CampaignJob], *, isolate_jobs: bool = False) -> list[dict]:
    preflight_targets(jobs)
    validate_campaign_input_coverage(jobs)
    validations: list[dict] = []
    for job in jobs:
        validation = validate_config(job.config)
        if not validation.get("passed"):
            raise ValueError(
                f"validation failed for {job.factor}/{job.mode}: {validation.get('errors', [])}"
            )
        validations.append(validation)

    results: list[dict] = []
    for index, (job, validation) in enumerate(zip(jobs, validations, strict=True), start=1):
        print(json.dumps({"event": "start", "index": index, "total": len(jobs), "factor": job.factor, "mode": job.mode}), flush=True)
        if isolate_jobs:
            result = _execute_job_isolated(job, validation)
        else:
            summary = run_from_config(
                job.config,
                output_root=job.output_root,
                input_manifest={"validation": validation},
            )
            if not summary.get("audit", {}).get("passed"):
                raise RuntimeError(
                    f"audit failed for {job.factor}/{job.mode}: "
                    f"{summary.get('audit', {}).get('errors', [])}"
                )
            report = generate_backtest_report(
                job.output_root,
                job.config.paths.result_view_root,
                factor_name=job.factor,
                mode=job.mode,
            )
            result = {
                "factor": job.factor,
                "mode": job.mode,
                "output_root": str(job.output_root),
                "report_root": str(job.report_root),
                "summary": summary,
                "report": report,
            }
        results.append(result)
        print(json.dumps({"event": "complete", "index": index, "total": len(jobs), "factor": job.factor, "mode": job.mode}), flush=True)
    return results


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base_config = load_config(BASE_CONFIG_PATH)
    if args.scope == "sample":
        split_id = "015"
        output_base = SAMPLE_OUTPUT_BASE
        result_base = SAMPLE_RESULT_BASE
    else:
        validate_full_split_coverage(FACTOR_PATH)
        split_id = None
        output_base = FULL_OUTPUT_BASE
        result_base = FULL_RESULT_BASE
    jobs = build_jobs(
        base_config,
        split_id=split_id,
        output_base=output_base,
        result_base=result_base,
        market_path=MARKET_PATH,
        factor_path=FACTOR_PATH,
        factor_manifest_path=FACTOR_MANIFEST_PATH,
    )
    results = execute_campaign(jobs, isolate_jobs=True)
    print(json.dumps({"scope": args.scope, "completed": len(results), "results": results}, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

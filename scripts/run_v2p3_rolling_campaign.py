from __future__ import annotations

import argparse
import json
import multiprocessing
import queue
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""} and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from backtrade.config.loader import load_config
from backtrade.config.schema import BacktradeConfig, DataSourceConfig, StrategyConfig
from backtrade.reporting import generate_backtest_report
from backtrade.run import run_from_config
from backtrade.runtime.validation import validate_config

BASE_CONFIG_PATH = PROJECT_ROOT / "configs" / "l1_imbalance_single_day_taker.yaml"
MARKET_PATH = Path("/data1/cws/future_l2/v2p3_dataset_24_26/raw_input/ag_con_tick.parquet")
FACTOR_PATH = Path(
    "/data1/cws/future_l2/v2p3_dataset_24_26_rolling_7f_v2/factors/"
    "ag_v2p3_rolling_24_26_factor_values_calibrated_5s_keyed.parquet"
)
FACTOR_MANIFEST_PATH = Path(
    "/data1/cws/backtrade/v2p3_rolling_24_26/input/factor_bundle_manifest.json"
)
MAIN_CONTRACT_MAP_PATH = Path("/data1/cws/future_l2/m3/pre_data/main_contract_map.parquet")
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
    parser.add_argument(
        "--factor-workers",
        type=_factor_worker_count,
        default=1,
        help="number of factors to run concurrently (1-6; each factor runs maker then taker)",
    )
    return parser


def _factor_worker_count(value: str) -> int:
    try:
        workers = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("factor workers must be an integer from 1 to 6") from exc
    if not 1 <= workers <= len(FACTOR_COLUMNS):
        raise argparse.ArgumentTypeError("factor workers must be from 1 to 6")
    return workers


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
    split_ids: list[str] | None = None,
) -> list[CampaignJob]:
    _validate_base_config(base_config)
    if split_id is not None and split_ids is not None:
        raise ValueError("split_id and split_ids cannot both be configured")
    if split_ids is not None:
        split_ids = DataSourceConfig(
            product=base_config.data.product,
            split_ids=split_ids,
            max_ticks=1,
        ).split_ids
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
                    "split_ids": split_ids,
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


def group_jobs_by_factor(jobs: list[CampaignJob]) -> list[list[CampaignJob]]:
    groups: dict[str, list[CampaignJob]] = {}
    seen: set[tuple[str, str]] = set()
    for job in jobs:
        key = (job.factor, job.mode)
        if key in seen:
            raise ValueError(f"duplicate campaign job: {job.factor}/{job.mode}")
        seen.add(key)
        groups.setdefault(job.factor, []).append(job)
    for factor, group in groups.items():
        modes = tuple(job.mode for job in group)
        if modes != MATCH_MODES:
            raise ValueError(
                f"campaign factor {factor} must run maker then taker; got {modes}"
            )
    return list(groups.values())


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


def _factor_filters(data: DataSourceConfig, schema: pa.Schema) -> list[tuple[str, str, object]]:
    filters: list[tuple[str, str, object]] = [("part", "in", list(data.parts))]
    if data.split_ids:
        split_type = schema.field("split_id").type
        if pa.types.is_integer(split_type):
            values: object = [int(value) for value in data.split_ids]
        elif pa.types.is_string(split_type) or pa.types.is_large_string(split_type):
            values = [str(value).zfill(3) for value in data.split_ids]
        else:
            raise ValueError(f"campaign factor split_id type is unsupported: {split_type}")
        filters.append(("split_id", "in", values))
    elif data.split_id is not None:
        split_type = schema.field("split_id").type
        if pa.types.is_integer(split_type):
            value: object = int(data.split_id)
        elif pa.types.is_string(split_type) or pa.types.is_large_string(split_type):
            value = str(data.split_id).zfill(3)
        else:
            raise ValueError(f"campaign factor split_id type is unsupported: {split_type}")
        filters.append(("split_id", "=", value))
    return filters


def _contracts_by_day(
    path: Path,
    *,
    columns: tuple[str, str, str],
    filters: list[tuple[str, str, object]],
    source_name: str,
) -> dict[str, set[str]]:
    parquet = pq.ParquetFile(path)
    missing = sorted(set(columns) - set(parquet.schema_arrow.names))
    if missing:
        raise ValueError(f"{source_name} is missing columns: {missing}")
    contracts: dict[str, set[str]] = {}
    expression = None
    for name, operator, value in filters:
        condition = ds.field(name).isin(value) if operator == "in" else ds.field(name) == value
        expression = condition if expression is None else expression & condition
    dataset = ds.dataset(path, format="parquet")
    batches = dataset.scanner(columns=list(columns), filter=expression, batch_size=131_072).to_batches()
    for batch in batches:
        days = batch.column(columns[0]).to_pylist()
        products = batch.column(columns[1]).to_pylist()
        values = batch.column(columns[2]).to_pylist()
        for day, product, contract in zip(days, products, values, strict=True):
            if day is None or product is None or contract is None:
                raise ValueError(f"{source_name} contains null product/day/contract keys")
            day_key = str(day)
            product_key = str(product).strip().lower()
            contract_key = str(contract).strip().upper()
            if product_key != "ag":
                raise ValueError(f"{source_name} contains unexpected product {product!r}")
            if not contract_key:
                raise ValueError(f"{source_name} contains an empty contract key")
            contracts.setdefault(day_key, set()).add(contract_key)
    return contracts


def validate_main_contract_alignment(
    jobs: list[CampaignJob],
    expected_days: set[str],
    map_path: Path = MAIN_CONTRACT_MAP_PATH,
    *,
    allow_mismatch: bool = False,
) -> dict[str, dict[str, str]]:
    """Reject inputs whose selected stream disagrees with the two-day TRVO map."""

    if not jobs:
        raise ValueError("campaign has no jobs")
    if not expected_days:
        raise ValueError("campaign main-contract check has no trading days")
    first = jobs[0].config.data
    factor_path = Path(first.factor_path).resolve()
    market_path = Path(first.market_path).resolve()
    map_path = Path(map_path).expanduser().resolve()
    if not map_path.is_file():
        raise FileNotFoundError(f"main contract map is missing: {map_path}")

    factor_schema = pq.ParquetFile(factor_path).schema_arrow
    required_factor = {"part", "trading_day", "underlying_secu_cd"}
    if first.split_id is not None or first.split_ids:
        required_factor.add("split_id")
    missing_factor = sorted(required_factor - set(factor_schema.names))
    if missing_factor:
        raise ValueError(f"campaign factor input is missing columns: {missing_factor}")
    day_filters = [("trading_day", "in", sorted(expected_days))]
    factor_contracts = _contracts_by_day(
        factor_path,
        columns=("trading_day", "product", "underlying_secu_cd"),
        filters=[*_factor_filters(first, factor_schema), ("trading_day", "in", sorted(expected_days))],
        source_name="campaign factor input",
    )
    market_contracts = _contracts_by_day(
        market_path,
        columns=("trading_day", "product", "underlying_secu_cd"),
        filters=day_filters,
        source_name="campaign market input",
    )
    map_contracts = _contracts_by_day(
        map_path,
        columns=("trading_day", "product", "main_secu_cd"),
        filters=[("product", "=", "ag"), *day_filters],
        source_name="main contract map",
    )

    def invalid(source: str, values: dict[str, set[str]]) -> None:
        missing = sorted(expected_days - values.keys())
        multiple = {day: sorted(values[day]) for day in expected_days if len(values.get(day, set())) != 1}
        if missing or multiple:
            raise ValueError(
                f"{source} does not contain exactly one ag contract per selected day: "
                f"missing={missing}, multiple={multiple}"
            )

    invalid("campaign factor input", factor_contracts)
    invalid("campaign market input", market_contracts)
    invalid("main contract map", map_contracts)
    mismatches = {
        day: {
            "factor": next(iter(factor_contracts[day])),
            "market": next(iter(market_contracts[day])),
            "expected": next(iter(map_contracts[day])),
        }
        for day in sorted(expected_days)
        if factor_contracts[day] != map_contracts[day] or market_contracts[day] != map_contracts[day]
    }
    if mismatches:
        if allow_mismatch:
            print(
                json.dumps(
                    {
                        "event": "main_contract_mismatch_allowed",
                        "count": len(mismatches),
                        "days": mismatches,
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
            return mismatches
        raise ValueError(
            "campaign main-contract alignment failed against the daily cumulative-TRVO "
            "map with two-consecutive-day rollover: "
            f"{mismatches}"
        )
    return {}

def validate_campaign_input_coverage(jobs: list[CampaignJob]) -> set[str]:
    if not jobs:
        raise ValueError("campaign has no jobs")
    first = jobs[0].config.data
    selection = (
        Path(first.market_path).resolve(),
        Path(first.factor_path).resolve(),
        first.split_id,
        tuple(first.split_ids or ()),
        tuple(first.parts),
    )
    for job in jobs[1:]:
        data = job.config.data
        candidate = (
            Path(data.market_path).resolve(),
            Path(data.factor_path).resolve(),
            data.split_id,
            tuple(data.split_ids or ()),
            tuple(data.parts),
        )
        if candidate != selection:
            raise ValueError("campaign jobs do not share one input selection")

    factor_path = selection[1]
    market_path = selection[0]
    factor_schema = pq.ParquetFile(factor_path).schema_arrow
    required_columns = {"part", "trading_day"}
    if first.split_id is not None or first.split_ids:
        required_columns.add("split_id")
    missing_columns = sorted(required_columns - set(factor_schema.names))
    if missing_columns:
        raise ValueError(f"campaign factor input is missing columns: {missing_columns}")
    factor_filters: list[tuple[str, str, object]] = [("part", "in", list(first.parts))]
    if first.split_ids:
        split_type = factor_schema.field("split_id").type
        if pa.types.is_integer(split_type):
            split_values: object = [int(value) for value in first.split_ids]
        elif pa.types.is_string(split_type) or pa.types.is_large_string(split_type):
            split_values = [str(value).zfill(3) for value in first.split_ids]
        else:
            raise ValueError(f"campaign factor split_id type is unsupported: {split_type}")
        factor_filters.append(("split_id", "in", split_values))
    elif first.split_id is not None:
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


def _execute_job(job: CampaignJob, validation: dict) -> dict:
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
    return {
        "factor": job.factor,
        "mode": job.mode,
        "output_root": str(job.output_root),
        "report_root": str(job.report_root),
        "summary": summary,
        "report": report,
    }


def _run_factor_group_child(
    group_index: int,
    total_groups: int,
    total_jobs: int,
    items: list[tuple[int, CampaignJob, dict]],
    execute_job: Callable[[CampaignJob, dict], dict],
    result_queue,
) -> None:
    try:
        print(
            json.dumps(
                {
                    "event": "factor_start",
                    "index": group_index + 1,
                    "total": total_groups,
                    "factor": items[0][1].factor,
                }
            ),
            flush=True,
        )
        results: list[tuple[int, dict]] = []
        for job_index, job, validation in items:
            print(
                json.dumps(
                    {
                        "event": "start",
                        "index": job_index + 1,
                        "total": total_jobs,
                        "factor": job.factor,
                        "mode": job.mode,
                    }
                ),
                flush=True,
            )
            results.append((job_index, execute_job(job, validation)))
            print(
                json.dumps(
                    {
                        "event": "complete",
                        "index": job_index + 1,
                        "total": total_jobs,
                        "factor": job.factor,
                        "mode": job.mode,
                    }
                ),
                flush=True,
            )
        result_queue.put(("ok", group_index, results))
    except Exception as exc:  # noqa: BLE001 - propagate every job failure to the parent
        result_queue.put(
            ("error", group_index, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
        )


def _factor_worker_context():
    return multiprocessing.get_context("spawn")


def _execute_factor_groups(
    jobs: list[CampaignJob],
    validations: list[dict],
    *,
    factor_workers: int,
    execute_job: Callable[[CampaignJob, dict], dict] = _execute_job,
) -> list[dict]:
    groups = group_jobs_by_factor(jobs)
    validation_by_job = {
        (job.factor, job.mode): validation
        for job, validation in zip(jobs, validations, strict=True)
    }
    position_by_job = {
        (job.factor, job.mode): index
        for index, job in enumerate(jobs)
    }
    group_items = [
        [
            (
                position_by_job[(job.factor, job.mode)],
                job,
                validation_by_job[(job.factor, job.mode)],
            )
            for job in group
        ]
        for group in groups
    ]
    context = _factor_worker_context()
    result_queue = context.Queue()
    active: dict[int, multiprocessing.Process] = {}
    completed: dict[int, list[tuple[int, dict]]] = {}
    deferred_messages: list[tuple[str, int, object]] = []
    blocked_dead: set[int] = set()
    next_group = 0

    def start_group(group_index: int) -> None:
        items = group_items[group_index]
        process = context.Process(
            target=_run_factor_group_child,
            args=(
                group_index,
                len(groups),
                len(jobs),
                items,
                execute_job,
                result_queue,
            ),
        )
        process.start()
        active[group_index] = process

    try:
        while next_group < len(groups) and len(active) < factor_workers:
            start_group(next_group)
            next_group += 1

        while active:
            if deferred_messages:
                status, group_index, payload = deferred_messages.pop(0)
            else:
                try:
                    status, group_index, payload = result_queue.get(timeout=0.5)
                except queue.Empty as exc:
                    dead = [
                        index for index, process in active.items() if not process.is_alive()
                    ]
                    if not dead:
                        continue
                    blocked_dead.update(dead)
                    failed_index = dead[0]
                    deadline = time.monotonic() + 1
                    while True:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            process = active[failed_index]
                            raise RuntimeError(
                                "factor worker exited without a result: "
                                f"{groups[failed_index][0].factor} "
                                f"(exitcode={process.exitcode})"
                            ) from exc
                        try:
                            message = result_queue.get(timeout=remaining)
                        except queue.Empty:
                            process = active[failed_index]
                            raise RuntimeError(
                                "factor worker exited without a result: "
                                f"{groups[failed_index][0].factor} "
                                f"(exitcode={process.exitcode})"
                            ) from exc
                        if message[1] in dead:
                            status, group_index, payload = message
                            break
                        deferred_messages.append(message)

            process = active.pop(group_index)
            process.join()
            blocked_dead.discard(group_index)
            if status != "ok":
                raise RuntimeError(payload)
            if process.exitcode != 0:
                raise RuntimeError(
                    f"factor worker exited with code {process.exitcode}: "
                    f"{groups[group_index][0].factor}"
                )
            completed[group_index] = payload
            print(
                json.dumps(
                    {
                        "event": "factor_complete",
                        "index": group_index + 1,
                        "total": len(groups),
                        "factor": groups[group_index][0].factor,
                    }
                ),
                flush=True,
            )
            blocked_dead.update(
                index for index, child in active.items() if not child.is_alive()
            )
            if (
                next_group < len(groups)
                and not blocked_dead
                and not deferred_messages
            ):
                start_group(next_group)
                next_group += 1
    finally:
        for process in active.values():
            if process.is_alive():
                process.terminate()
        for process in active.values():
            process.join()
        result_queue.close()
        result_queue.join_thread()

    results_by_position = {
        position: result
        for group_index in range(len(groups))
        for position, result in completed[group_index]
    }
    return [results_by_position[index] for index in range(len(jobs))]


def execute_campaign(
    jobs: list[CampaignJob],
    *,
    isolate_jobs: bool = False,
    factor_workers: int = 1,
    allow_main_contract_mismatch: bool = True,
) -> list[dict]:
    if not isinstance(factor_workers, int) or isinstance(factor_workers, bool):
        raise TypeError("factor_workers must be an integer from 1 to 6")
    if not 1 <= factor_workers <= len(FACTOR_COLUMNS):
        raise ValueError("factor_workers must be from 1 to 6")
    selected_days = validate_campaign_input_coverage(jobs)
    if selected_days:
        validate_main_contract_alignment(jobs, selected_days, allow_mismatch=allow_main_contract_mismatch)
    preflight_targets(jobs)
    validations: list[dict] = []
    for job in jobs:
        validation = validate_config(job.config)
        if not validation.get("passed"):
            raise ValueError(
                f"validation failed for {job.factor}/{job.mode}: {validation.get('errors', [])}"
            )
        validations.append(validation)

    if isolate_jobs or factor_workers > 1:
        return _execute_factor_groups(
            jobs,
            validations,
            factor_workers=factor_workers,
        )

    results: list[dict] = []
    for index, (job, validation) in enumerate(zip(jobs, validations, strict=True), start=1):
        print(json.dumps({"event": "start", "index": index, "total": len(jobs), "factor": job.factor, "mode": job.mode}), flush=True)
        result = _execute_job(job, validation)
        results.append(result)
        print(json.dumps({"event": "complete", "index": index, "total": len(jobs), "factor": job.factor, "mode": job.mode}), flush=True)
    return results


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base_config = load_config(BASE_CONFIG_PATH)
    if args.scope == "sample":
        split_id = "015"
        split_ids = None
        output_base = SAMPLE_OUTPUT_BASE
        result_base = SAMPLE_RESULT_BASE
    else:
        validate_full_split_coverage(FACTOR_PATH)
        split_id = None
        split_ids = sorted(EXPECTED_TEST_SPLITS)
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
        split_ids=split_ids,
    )
    results = execute_campaign(
        jobs,
        isolate_jobs=True,
        factor_workers=args.factor_workers,
    )
    print(json.dumps({"scope": args.scope, "completed": len(results), "results": results}, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

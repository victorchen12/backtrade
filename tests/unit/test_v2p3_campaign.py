from __future__ import annotations

import multiprocessing
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

from backtrade.config.schema import BacktradeConfig


def _campaign_module():
    from scripts import run_v2p3_rolling_campaign

    return run_v2p3_rolling_campaign


def _base_config() -> BacktradeConfig:
    return BacktradeConfig.model_validate(
        {
            "initial_cash": 1_000_000,
            "data": {"product": "ag", "max_ticks": 1, "eof_is_day_end": False},
            "contracts": {
                "ag": {
                    "code": "AG",
                    "exchange": "SHFE",
                    "tick_size": 1.0,
                    "multiplier": 15.0,
                    "fee": {
                        "open": {"mode": "rate", "value": 0.00005},
                        "close": {"mode": "rate", "value": 0.00005},
                        "close_today": {"mode": "rate", "value": 0.00005},
                    },
                    "price_limit": {"mode": "percent", "value": 0.2},
                }
            },
        }
    )


def _jobs(tmp_path: Path):
    campaign = _campaign_module()
    return campaign.build_jobs(
        _base_config(),
        split_id="015",
        output_base=tmp_path / "raw",
        result_base=tmp_path / "reports",
        market_path=tmp_path / "market.parquet",
        factor_path=tmp_path / "factors.parquet",
        factor_manifest_path=tmp_path / "factor_bundle_manifest.json",
    )


def _spawn_worker_execute_job(job, validation):
    factor = job.factor
    mode = job.mode
    events_path = Path(validation["events_path"])
    with events_path.open("a", encoding="utf-8") as stream:
        stream.write(f"{factor}:{mode}\n")
    if set_event := validation.get("set_event"):
        set_event.set()
    if wait_event := validation.get("wait_event"):
        wait_event.wait(timeout=5)

    action = validation.get("action")
    if action == "barrier":
        validation["barrier"].wait(timeout=10)
    elif action == "raise":
        raise RuntimeError("forced worker failure")
    elif action == "wait":
        validation["hold_event"].wait(timeout=10)
    elif action == "hard_exit":
        time.sleep(0.05)
        os._exit(7)
    elif action == "delay":
        time.sleep(0.7)
    return {"factor": factor, "mode": mode}


def test_campaign_builds_six_factor_by_two_mode_jobs_without_yaml_duplication(tmp_path: Path) -> None:
    campaign = _campaign_module()
    jobs = _jobs(tmp_path)

    assert len(jobs) == 12
    assert [(job.factor, job.mode) for job in jobs] == [
        (factor, mode) for factor in campaign.FACTOR_COLUMNS for mode in ("maker", "taker")
    ]
    for job in jobs:
        assert job.config.data.split_id == "015"
        assert job.config.data.parts == ["test"]
        assert job.config.data.eof_is_day_end is True
        assert job.config.data.max_ticks is None
        assert job.config.strategy.signal_mode == "ecdf_tail"
        assert job.config.strategy.short_threshold == -0.8
        assert job.config.strategy.long_threshold == 0.8
        assert job.output_root == tmp_path / "raw" / job.factor / job.mode
        assert job.report_root == tmp_path / "reports" / job.factor / job.mode


def test_campaign_preflights_every_target_before_running(tmp_path: Path) -> None:
    campaign = _campaign_module()
    jobs = _jobs(tmp_path)
    blocked = jobs[-1].report_root
    blocked.mkdir(parents=True)
    (blocked / "existing.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match=str(blocked)):
        campaign.preflight_targets(jobs)


def test_campaign_preflight_rejects_missing_market_days(tmp_path: Path) -> None:
    campaign = _campaign_module()
    jobs = _jobs(tmp_path)
    factor_path = jobs[0].config.data.factor_path
    market_path = jobs[0].config.data.market_path
    pd.DataFrame(
        {
            "split_id": [15, 15],
            "part": ["test", "test"],
            "trading_day": ["2025-04-15", "2025-04-16"],
        }
    ).to_parquet(factor_path, index=False)
    pd.DataFrame({"trading_day": ["2025-04-15"]}).to_parquet(market_path, index=False)

    with pytest.raises(ValueError, match="missing.*2025-04-16"):
        campaign.validate_campaign_input_coverage(jobs)

    pd.DataFrame(
        {"trading_day": ["2025-04-15", "2025-04-16"]}
    ).to_parquet(market_path, index=False)
    assert campaign.validate_campaign_input_coverage(jobs) == {
        "2025-04-15",
        "2025-04-16",
    }


def test_campaign_validates_all_jobs_before_sequential_fail_fast_execution(tmp_path: Path, monkeypatch) -> None:
    campaign = _campaign_module()
    jobs = _jobs(tmp_path)
    events: list[tuple[str, str, str]] = []

    def fake_validate(config):
        events.append(("validate", config.strategy.factor_name, config.match.mode))
        return {"passed": True, "errors": []}

    def fake_run(config, *, output_root, **kwargs):
        events.append(("run", config.strategy.factor_name, config.match.mode))
        passed = len([event for event in events if event[0] == "run"]) == 1
        return {"audit": {"passed": passed, "errors": [] if passed else ["forced failure"]}}

    def fake_report(run_root, result_root, *, factor_name, mode):
        events.append(("report", factor_name, mode))
        return {"root": str(Path(result_root) / factor_name / mode)}

    monkeypatch.setattr(campaign, "validate_config", fake_validate)
    monkeypatch.setattr(campaign, "validate_campaign_input_coverage", lambda jobs: set())
    monkeypatch.setattr(campaign, "run_from_config", fake_run)
    monkeypatch.setattr(campaign, "generate_backtest_report", fake_report)

    with pytest.raises(RuntimeError, match="audit failed"):
        campaign.execute_campaign(jobs)

    assert [event[0] for event in events[:12]] == ["validate"] * 12
    assert [event[0] for event in events[12:]] == ["run", "report", "run"]


def test_campaign_parser_requires_explicit_sample_or_full_scope() -> None:
    campaign = _campaign_module()
    parser = campaign.build_parser()
    assert parser.parse_args(["--scope", "sample"]).scope == "sample"
    assert parser.parse_args(["--scope", "full"]).scope == "full"
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_campaign_parser_validates_factor_worker_count() -> None:
    campaign = _campaign_module()
    parser = campaign.build_parser()

    assert parser.parse_args(["--scope", "full"]).factor_workers == 1
    assert parser.parse_args(["--scope", "full", "--factor-workers", "4"]).factor_workers == 4
    for value in ("0", "7", "not-an-int"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--scope", "full", "--factor-workers", value])


@pytest.mark.parametrize(
    ("value", "error_type"),
    [(0, ValueError), (7, ValueError), (True, TypeError), (1.5, TypeError)],
)
def test_campaign_execute_rejects_invalid_factor_worker_count(
    value,
    error_type,
    monkeypatch,
) -> None:
    campaign = _campaign_module()
    monkeypatch.setattr(campaign, "validate_campaign_input_coverage", lambda jobs: set())

    with pytest.raises(error_type, match="factor_workers"):
        campaign.execute_campaign([], factor_workers=value)


def test_campaign_groups_maker_and_taker_jobs_by_factor(tmp_path: Path) -> None:
    campaign = _campaign_module()

    groups = campaign.group_jobs_by_factor(_jobs(tmp_path))

    assert len(groups) == len(campaign.FACTOR_COLUMNS)
    assert [
        (group[0].factor, [job.mode for job in group])
        for group in groups
    ] == [
        (factor, ["maker", "taker"])
        for factor in campaign.FACTOR_COLUMNS
    ]


def test_campaign_rejects_incomplete_or_reordered_factor_group(tmp_path: Path) -> None:
    campaign = _campaign_module()
    jobs = _jobs(tmp_path)

    for invalid_group in ([jobs[0]], [jobs[1], jobs[0]]):
        with pytest.raises(ValueError, match="maker.*taker"):
            campaign.group_jobs_by_factor(invalid_group)


def test_campaign_factor_workers_use_spawn_context() -> None:
    campaign = _campaign_module()

    assert campaign._factor_worker_context().get_start_method() == "spawn"


def test_campaign_runs_factor_groups_concurrently_and_keeps_modes_ordered(
    tmp_path: Path,
) -> None:
    campaign = _campaign_module()
    jobs = _jobs(tmp_path)[:4]
    context = multiprocessing.get_context("spawn")
    maker_barrier = context.Barrier(2)
    events_path = tmp_path / "events.txt"
    validations = [
        {
            "events_path": str(events_path),
            "action": "barrier" if job.mode == "maker" else None,
            "barrier": maker_barrier,
        }
        for job in jobs
    ]

    results = campaign._execute_factor_groups(
        jobs,
        validations,
        factor_workers=2,
        execute_job=_spawn_worker_execute_job,
    )

    assert [(result["factor"], result["mode"]) for result in results] == [
        (job.factor, job.mode) for job in jobs
    ]
    events = [
        line.strip().split(":")
        for line in events_path.read_text(encoding="utf-8").splitlines()
    ]
    for factor in campaign.FACTOR_COLUMNS[:2]:
        assert [mode for event_factor, mode in events if event_factor == factor] == [
            "maker",
            "taker",
        ]


def test_campaign_stops_pending_factor_groups_after_worker_failure(
    tmp_path: Path,
) -> None:
    campaign = _campaign_module()
    jobs = _jobs(tmp_path)[:6]
    first_factor, second_factor, pending_factor = campaign.FACTOR_COLUMNS[:3]
    context = multiprocessing.get_context("spawn")
    second_started = context.Event()
    hold_second_worker = context.Event()
    events_path = tmp_path / "failure-events.txt"
    validations = []
    for job in jobs:
        validation = {"events_path": str(events_path)}
        if job.factor == first_factor and job.mode == "maker":
            validation.update({"action": "raise", "wait_event": second_started})
        elif job.factor == second_factor and job.mode == "maker":
            validation.update(
                {
                    "action": "wait",
                    "set_event": second_started,
                    "hold_event": hold_second_worker,
                }
            )
        validations.append(validation)

    with pytest.raises(RuntimeError, match="forced worker failure"):
        campaign._execute_factor_groups(
            jobs,
            validations,
            factor_workers=2,
            execute_job=_spawn_worker_execute_job,
        )

    events = events_path.read_text(encoding="utf-8").splitlines()
    assert any(line.startswith(f"{first_factor}:") for line in events)
    assert any(line.startswith(f"{second_factor}:") for line in events)
    assert not any(line.startswith(f"{pending_factor}:") for line in events)


def test_campaign_does_not_start_pending_group_after_hard_worker_exit(
    tmp_path: Path,
) -> None:
    campaign = _campaign_module()
    jobs = _jobs(tmp_path)[:6]
    first_factor, second_factor, pending_factor = campaign.FACTOR_COLUMNS[:3]
    context = multiprocessing.get_context("spawn")
    second_started = context.Event()
    events_path = tmp_path / "hard-exit-events.txt"
    validations = []
    for job in jobs:
        validation = {"events_path": str(events_path)}
        if job.factor == first_factor and job.mode == "maker":
            validation.update(
                {"action": "hard_exit", "wait_event": second_started}
            )
        elif job.factor == second_factor and job.mode == "maker":
            validation.update(
                {"action": "delay", "set_event": second_started}
            )
        validations.append(validation)

    with pytest.raises(RuntimeError, match=f"exited without a result: {first_factor}"):
        campaign._execute_factor_groups(
            jobs,
            validations,
            factor_workers=2,
            execute_job=_spawn_worker_execute_job,
        )

    events = events_path.read_text(encoding="utf-8").splitlines()
    assert any(line.startswith(f"{first_factor}:") for line in events)
    assert any(line.startswith(f"{second_factor}:") for line in events)
    assert not any(line.startswith(f"{pending_factor}:") for line in events)


def test_campaign_uses_confirmed_24_26_market_input() -> None:
    campaign = _campaign_module()

    assert campaign.MARKET_PATH == Path(
        "/data1/cws/future_l2/v2p3_dataset_24_26/raw_input/ag_con_tick.parquet"
    )


def test_campaign_direct_script_resolves_backtrade_from_its_own_worktree(tmp_path: Path) -> None:
    campaign = _campaign_module()
    script_path = Path(campaign.__file__).resolve()
    probe = (
        "import runpy; "
        f"namespace = runpy.run_path({str(script_path)!r}, run_name='campaign_probe'); "
        "print(','.join(sorted(namespace['StrategyConfig'].model_fields)))"
    )

    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "signal_mode" in completed.stdout.strip().split(",")


@pytest.mark.parametrize(
    "split_ids, missing_or_extra",
    [
        ([f"{index:03d}" for index in range(31)], "missing.*031"),
        ([f"{index:03d}" for index in range(33)], "unexpected.*032"),
    ],
)
def test_full_scope_requires_exact_32_test_splits(
    tmp_path: Path,
    split_ids: list[str],
    missing_or_extra: str,
) -> None:
    campaign = _campaign_module()
    factor_path = tmp_path / "factors.parquet"
    pd.DataFrame(
        {
            "split_id": split_ids,
            "part": ["test"] * len(split_ids),
        }
    ).to_parquet(factor_path, index=False)

    with pytest.raises(ValueError, match=missing_or_extra):
        campaign.validate_full_split_coverage(factor_path)

    pd.DataFrame(
        {
            "split_id": [f"{index:03d}" for index in range(32)],
            "part": ["test"] * 32,
        }
    ).to_parquet(factor_path, index=False)
    assert campaign.validate_full_split_coverage(factor_path) == {
        f"{index:03d}" for index in range(32)
    }

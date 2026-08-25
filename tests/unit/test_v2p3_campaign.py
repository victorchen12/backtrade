from __future__ import annotations

from pathlib import Path
import subprocess
import sys

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

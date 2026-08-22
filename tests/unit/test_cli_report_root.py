from __future__ import annotations

import backtrade.reporting as reporting
import pytest

from backtrade.cli import build_parser
from backtrade.config.loader import load_config, validate_result_view_root
from backtrade.reporting import generate_backtest_report


RESULT_VIEW_ROOT = "./result_view"


def test_run_and_validate_accept_result_view_root():
    for command in ("validate", "run"):
        args = build_parser().parse_args(
            [
                command,
                "--config",
                "configs/l1_imbalance_single_day_taker.yaml",
                "--result-view-root",
                f"{RESULT_VIEW_ROOT}/team-check",
            ]
        )
        assert args.result_view_root == f"{RESULT_VIEW_ROOT}/team-check"


def test_result_view_root_accepts_user_selected_path(tmp_path):
    report_root = tmp_path / "backtrade-report"
    assert validate_result_view_root(report_root) == report_root

    with pytest.raises(ValueError, match="result_view_root"):
        validate_result_view_root("/mnt/nvme/backtrade-report")


def test_config_accepts_user_selected_result_view_root(tmp_path):
    config_path = tmp_path / "run.yaml"
    report_root = tmp_path / "backtrade-report"
    config_path.write_text(
        "\n".join(
            [
                "paths:",
                f"  result_view_root: {report_root}",
                "initial_cash: 1000",
                "data:",
                "  product: ag",
                "  eof_is_day_end: true",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config(config_path)
    assert cfg.paths.result_view_root == report_root


def test_report_root_cannot_be_inside_run_root(tmp_path, monkeypatch):
    run_root = tmp_path / "run"
    run_root.mkdir()
    monkeypatch.setattr(
        reporting,
        "load_report_data",
        lambda _: {
            "manifest": {
                "factor_name": "my_ofi",
                "match_mode": "taker",
                "config": {"initial_cash": 1000},
            },
            "accounts": [],
            "snapshots": [],
            "activity": [],
        },
    )
    monkeypatch.setattr(reporting, "compute_report_metrics", lambda *args, **kwargs: {})
    monkeypatch.setattr(reporting, "render_report_html", lambda *args, **kwargs: "")

    with pytest.raises(ValueError, match="report root.*run root"):
        generate_backtest_report(run_root, run_root / "result_view")

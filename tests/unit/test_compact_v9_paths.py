from pathlib import Path

from backtrade.cli import _load_and_validate, build_parser


def test_run_and_validate_accept_explicit_data_paths():
    for command in ("validate", "run"):
        args = build_parser().parse_args(
            [
                command,
                "--config",
                "configs/l1_imbalance_single_day_taker.yaml",
                "--market-path",
                "/tmp/team/market_ticks.parquet",
                "--factor-path",
                "/tmp/team/l1_imbalance.parquet",
            ]
        )
        assert args.market_path.endswith("market_ticks.parquet")
        assert args.factor_path.endswith("l1_imbalance.parquet")


def test_explicit_data_paths_override_input_root(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "initial_cash: 1000",
                "data:",
                "  product: ag",
                "  eof_is_day_end: true",
            ]
        ),
        encoding="utf-8",
    )
    input_root = tmp_path / "input-root"
    market_path = tmp_path / "custom-market.parquet"
    factor_path = tmp_path / "custom-factor.parquet"
    cfg, _ = _load_and_validate(
        str(config_path),
        None,
        str(input_root),
        None,
        None,
        str(market_path),
        str(factor_path),
    )
    assert cfg.data.market_path == market_path.resolve()
    assert cfg.data.factor_path == factor_path.resolve()

def test_prepare_input_accepts_custom_file_paths():
    args = build_parser().parse_args(
        [
            "prepare-input",
            "--product",
            "ag",
            "--market-path",
            "/tmp/team/market_ticks.parquet",
            "--factor-path",
            "/tmp/team/l1_imbalance.parquet",
        ]
    )
    assert args.root is None

def test_config_accepts_user_selected_write_and_report_roots(tmp_path):
    config_path = tmp_path / "config.yaml"
    project_root = tmp_path / "project"
    output_root = tmp_path / "results"
    report_root = tmp_path / "reports"
    config_path.write_text(
        "\n".join(
            [
                f"paths:\n  project_root: {project_root}\n  output_root: {output_root}\n  result_view_root: {report_root}",
                "initial_cash: 1000",
                "data:",
                "  product: ag",
                "  eof_is_day_end: true",
            ]
        ),
        encoding="utf-8",
    )
    from backtrade.config.loader import load_config

    cfg = load_config(config_path)
    assert cfg.paths.project_root == project_root
    assert cfg.paths.output_root == output_root
    assert cfg.paths.result_view_root == report_root

def test_prepare_input_writes_manifest_next_to_custom_factor(tmp_path):
    import pandas as pd
    import pyarrow.parquet as pq

    from backtrade.cli import _prepare_input
    from backtrade.data.future_l2 import MARKET_COLUMNS

    market_path = tmp_path / "ticks.custom.parquet"
    factor_path = tmp_path / "features" / "l1.custom.parquet"
    factor_path.parent.mkdir()
    values = {column: [1.0] for column in MARKET_COLUMNS}
    values.update(
        {
            "trading_day": ["2026-01-05"],
            "session_id": ["day"],
            "tick_ts": [pd.Timestamp("2026-01-05 09:00:00")],
            "underlying_secu_cd": ["ag2601"],
            "vol_inc": [1],
            "amt_inc": [15.0],
            **{f"bid{i}_qty": [1] for i in range(1, 6)},
            **{f"ask{i}_qty": [1] for i in range(1, 6)},
        }
    )
    pq.write_table(pd.DataFrame(values).to_arrow() if hasattr(pd.DataFrame(values), "to_arrow") else __import__("pyarrow").Table.from_pandas(pd.DataFrame(values)), market_path)
    pq.write_table(__import__("pyarrow").Table.from_pandas(pd.DataFrame({"tick_ts": [pd.Timestamp("2026-01-05 09:00:00")], "l1_imbalance": [1.0]})), factor_path)

    result = _prepare_input(None, "ag", str(market_path), str(factor_path))

    assert Path(result["manifest"]) == factor_path.with_name("manifest.json")
    assert factor_path.with_name("manifest.json").is_file()

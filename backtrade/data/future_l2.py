from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from backtrade.config.schema import BacktradeConfig
from backtrade.data.limit_reference import (
    PriceLimitReference,
    load_prev_day_vwap_limit_references,
    load_price_limit_reference_snapshot,
)
from backtrade.data.market_quality import derive_market_quality_flags
from backtrade.data.tabular import read_table, resolve_table_candidate, table_columns, table_format
from backtrade.simulation.events import MarketTick
from backtrade.strategies.factors import L1_IMBALANCE_NAME, validate_factor_name
from backtrade.simulation.state import OrderSide


JOIN_KEYS = ["product", "trading_day", "session_id", "tick_ts", "underlying_secu_cd"]
FACTOR_GROUP_KEYS = ["product", "trading_day", "session_id", "underlying_secu_cd"]
MARKET_COLUMNS = [
    "trading_day", "session_id", "tick_ts", "underlying_secu_cd", "last_prc", "vol_inc", "amt_inc",
    "bid1_prc", "bid2_prc", "bid3_prc", "bid4_prc", "bid5_prc",
    "ask1_prc", "ask2_prc", "ask3_prc", "ask4_prc", "ask5_prc",
    "bid1_qty", "bid2_qty", "bid3_qty", "bid4_qty", "bid5_qty",
    "ask1_qty", "ask2_qty", "ask3_qty", "ask4_qty", "ask5_qty",
]
MARKET_OPTIONAL_COLUMNS = [
    "product", "secu_cd", "vohp", "snd_ts", "mid1", "spread1",
    "cancel_bid_tick", "cancel_ask_tick", "cancel_total_tick", "cancel_imbalance_tick",
    "cancel_reliability_score", "stale_ms", "cancel_event_flag", "quote_change_flag",
    "side_ambiguous_flag", "level_shift_flag", "is_anomaly", "is_stale",
    "adj_factor", "last_prc_adj",
]
FACTOR_NAME = L1_IMBALANCE_NAME


def _split_text(cfg: BacktradeConfig) -> str:
    return str(cfg.data.split_id or "000").zfill(3)


def processed_market_path(cfg: BacktradeConfig) -> Path:
    # [README-1] 数据输入路径：显式 market_path 优先，否则按 future_l2_data_root 推导。
    if cfg.data.market_path is not None:
        return Path(cfg.data.market_path)
    return resolve_table_candidate(
        cfg.paths.future_l2_data_root / "pre_data" / "continuous_main_tick",
        f"{cfg.data.product}_con_tick",
    )


def selected_factor_screen_path(cfg: BacktradeConfig) -> Path:
    # [README-1] 因子输入路径：显式 factor_path 优先，manifest.json 与其同目录。
    if cfg.data.factor_path is not None:
        return Path(cfg.data.factor_path)
    return resolve_table_candidate(
        cfg.paths.future_l2_data_root / "factor_data" / str(cfg.data.product).lower(),
        cfg.strategy.factor_column,
    )


def factor_grid_mode(cfg: BacktradeConfig) -> str:
    if cfg.data.factor_grid_mode != "decision_grid":
        raise ValueError("compact_v9 requires factor_grid_mode=decision_grid")
    return "decision_grid"


def _clean_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["tick_ts"] = pd.to_datetime(out["tick_ts"])
    out["product"] = out["product"].astype(str).str.lower()
    out["trading_day"] = out["trading_day"].astype(str)
    out["session_id"] = out["session_id"].astype(str)
    out["underlying_secu_cd"] = out["underlying_secu_cd"].astype(str).str.lower()
    return out


def _require_keys(frame: pd.DataFrame, name: str) -> None:
    missing = sorted(set(JOIN_KEYS) - set(frame.columns))
    if missing:
        raise KeyError(f"{name} is missing JOIN_KEYS: {missing}")
    nulls = [key for key in JOIN_KEYS if frame[key].isna().any()]
    if nulls:
        raise ValueError(f"{name} has null JOIN_KEYS: {nulls}")


def enrich_factor_keys(market: pd.DataFrame, factors: pd.DataFrame, *, factor_name: str = FACTOR_NAME) -> pd.DataFrame:
    """Attach market context to a minimal ``tick_ts``/factor input.

    A minimal factor file is deliberately exact-tick keyed.  If a timestamp
    maps to more than one market row, the input is ambiguous and is rejected
    instead of guessing a contract or session.
    """

    factor_name = validate_factor_name(factor_name)
    if "tick_ts" not in factors.columns or factor_name not in factors.columns:
        raise KeyError(f"minimal factor input requires tick_ts and {factor_name}")
    context_keys = set(JOIN_KEYS) - {"tick_ts"}
    partial = context_keys & set(factors.columns)
    if partial and partial != context_keys:
        raise ValueError("factor context columns must be complete JOIN_KEYS or omitted")
    market_clean = _clean_keys(market)
    factor_clean = factors.copy()
    factor_clean["tick_ts"] = pd.to_datetime(factor_clean["tick_ts"])
    if factor_clean["tick_ts"].isna().any():
        raise ValueError("minimal factor tick_ts contains null values")
    if factor_clean.duplicated(["tick_ts"]).any():
        raise ValueError("duplicate factor tick_ts")
    if market_clean.duplicated(["tick_ts"]).any():
        raise ValueError("minimal factor tick_ts does not uniquely identify a market tick")
    if not factor_clean[factor_name].map(lambda value: pd.notna(value) and np.isfinite(float(value))).all():
        raise ValueError(f"{factor_name} contains non-finite values")
    context = market_clean[[*JOIN_KEYS]].copy()
    merged = factor_clean.merge(context, on="tick_ts", how="left", validate="one_to_one", suffixes=("", "_market"))
    missing = merged["product"].isna()
    if bool(missing.any()):
        raise ValueError("minimal factor tick_ts does not match a market tick")
    for key in JOIN_KEYS:
        if key != "tick_ts":
            merged[key] = merged[key].astype(str)
    merged["active_factor"] = merged[factor_name].astype(float)
    merged["factor_source_ts"] = merged["tick_ts"]
    merged["factor_decision"] = True
    merged["factor_age_ms"] = 0.0
    out = _clean_keys(merged)
    out["underlying_secu_cd"] = out["underlying_secu_cd"].str.upper()
    return out


def merge_market_and_factors(market: pd.DataFrame, factors: pd.DataFrame, *, factor_grid_mode: str = "decision_grid", factor_name: str = FACTOR_NAME) -> pd.DataFrame:
    # [README-3] 只做 backward as-of 对齐；因子源 tick 才触发新决策。
    factor_name = validate_factor_name(factor_name)
    if factor_grid_mode != "decision_grid":
        raise ValueError("compact_v9 only supports causal decision_grid joins")
    _require_keys(market, "market")
    _require_keys(factors, "factors")
    left = _clean_keys(market)
    right = _clean_keys(factors)
    if right.duplicated(JOIN_KEYS).any():
        raise ValueError("duplicate factor JOIN_KEYS")
    if factor_name not in right.columns:
        raise KeyError(f"factor input is missing {factor_name}")
    if "active_factor" not in right.columns:
        right["active_factor"] = right[factor_name].astype(float)
    factor_cols = [column for column in ("active_factor", factor_name) if column in right.columns]
    right = right[[*FACTOR_GROUP_KEYS, "tick_ts", *factor_cols]].rename(columns={"tick_ts": "factor_source_ts"})
    right["factor_decision"] = True
    left["_market_order"] = np.arange(len(left), dtype="int64")
    left = left.sort_values(["tick_ts", *FACTOR_GROUP_KEYS], kind="stable")
    right = right.sort_values(["factor_source_ts", *FACTOR_GROUP_KEYS], kind="stable")
    aligned = pd.merge_asof(left, right, left_on="tick_ts", right_on="factor_source_ts", by=FACTOR_GROUP_KEYS, direction="backward", allow_exact_matches=True)
    aligned["factor_decision"] = aligned["factor_source_ts"].notna() & (aligned["tick_ts"] == aligned["factor_source_ts"])
    aligned["factor_age_ms"] = (aligned["tick_ts"] - pd.to_datetime(aligned["factor_source_ts"])).dt.total_seconds() * 1000.0
    aligned = aligned.sort_values("_market_order", kind="stable").drop(columns="_market_order")
    return aligned


def _factor_manifest_path(cfg: BacktradeConfig) -> Path:
    return selected_factor_screen_path(cfg).with_name("manifest.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_factor_manifest(cfg: BacktradeConfig, factor_path: Path) -> dict:
    factor_name = cfg.strategy.factor_column
    factor_name = validate_factor_name(factor_name)
    manifest_path = _factor_manifest_path(cfg)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"canonical factor manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"canonical factor manifest cannot be read: {manifest_path}") from exc
    allowed_versions = {f"{factor_name}_v1", f"{factor_name}_minimal_v1"}
    if manifest.get("schema_version") not in allowed_versions:
        raise ValueError(f"{factor_name} factor manifest version is invalid")
    if manifest.get("factor_columns") != [factor_name]:
        raise ValueError(f"factor manifest must expose only {factor_name}")
    expected_hash = manifest.get("factor_values_sha256")
    if not isinstance(expected_hash, str) or _sha256(factor_path) != expected_hash:
        raise ValueError("canonical factor input hash does not match adjacent manifest")
    declared_path = manifest.get("factor_values_keyed_path") or manifest.get("factor_values_path")
    if declared_path and Path(declared_path).name != factor_path.name:
        raise ValueError("canonical factor manifest points to a different input")
    declared_product = manifest.get("product")
    if declared_product is not None and str(declared_product).lower() != str(cfg.data.product).lower():
        raise ValueError("canonical factor manifest product does not match configured product")
    market_path = processed_market_path(cfg).expanduser().resolve()
    if not market_path.is_file():
        raise FileNotFoundError(f"configured market input is missing: {market_path}")
    declared_market = manifest.get("market_path")
    declared_market_hash = manifest.get("market_sha256")
    if declared_market_hash is not None:
        if not isinstance(declared_market_hash, str) or len(declared_market_hash) != 64:
            raise ValueError("canonical factor manifest market hash is invalid")
        if _sha256(market_path) != declared_market_hash:
            raise ValueError("canonical factor manifest market hash does not match configured market")
    elif manifest.get("schema_version") == f"{factor_name}_minimal_v1":
        raise ValueError("canonical factor manifest market hash is required for minimal inputs")
    elif declared_market:
        declared_market_path = Path(str(declared_market)).expanduser().resolve()
        if declared_market_path != market_path:
            raise ValueError("canonical factor manifest market path does not match configured market")
    else:
        raise ValueError("canonical factor manifest market binding is missing")
    return manifest


def load_factor_frame_for_split(cfg: BacktradeConfig, nrows: int | None = None) -> pd.DataFrame:
    path = selected_factor_screen_path(cfg)
    if not path.is_file():
        raise FileNotFoundError(f"canonical factor input is missing: {path}")
    _validate_factor_manifest(cfg, path)
    names = set(table_columns(path))
    factor_name = cfg.strategy.factor_column
    required = {"tick_ts", factor_name}
    missing = sorted(required - names)
    if missing:
        raise KeyError(f"canonical factor input is missing columns: {missing}")
    has_context = set(JOIN_KEYS).issubset(names)
    partial_context = (set(JOIN_KEYS) - {"tick_ts"}) & names
    if partial_context and not has_context:
        raise ValueError("factor context columns must be complete JOIN_KEYS or omitted")
    columns = [column for column in ["part", "split_id", *JOIN_KEYS, "active_factor", factor_name, "factor_decision", "factor_source_ts", "factor_age_ms"] if column in names]
    frame = read_table(path, columns=columns)
    if cfg.data.parts and "part" in frame:
        frame = frame[frame["part"].astype(str).isin({str(value) for value in cfg.data.parts})]
    if cfg.data.split_id is not None and "split_id" in frame:
        expected_split = str(cfg.data.split_id).zfill(3)
        actual_split = frame["split_id"].astype(str).str.zfill(3)
        frame = frame[actual_split == expected_split]
    if cfg.data.trading_days and "trading_day" in frame:
        frame = frame[frame["trading_day"].astype(str).isin({str(value) for value in cfg.data.trading_days})]
    if frame.empty:
        raise ValueError("canonical factor selection is empty")
    frame["tick_ts"] = pd.to_datetime(frame["tick_ts"])
    if has_context:
        frame = _clean_keys(frame)
        if frame.duplicated(JOIN_KEYS).any():
            raise ValueError("canonical factor input contains duplicate JOIN_KEYS")
    elif frame.duplicated(["tick_ts"]).any():
        raise ValueError("duplicate factor tick_ts")
    if "product" in frame and (frame["product"].astype(str).str.lower() != str(cfg.data.product).lower()).any():
        raise ValueError("canonical factor product does not match configured product")
    if not frame[factor_name].map(lambda value: pd.notna(value) and np.isfinite(float(value))).all():
        raise ValueError(f"{factor_name} contains non-finite values")
    if "active_factor" in frame and not np.allclose(frame["active_factor"].to_numpy(float), frame[factor_name].to_numpy(float), equal_nan=False):
        raise ValueError("active_factor is not an exact alias of the configured factor")
    if nrows is not None:
        if int(nrows) <= 0:
            raise ValueError("nrows must be positive")
        frame = frame.head(int(nrows))
    return frame


def _row_groups_for_days(parquet: pq.ParquetFile, days: set[str]) -> list[int]:
    if not days or "trading_day" not in parquet.schema.names:
        return list(range(parquet.num_row_groups))
    index = parquet.schema.names.index("trading_day")
    selected = []
    for group in range(parquet.num_row_groups):
        stats = parquet.metadata.row_group(group).column(index).statistics
        if stats is None or any(str(stats.min) <= day <= str(stats.max) for day in days):
            selected.append(group)
    return selected


def _read_market(path: Path, days: set[str], *, product: str, tick_size: float) -> pd.DataFrame:
    table_kind = table_format(path)
    names = set(table_columns(path))
    missing = sorted(set(MARKET_COLUMNS) - names)
    if missing:
        raise KeyError(f"market input is missing columns: {missing}")
    columns = [column for column in [*MARKET_COLUMNS, *MARKET_OPTIONAL_COLUMNS] if column in names]
    frames = []
    if table_kind == "parquet":
        parquet = pq.ParquetFile(path)
        offset = 0
        selected = set(_row_groups_for_days(parquet, days))
        for group in range(parquet.num_row_groups):
            count = parquet.metadata.row_group(group).num_rows
            if group in selected:
                frame = parquet.read_row_group(group, columns=columns).to_pandas()
                frame["source_seq"] = np.arange(offset, offset + count, dtype="int64")
                if days:
                    frame = frame[frame["trading_day"].astype(str).isin(days)]
                frames.append(frame)
            offset += count
    else:
        frame = read_table(path, columns=columns)
        frame["source_seq"] = np.arange(len(frame), dtype="int64")
        if days:
            frame = frame[frame["trading_day"].astype(str).isin(days)]
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=[*MARKET_COLUMNS, "product", "source_seq"])
    out = pd.concat(frames, ignore_index=True)
    if "product" not in out:
        out["product"] = str(product).lower()
    else:
        out["product"] = out["product"].fillna(product).astype(str).str.lower()
    if (out["product"] != str(product).lower()).any():
        raise ValueError("market product does not match configured product")
    out = _clean_keys(out)
    return derive_market_quality_flags(out, tick_size=tick_size)


def _infer_direction(last_price: float | None, bid: float, ask: float, vol_inc: int, ambiguous: bool, stale: bool, anomaly: bool) -> tuple[OrderSide | None, str | None, str | None]:
    if last_price is None or vol_inc <= 0 or ambiguous or stale or anomaly:
        return None, None, None
    if last_price >= ask:
        return OrderSide.BUY, "ask_touch", "high"
    if last_price <= bid:
        return OrderSide.SELL, "bid_touch", "high"
    return None, None, None


def _number(row, field: str, default: float = 0.0) -> float:
    value = getattr(row, field, default)
    return float(value) if pd.notna(value) else default


def _tick(row, source_seq: int, limit_reference: PriceLimitReference | None, previous_last_price: float | None) -> MarketTick:
    bid_prices = tuple(float(getattr(row, f"bid{i}_prc")) for i in range(1, 6))
    ask_prices = tuple(float(getattr(row, f"ask{i}_prc")) for i in range(1, 6))
    bid_qtys = tuple(int(getattr(row, f"bid{i}_qty")) for i in range(1, 6))
    ask_qtys = tuple(int(getattr(row, f"ask{i}_qty")) for i in range(1, 6))
    observed = float(row.last_prc) if pd.notna(row.last_prc) else None
    vol_inc = int(row.vol_inc) if pd.notna(row.vol_inc) else 0
    ambiguous = bool(_number(row, "side_ambiguous_flag", 0))
    stale = bool(getattr(row, "is_stale", False))
    anomaly = bool(getattr(row, "is_anomaly", False))
    direction, source, confidence = _infer_direction(observed, bid_prices[0], ask_prices[0], vol_inc, ambiguous, stale, anomaly)
    source_value = getattr(row, "factor_source_ts", None)
    factor_source_ts = source_value.to_pydatetime() if hasattr(source_value, "to_pydatetime") and pd.notna(source_value) else source_value if pd.notna(source_value) else None
    factor_decision = bool(getattr(row, "factor_decision", False))
    raw_factor = getattr(row, "active_factor", None)
    factor_value = float(raw_factor) if raw_factor is not None and pd.notna(raw_factor) else None
    return MarketTick(
        product=str(row.product).lower(), contract=str(row.underlying_secu_cd).upper(),
        tick_ts=row.tick_ts.to_pydatetime() if hasattr(row.tick_ts, "to_pydatetime") else row.tick_ts,
        last_price=observed if observed is not None else float(row.mid1), bid_prices=bid_prices, bid_qtys=bid_qtys,
        ask_prices=ask_prices, ask_qtys=ask_qtys, vol_inc=vol_inc, amount_inc=_number(row, "amt_inc"),
        factors={"active_factor": factor_value} if factor_value is not None else {}, trading_day=str(row.trading_day),
        price_limit_up=limit_reference.limit_up if limit_reference else None, price_limit_down=limit_reference.limit_down if limit_reference else None,
        price_limit_reference_price=limit_reference.reference_price if limit_reference else None,
        price_limit_reference_source=limit_reference.source if limit_reference else None, price_limit_rule_version=limit_reference.rule_version if limit_reference else None,
        source_seq=source_seq, session_id=str(row.session_id), cancel_bid_tick=_number(row, "cancel_bid_tick"), cancel_ask_tick=_number(row, "cancel_ask_tick"),
        cancel_total_tick=_number(row, "cancel_total_tick"), cancel_imbalance_tick=_number(row, "cancel_imbalance_tick"), cancel_reliability_score=_number(row, "cancel_reliability_score"), stale_ms=_number(row, "stale_ms"),
        cancel_event_flag=int(_number(row, "cancel_event_flag")), quote_change_flag=int(_number(row, "quote_change_flag")), side_ambiguous_flag=int(_number(row, "side_ambiguous_flag")), level_shift_flag=int(_number(row, "level_shift_flag")),
        is_anomaly=anomaly, is_stale=stale, trade_direction=direction, trade_direction_source=source, trade_direction_confidence=confidence,
        direction_source=source, direction_confidence=confidence, trade_direction_quality=confidence, direction_quality=confidence,
        factor_decision=factor_decision, factor_source_ts=factor_source_ts, factor_age_ms=float(row.factor_age_ms) if pd.notna(row.factor_age_ms) else None,
    )


def _limit_references(cfg: BacktradeConfig, factors: pd.DataFrame, market_path: Path) -> dict[tuple[str, str], PriceLimitReference]:
    if cfg.limit_reference.mode == "disabled":
        return {}
    required = {(str(row.trading_day), str(row.underlying_secu_cd).upper()) for row in factors[["trading_day", "underlying_secu_cd"]].itertuples(index=False)}
    if cfg.limit_reference.snapshot_path is not None:
        refs = load_price_limit_reference_snapshot(cfg.limit_reference.snapshot_path)
    elif cfg.limit_reference.mode == "prev_day_vwap_proxy":
        refs = load_prev_day_vwap_limit_references(market_path, cfg.contracts, {day for day, _ in required}, cfg.limit_reference.shfe_new_rule_effective_date)
    else:
        raise ValueError("official price-limit mode requires an immutable reference snapshot")
    missing = sorted(required - set(refs))
    if missing:
        raise ValueError(f"price-limit reference coverage is incomplete: missing={len(missing)} first={missing[:5]}")
    if cfg.limit_reference.mode == "official":
        non_official = sorted(key for key in required if refs[key].source != "official")
        if non_official:
            raise ValueError(f"official price-limit snapshot contains non-official sources: {non_official[:5]}")
    if any(refs[key].source == "missing" for key in required):
        raise ValueError("price-limit reference source=missing cannot disable price-limit enforcement")
    return refs


def load_future_l2_frame(cfg: BacktradeConfig, max_events: int | None = None) -> pd.DataFrame:
    if max_events is not None and int(max_events) <= 0:
        raise ValueError("max_events must be positive")
    market_path = processed_market_path(cfg)
    factors = load_factor_frame_for_split(cfg, nrows=None)
    days = set(factors["trading_day"].astype(str)) if "trading_day" in factors else set(cfg.data.trading_days or [])
    market = _read_market(market_path, days, product=cfg.data.product, tick_size=float(cfg.contract_rule().tick_size))
    if "trading_day" not in factors:
        factors = enrich_factor_keys(market, factors, factor_name=cfg.strategy.factor_column)
        days = set(factors["trading_day"].astype(str))
        market = market[market["trading_day"].astype(str).isin(days)].copy()
    aligned = merge_market_and_factors(market, factors, factor_grid_mode="decision_grid", factor_name=cfg.strategy.factor_column)
    if max_events is not None:
        aligned = aligned.head(int(max_events))
    return aligned


def iter_future_l2_ticks(cfg: BacktradeConfig, max_events: int | None = None, batch_size: int = 100_000) -> Iterator[MarketTick]:
    if max_events is not None and int(max_events) <= 0:
        raise ValueError("max_events must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    market_path = processed_market_path(cfg)
    if not market_path.is_file():
        raise FileNotFoundError(f"market input is missing: {market_path}")
    factors = load_factor_frame_for_split(cfg)
    days = set(factors["trading_day"].astype(str)) if "trading_day" in factors else set(cfg.data.trading_days or [])
    market = _read_market(market_path, days, product=cfg.data.product, tick_size=float(cfg.contract_rule().tick_size))
    if "trading_day" not in factors:
        factors = enrich_factor_keys(market, factors, factor_name=cfg.strategy.factor_column)
        days = set(factors["trading_day"].astype(str))
        market = market[market["trading_day"].astype(str).isin(days)].copy()
    refs = _limit_references(cfg, factors, market_path)
    aligned = merge_market_and_factors(market, factors, factor_grid_mode="decision_grid", factor_name=cfg.strategy.factor_column)
    previous_last: dict[tuple[str, str, str, str], float] = {}
    limit = max_events if max_events is not None else cfg.data.max_ticks
    produced = 0
    for row in aligned.itertuples(index=False):
        key = (str(row.product).lower(), str(row.underlying_secu_cd).upper(), str(row.session_id), str(row.trading_day))
        tick = _tick(row, int(row.source_seq), refs.get((str(row.trading_day), str(row.underlying_secu_cd).upper())), previous_last.get(key))
        yield tick
        produced += 1
        if tick.vol_inc > 0 and not tick.side_ambiguous_flag and not tick.is_stale and not tick.is_anomaly:
            previous_last[key] = tick.last_price
        if limit is not None and produced >= int(limit):
            return


def load_future_l2_ticks(cfg: BacktradeConfig, max_events: int | None = None) -> list[MarketTick]:
    return list(iter_future_l2_ticks(cfg, max_events=max_events))


def validate_future_l2_replay(cfg: BacktradeConfig, max_events: int | None = None) -> dict:
    ticks = load_future_l2_ticks(cfg, max_events=max_events)
    violations = sum(1 for left, right in zip(ticks, ticks[1:]) if (right.tick_ts, right.source_seq) <= (left.tick_ts, left.source_seq))
    return {"row_count": len(ticks), "physical_order_monotonic": violations == 0, "physical_order_violations": violations, "missing_factor_count": sum(1 for tick in ticks if tick.factor_decision and "active_factor" not in tick.factors)}


def summarize_future_l2_tick_stream(cfg: BacktradeConfig, max_events: int | None = None, batch_size: int = 100_000) -> dict:
    ticks = load_future_l2_ticks(cfg, max_events=max_events)
    days = sorted({str(tick.trading_day) for tick in ticks})
    contracts = sorted({tick.contract for tick in ticks})
    return {"tick_count": len(ticks), "trading_day_count": len(days), "contract_count": len(contracts), "trading_days": days, "contracts": contracts, "first_ts": str(ticks[0].tick_ts) if ticks else None, "last_ts": str(ticks[-1].tick_ts) if ticks else None}


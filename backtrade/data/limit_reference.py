from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow.parquet as pq

from backtrade.config.schema import ContractRule
from backtrade.data.tabular import read_table, table_columns, table_format


@dataclass(frozen=True, slots=True)
class PriceLimitReference:
    trading_day: str
    contract: str
    reference_price: float | None
    limit_rate: float | None
    limit_up: float | None
    limit_down: float | None
    source: str
    rule_version: str


SNAPSHOT_COLUMNS = (
    "trading_day",
    "contract",
    "reference_price",
    "limit_rate",
    "limit_up",
    "limit_down",
    "source",
    "rule_version",
)
SNAPSHOT_SOURCES = frozenset({"official", "prev_day_vwap_proxy", "missing"})


def load_price_limit_reference_snapshot(
    path: str | Path,
) -> dict[tuple[str, str], PriceLimitReference]:
    """Load and validate the immutable price-limit reference snapshot."""

    snapshot_path = Path(path).expanduser()
    if not snapshot_path.is_file():
        raise FileNotFoundError(f"price-limit reference snapshot not found: {snapshot_path}")
    names = set(table_columns(snapshot_path))
    missing = [column for column in SNAPSHOT_COLUMNS if column not in names]
    if missing:
        raise ValueError(f"price-limit reference snapshot missing columns: {missing}")
    frame = read_table(snapshot_path, columns=list(SNAPSHOT_COLUMNS))
    frame = frame.copy()
    frame["trading_day"] = frame["trading_day"].astype(str)
    frame["contract"] = frame["contract"].astype(str).str.upper()
    frame["source"] = frame["source"].astype(str)
    frame["rule_version"] = frame["rule_version"].astype(str)
    key_columns = ["trading_day", "contract"]
    if frame.duplicated(key_columns).any():
        raise ValueError("price-limit reference snapshot has duplicate trading_day/contract keys")
    unknown_sources = sorted(set(frame["source"]) - SNAPSHOT_SOURCES)
    if unknown_sources:
        raise ValueError(f"price-limit reference snapshot has unsupported sources: {unknown_sources}")

    numeric_columns = ["reference_price", "limit_rate", "limit_up", "limit_down"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    missing_source = frame["source"].eq("missing")
    populated = ~missing_source
    if frame.loc[populated, numeric_columns].isna().any(axis=None):
        raise ValueError("non-missing price-limit references must contain all numeric values")
    if (frame.loc[populated, "reference_price"] <= 0).any():
        raise ValueError("price-limit reference prices must be positive")
    if (frame.loc[populated, "limit_up"] < frame.loc[populated, "limit_down"]).any():
        raise ValueError("price-limit upper bound cannot be below lower bound")
    if frame["rule_version"].eq("").any():
        raise ValueError("price-limit reference snapshot has blank rule_version")

    references: dict[tuple[str, str], PriceLimitReference] = {}
    for row in frame.itertuples(index=False):
        references[(str(row.trading_day), str(row.contract))] = PriceLimitReference(
            trading_day=str(row.trading_day),
            contract=str(row.contract),
            reference_price=float(row.reference_price) if pd.notna(row.reference_price) else None,
            limit_rate=float(row.limit_rate) if pd.notna(row.limit_rate) else None,
            limit_up=float(row.limit_up) if pd.notna(row.limit_up) else None,
            limit_down=float(row.limit_down) if pd.notna(row.limit_down) else None,
            source=str(row.source),
            rule_version=str(row.rule_version),
        )
    return references


def _product_from_contract(contract: str) -> str:
    return "".join(ch for ch in contract if ch.isalpha()).lower()


def _rule_for_contract(contract: str, contracts: dict[str, ContractRule]) -> ContractRule | None:
    normalized = contract.upper()
    if normalized in contracts:
        return contracts[normalized]
    product = _product_from_contract(contract)
    return contracts.get(product) or contracts.get(product.upper())


def _rule_version(trading_day: str, new_rule_effective_date: date) -> str:
    day = pd.to_datetime(trading_day).date()
    return "shfe_20260528_new" if day >= new_rule_effective_date else "shfe_pre_20260528"


def compute_prev_day_vwap_limit_references(
    market: pd.DataFrame,
    contracts: dict[str, ContractRule],
    new_rule_effective_date: date = date(2026, 5, 28),
    target_days: set[str] | None = None,
) -> dict[tuple[str, str], PriceLimitReference]:
    """Build approximate price-limit references from previous observed same-contract VWAP."""

    if market.empty:
        return {}
    required = {"trading_day", "underlying_secu_cd", "vol_inc", "amt_inc"}
    missing = required.difference(market.columns)
    if missing:
        raise ValueError(f"missing columns for price-limit VWAP reference: {sorted(missing)}")

    guard_column = "mid1" if "mid1" in market.columns else "last_prc" if "last_prc" in market.columns else None
    columns = list(required) + ([guard_column] if guard_column else [])
    frame = market[columns].copy()
    frame["trading_day"] = frame["trading_day"].astype(str)
    frame["contract"] = frame["underlying_secu_cd"].astype(str).str.upper()
    frame["vol_inc"] = pd.to_numeric(frame["vol_inc"], errors="coerce").fillna(0.0)
    frame["amt_inc"] = pd.to_numeric(frame["amt_inc"], errors="coerce").fillna(0.0)
    if guard_column:
        frame["guard_price"] = pd.to_numeric(frame[guard_column], errors="coerce")
    frame = frame[(frame["vol_inc"] > 0) & (frame["amt_inc"] > 0)]
    if frame.empty:
        return {}

    daily_rows = []
    for (contract, trading_day), daily_frame in frame.groupby(["contract", "trading_day"], sort=True):
        rule = _rule_for_contract(str(contract), contracts)
        if rule is None:
            continue
        daily_frame = daily_frame.copy()
        daily_frame["implied_price"] = daily_frame["amt_inc"] / daily_frame["vol_inc"] / rule.multiplier
        reference_frame = daily_frame
        guard_price = None
        if "guard_price" in daily_frame.columns:
            guard_values = daily_frame["guard_price"].dropna()
            guard_values = guard_values[guard_values > 0]
            if not guard_values.empty:
                guard_price = float(guard_values.median())
                lower = guard_price * 0.5
                upper = guard_price * 1.5
                guarded = daily_frame[
                    (daily_frame["implied_price"] >= lower)
                    & (daily_frame["implied_price"] <= upper)
                ]
                if not guarded.empty:
                    reference_frame = guarded
        volume = float(reference_frame["vol_inc"].sum())
        if volume <= 0:
            continue
        reference_price = float(reference_frame["amt_inc"].sum()) / volume / rule.multiplier
        if guard_price is not None and not (guard_price * 0.5 <= reference_price <= guard_price * 1.5):
            reference_price = guard_price
        daily_rows.append(
            {
                "contract": str(contract),
                "trading_day": str(trading_day),
                "reference_price": reference_price,
            }
        )

    grouped = pd.DataFrame(daily_rows).sort_values(["contract", "trading_day"])
    if grouped.empty:
        return {}
    refs: dict[tuple[str, str], PriceLimitReference] = {}
    target_days = {str(day) for day in target_days} if target_days else None
    for contract, contract_frame in grouped.groupby("contract", sort=True):
        contract_frame = contract_frame.sort_values("trading_day").reset_index(drop=True)
        rule = _rule_for_contract(contract, contracts)
        if rule is None or rule.price_limit.mode == "none":
            continue
        for idx in range(1, len(contract_frame)):
            current = contract_frame.iloc[idx]
            day = str(current["trading_day"])
            if target_days is not None and day not in target_days:
                continue
            previous = contract_frame.iloc[idx - 1]
            reference_price = float(previous["reference_price"])
            if reference_price <= 0:
                continue
            if rule.price_limit.mode == "percent":
                limit_rate = rule.price_limit.value_for(contract)
                limit_up = reference_price * (1 + limit_rate)
                limit_down = reference_price * (1 - limit_rate)
            elif rule.price_limit.mode == "absolute":
                limit_rate = rule.price_limit.value_for(contract)
                limit_up = reference_price + limit_rate
                limit_down = max(rule.tick_size, reference_price - limit_rate)
            else:
                continue
            refs[(day, contract)] = PriceLimitReference(
                trading_day=day,
                contract=contract,
                reference_price=reference_price,
                limit_rate=limit_rate,
                limit_up=limit_up,
                limit_down=limit_down,
                source="prev_day_vwap_proxy",
                rule_version=_rule_version(day, new_rule_effective_date),
            )
    return refs


def load_prev_day_vwap_limit_references(
    market_path: str | Path,
    contracts: dict[str, ContractRule],
    target_days: Iterable[str],
    new_rule_effective_date: date,
    batch_size: int = 200_000,
) -> dict[tuple[str, str], PriceLimitReference]:
    target_day_set = {str(day) for day in target_days}
    if not target_day_set:
        return {}

    aggregates: dict[tuple[str, str], list[float]] = {}
    market_path = Path(market_path).expanduser()
    table_kind = table_format(market_path)
    schema_names = set(table_columns(market_path))
    columns = ["trading_day", "underlying_secu_cd", "vol_inc", "amt_inc"]
    if "mid1" in schema_names:
        columns.append("mid1")
    elif "last_prc" in schema_names:
        columns.append("last_prc")
    if table_kind == "parquet":
        parquet = pq.ParquetFile(market_path)
        batches = (batch.to_pandas() for batch in parquet.iter_batches(batch_size=batch_size, columns=columns))
    else:
        batches = (read_table(market_path, columns=columns),)
    for frame in batches:
        frame["trading_day"] = frame["trading_day"].astype(str)
        frame["contract"] = frame["underlying_secu_cd"].astype(str).str.upper()
        frame["vol_inc"] = pd.to_numeric(frame["vol_inc"], errors="coerce").fillna(0.0)
        frame["amt_inc"] = pd.to_numeric(frame["amt_inc"], errors="coerce").fillna(0.0)
        if "mid1" in frame.columns:
            frame["guard_price"] = pd.to_numeric(frame["mid1"], errors="coerce")
        elif "last_prc" in frame.columns:
            frame["guard_price"] = pd.to_numeric(frame["last_prc"], errors="coerce")
        frame = frame[(frame["vol_inc"] > 0) & (frame["amt_inc"] > 0)]
        if frame.empty:
            continue
        agg_dict = {"vol_inc": "sum", "amt_inc": "sum"}
        if "guard_price" in frame.columns:
            agg_dict["guard_price"] = "median"
        grouped = frame.groupby(["contract", "trading_day"], as_index=False).agg(agg_dict)
        for row in grouped.itertuples(index=False):
            key = (str(row.contract), str(row.trading_day))
            if _rule_for_contract(key[0], contracts) is None:
                continue
            acc = aggregates.setdefault(key, [0.0, 0.0, []])
            acc[0] += float(row.vol_inc)
            acc[1] += float(row.amt_inc)
            if hasattr(row, "guard_price") and pd.notna(row.guard_price) and float(row.guard_price) > 0:
                acc[2].append(float(row.guard_price))

    rows = [
        {
            "underlying_secu_cd": contract,
            "trading_day": trading_day,
                "vol_inc": sums[0],
                "amt_inc": sums[1],
                "mid1": float(pd.Series(sums[2]).median()) if len(sums) > 2 and sums[2] else None,
            }
            for (contract, trading_day), sums in aggregates.items()
    ]
    return compute_prev_day_vwap_limit_references(
        pd.DataFrame(rows),
        contracts,
        new_rule_effective_date=new_rule_effective_date,
        target_days=target_day_set,
    )

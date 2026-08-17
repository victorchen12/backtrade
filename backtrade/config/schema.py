from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


FeeMode = Literal["per_lot", "rate", "bps"]
PriceLimitMode = Literal["percent", "absolute", "none"]
LimitReferenceMode = Literal["disabled", "prev_day_vwap_proxy", "official"]
MatchModeName = Literal["maker", "taker"]


def normalize_contract_code(contract_code: str | None) -> str:
    return (contract_code or "").upper()


class FeeRule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: FeeMode
    value: float = Field(ge=0)
    contract_overrides: dict[str, float] = Field(default_factory=dict)

    def value_for(self, contract_code: str | None) -> float:
        return {normalize_contract_code(k): v for k, v in self.contract_overrides.items()}.get(
            normalize_contract_code(contract_code), self.value
        )


class FeeSet(BaseModel):
    model_config = ConfigDict(extra="forbid")
    open: FeeRule
    close: FeeRule
    close_today: FeeRule


class PriceLimitRule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: PriceLimitMode
    value: float = Field(ge=0)
    contract_overrides: dict[str, float] = Field(default_factory=dict)

    def value_for(self, contract_code: str | None) -> float:
        return {normalize_contract_code(k): v for k, v in self.contract_overrides.items()}.get(
            normalize_contract_code(contract_code), self.value
        )


class SessionWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: str
    end: str


class DaySessionRule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    morning_start: str
    morning_break: str
    morning_resume: str
    morning_end: str
    afternoon_start: str
    afternoon_end: str


class ContractRule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: str | None = None
    exchange: str
    tick_size: float = Field(gt=0)
    multiplier: float = Field(gt=0)
    tick_value: float | None = Field(default=None, gt=0)
    fee: FeeSet
    price_limit: PriceLimitRule
    night_session: bool | SessionWindow = False
    day_session: DaySessionRule | None = None
    close_today_rule: str = "same_trading_day"

    @model_validator(mode="after")
    def validate_lot_bounds(self) -> "ContractRule":
        if self.tick_value is not None and abs(self.tick_value - self.tick_size * self.multiplier) > 1e-9:
            raise ValueError("tick_value must equal tick_size * multiplier when provided")
        return self


class PathConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project_root: Path = Path("/home/cws/QUANT/Backtrade")
    output_root: Path = Path("/data1/cws/backtrade")
    future_l2_data_root: Path = Path("/data1/cws/future_l2/dataset")

    @field_validator("project_root", "output_root", "future_l2_data_root", mode="before")
    @classmethod
    def expand_path(cls, value: str | Path) -> Path:
        return Path(str(value)).expanduser()


class DataSourceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: Literal["future_l2"] = "future_l2"
    product: str
    split_id: str | None = None
    max_ticks: int | None = Field(default=None, gt=0)
    market_path: Path | None = None
    factor_path: Path | None = None
    parts: list[str] = Field(default_factory=lambda: ["test"])
    trading_days: list[str] | None = None
    factor_grid_mode: Literal["decision_grid"] = "decision_grid"
    eof_is_day_end: bool = False

    @model_validator(mode="after")
    def validate_eof_contract(self) -> "DataSourceConfig":
        if self.max_ticks is None and not self.eof_is_day_end:
            raise ValueError("unbounded real-data runs must explicitly set data.eof_is_day_end=true")
        return self


class StrategyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    factor_name: Literal["ofi_cks_best_level_5s"] = "ofi_cks_best_level_5s"
    factor_column: Literal["ofi_cks_best_level_5s"] = "ofi_cks_best_level_5s"


class RiskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_daily_loss: float = Field(default=1_000_000.0, ge=0)
    max_holding_ms: int = Field(default=86_400_000, ge=0)
    stop_on_capital_depleted: bool = True
    capital_floor: float = 0.0


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    latency_ms: int = Field(default=5, ge=0)
    day_end_flatten_window_ms: int = Field(default=5_000, ge=0)

    @property
    def latency(self) -> timedelta:
        return timedelta(milliseconds=self.latency_ms)


class MatchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: MatchModeName = "taker"


class LimitReferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: LimitReferenceMode = "disabled"
    snapshot_path: Path | None = None
    shfe_new_rule_effective_date: date = date(2026, 5, 28)


class BacktradeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    contract_files: list[Path] = Field(default_factory=list)
    paths: PathConfig = Field(default_factory=PathConfig)
    data: DataSourceConfig
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    match: MatchConfig = Field(default_factory=MatchConfig)
    limit_reference: LimitReferenceConfig = Field(default_factory=LimitReferenceConfig)
    contracts: dict[str, ContractRule] = Field(default_factory=dict)
    initial_cash: float

    def contract_rule(self, product: str | None = None) -> ContractRule:
        key = product or self.data.product
        if key in self.contracts:
            return self.contracts[key]
        for candidate in (key.lower(), key.upper(), "".join(ch for ch in key if ch.isalpha()).lower()):
            if candidate in self.contracts:
                return self.contracts[candidate]
        raise ValueError(f"missing contract rule for {key}")

    def require_contract_for_real_run(self) -> None:
        if self.data.product not in self.contracts and self.data.product.lower() not in self.contracts:
            raise ValueError("real-data backtest requires a local contract rule")

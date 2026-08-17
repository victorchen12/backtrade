# Backtrade compact_v9 回测内核

Backtrade 是面向期货 L2 盘口快照和时间序列因子的确定性回测内核。
当前只保留 `future_l2`、`ofi_cks_best_level_5s`、`maker`/`taker` 和单品种单手账户。

## 快速运行

```sh
python -m backtrade.cli validate --config configs/ofi_compact_v9_single_day.yaml
python -m backtrade.cli run --config configs/ofi_compact_v9_single_day.yaml --output-root /data1/cws/backtrade/campaigns/<新目录>
python -m backtrade.cli inspect --run /data1/cws/backtrade/campaigns/<新目录>
```

`ofi_compact_v9_single_day_maker.yaml` 使用严格 MBP expected-queue 撮合。
完整无界流必须显式设置 `eof_is_day_end: true` 才把 EOF 当作已知日末；
使用 `--max-events` 或 `data.max_ticks` 的抽样运行始终以 `end_of_data`
结束。零日末窗口不会把第一 tick 当作边界。

## 输入契约

市场 parquet 必须提供 `backtrade.data.future_l2.MARKET_COLUMNS` 中的 L1-L5
盘口、成交增量、质量标记和连接键。因子 parquet 必须是 canonical
`ofi_cks_best_level_5s`，并在同目录有 `manifest.json`，声明
`ofi_cks_best_level_5s_v1`、唯一因子列和 parquet SHA-256。

只允许因果 `decision_grid` backward as-of join：因子源时间与行情时间完全
相等才设置 `factor_decision=true`，中间行情只携带最近的已完成因子。价格
限制参考必须覆盖实际交易日/合约，`source=missing` 直接失败。团队可以使用
自己的快照和因子值，但必须整理成上述列契约并提供合约、费用和限制配置。

## 信号、撮合和账务

`ofi_sign_v1`：正值目标做多，负值目标做空，零值保持持仓；反向信号先平仓。
持仓只允许 `-1/0/+1`。Taker 在到达 tick 以对手 L1 成交。Maker 是保守的
MBP expected-queue 模型，不是交易所 FIFO：

- 等价挂单价的主动成交先消耗 `queue_ahead`，队列到零才成交；
- 只有高置信度、方向相反且成交价严格穿过挂单价才是 trade-through；
- stale、anomaly、side-ambiguous 不成交也不推进队列；
- 初次不在 L1 或会立即吃单则拒绝，排队后离开 L1 则撤单。

账户是单手模型，不模拟多手、保证金或组合持仓。开仓成交的 `net_pnl` 为
`-open_fee`，平仓成交为 `gross_pnl-close_fee`；审计检查逐笔现金、累计
PnL/费用守恒和最终 flat，并保留 `close_today` 费用规则。

## 产物

run 目录固定包含 `activity_ledger.parquet`、`account_ledger.parquet`、
`state_snapshots.parquet`，maker 模式额外有 `maker_events.parquet`，以及
`manifest.json`。manifest 记录 config digest、实际解析的 market/factor/
factor manifest/价格限制/合约文件身份、最终文件哈希、Git revision/dirty、
延迟、EOF 边界和撮合声明。reader 和
`back_logicvalid/scripts/audit_compact_v9.py` 会拒绝旧 schema、额外文件、哈希
不一致或缺失输入身份。

`prev_day_vwap_proxy` 是上一交易日同合约 VWAP 近似，不是官方结算价。
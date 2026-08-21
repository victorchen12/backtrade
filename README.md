# Backtrade v0.1.0

Backtrade 是面向期货 L2 盘口快照的确定性模拟交易回测框架。
v0.1 只正式支持一个 L1 盘口量不平衡因子、单品种、单手、
Taker 和 Maker 两种撮合模式。

## 使用边界

当前版本不模拟多手、保证金、组合持仓、交易所 FIFO 重建或官方结算价。
Maker 使用保守的 MBP expected-queue 估计。
价格限制可以关闭，也可以由使用者提供明确的参考快照。

路径可以在 YAML、profile 或命令行中修改。
代码中的 [README-n] 注释用于定位本文对应章节。

## 1. 输入文件和表头

一次回测可以使用市场快照加已有因子，也可以只使用市场快照后现场生成因子。

### 1.1 市场快照

市场文件必须是 parquet，至少包含：

\`\`\`text
trading_day, session_id, tick_ts, underlying_secu_cd,
last_prc, vol_inc, amt_inc,
bid1_prc, bid2_prc, bid3_prc, bid4_prc, bid5_prc,
ask1_prc, ask2_prc, ask3_prc, ask4_prc, ask5_prc,
bid1_qty, bid2_qty, bid3_qty, bid4_qty, bid5_qty,
ask1_qty, ask2_qty, ask3_qty, ask4_qty, ask5_qty
\`\`\`

\`product\` 可以省略，由 YAML 的 \`data.product\` 补齐。
价格必须有限且符合盘口单调性；数量必须非负。
\`last_prc_adj\` 和 \`adj_factor\` 只用于报告展示，不改变成交和账务价格。

### 1.2 L1 因子文件

已有因子文件至少包含：

\`\`\`text
tick_ts, l1_imbalance
\`\`\`

也可以提供完整上下文：

\`\`\`text
product, trading_day, session_id, underlying_secu_cd,
tick_ts, l1_imbalance
\`\`\`

\`tick_ts\` 必须唯一，并且必须对应市场快照中的实际 tick。
因子文件同目录必须有 \`manifest.json\`，其中绑定因子哈希、市场哈希、
商品和 \`factor_columns=["l1_imbalance"]\`。

### 1.3 从市场快照生成因子

\`l1_imbalance\` 的定义是：

\`\`\`text
(bid1_qty - ask1_qty) / (bid1_qty + ask1_qty)
\`\`\`

当买一和卖一数量都为零时，结果为 0。
只使用当前 tick 的买一和卖一数量，不读取未来行情。

## 2. 路径和 YAML

建议仓库外准备输入目录：

\`\`\`text
input/
├── market.parquet
└── l1_imbalance.parquet
\`\`\`

也可以直接在命令行传入任意文件路径。
推荐从 \`configs/l1_imbalance_single_day_taker.yaml\` 或
\`configs/l1_imbalance_single_day_maker.yaml\` 复制配置。

关键字段：

\`\`\`yaml
paths:
  project_root: .
  output_root: ./runs/l1-imbalance-taker
  future_l2_data_root: ./input
  result_view_root: ./result_view

data:
  product: ag
  market_path: ./input/market.parquet
  factor_path: ./input/l1_imbalance.parquet
  factor_grid_mode: decision_grid
  eof_is_day_end: true

strategy:
  factor_name: l1_imbalance
  factor_column: l1_imbalance

match:
  mode: taker
\`\`\`

\`run --market-path\` 和 \`run --factor-path\` 会覆盖 YAML 中的路径。
输出目录和报告目录必须是新的空目录，框架拒绝覆盖已有产物。

## 3. 数据回放和因子对齐

Data 层先校验市场、因子和 manifest 身份，再执行因果
\`decision_grid\` backward as-of join。

只有因子源 tick 设置 \`factor_decision=true\`。
中间行情只携带最近一个已完成的因子值，不会使用未来因子。

完整未截断的数据流需要设置 \`eof_is_day_end: true\`。
如果使用 \`--max-events\` 或 \`max_ticks\`，末尾只视为 \`end_of_data\`。

## 4. 策略和信号

当前策略语义为 \`signed_factor_v1\`：

- 因子大于 0：目标持有一手多头；
- 因子小于 0：目标持有一手空头；
- 因子等于 0：保持当前仓位；
- 反向信号：先平仓，再开反向仓位。

策略输出目标仓位、因子值、因子源时间和因子延迟。
后续撮合和会计模块不重新计算信号。

## 5. Simulation、订单和撮合

Taker 在订单到达 tick 使用对手 L1 成交。
Maker 在决策时挂同侧 L1，并使用 MBP expected-queue 估计。

Maker 规则：

- 等价挂单价先消耗 queue_ahead，队列到零后才成交；
- 只有成交价严格穿过挂单价才是 trade-through；
- stale、anomaly、side-ambiguous 行情不成交也不推进队列；
- 初次不在 L1 或会立即吃单的挂单进入 rejected；
- 已排队后离开 L1 才 cancel。

## 6. 会计、产物和报告

成交是唯一账务事件。

开仓成交的逐行 \`net_pnl\` 为 \`-open_fee\`。
平仓成交的逐行 \`net_pnl\` 为 \`gross_pnl-close_fee\`。

必须满足：

\`\`\`text
sum(net_pnl) = final_cash - initial_cash
sum(net_pnl) = realized_pnl - total_fee
\`\`\`

运行目录包含：

\`\`\`text
activity_ledger.parquet
account_ledger.parquet
state_snapshots.parquet
maker_events.parquet  # 仅 Maker
manifest.json
\`\`\`

\`inspect\` 会校验 manifest、parquet schema、文件哈希、成交时序、
现金和 PnL 守恒以及最终空仓。
审计通过后才生成 HTML 报告。

## 7. 最短命令流程

进入仓库根目录：

\`\`\`sh
cd /path/to/Backtrade
\`\`\`

只有市场快照时：

\`\`\`sh
python -m backtrade.cli derive-factor \
  --product ag \
  --market-path /data/team/market.parquet \
  --factor-path /data/team/l1_imbalance.parquet

python -m backtrade.cli prepare-input \
  --product ag \
  --market-path /data/team/market.parquet \
  --factor-path /data/team/l1_imbalance.parquet \
  --factor-column l1_imbalance
\`\`\`

已有两份输入时，直接执行 \`prepare-input\` 即可。

先校验：

\`\`\`sh
python -m backtrade.cli validate \
  --config configs/l1_imbalance_single_day_taker.yaml \
  --market-path /data/team/market.parquet \
  --factor-path /data/team/l1_imbalance.parquet \
  --trading-days 2026-01-05 2026-01-06 2026-01-07
\`\`\`

运行 Taker：

\`\`\`sh
python -m backtrade.cli run \
  --config configs/l1_imbalance_single_day_taker.yaml \
  --market-path /data/team/market.parquet \
  --factor-path /data/team/l1_imbalance.parquet \
  --trading-days 2026-01-05 2026-01-06 2026-01-07 \
  --output-root /data/team/runs/l1-taker \
  --result-view-root /data/team/result_view/l1-taker
\`\`\`

运行 Maker 时只替换配置文件：

\`\`\`sh
python -m backtrade.cli run \
  --config configs/l1_imbalance_single_day_maker.yaml \
  --market-path /data/team/market.parquet \
  --factor-path /data/team/l1_imbalance.parquet \
  --trading-days 2026-01-05 2026-01-06 2026-01-07 \
  --output-root /data/team/runs/l1-maker \
  --result-view-root /data/team/result_view/l1-maker
\`\`\`

检查产物：

\`\`\`sh
python -m backtrade.cli inspect \
  --run /data/team/runs/l1-taker
\`\`\`

## 8. 发布和验收边界

v0.1.0 是首个团队内部可用版本，不代表接口已经稳定到 v1.0。
单手和无保证金是当前模型边界，也是暂不发布 v1.0 的重要原因。

提交前必须确认：

- 输入市场和因子属于同一商品、合约和交易日期；
- 因子 manifest 的市场哈希和因子哈希通过；
- \`validate\` 通过；
- \`run\` 返回 audit passed；
- \`inspect\` 显示最终空仓；
- HTML、manifest 和 parquet 文件都存在；
- 真实数据和运行产物没有进入 Git 仓库。

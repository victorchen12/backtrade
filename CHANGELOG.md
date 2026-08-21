# 更新记录

## v0.1.1

- 支持在 YAML 和 \`prepare-input\` 中配置任意安全的有符号因子名称。
- \`derive-factor\` 仍仅生成内置 \`l1_imbalance\`，不会凭空推导用户自定义因子。

## v0.1.0

- 首个团队内部可用版本。
- 提供内置 \`l1_imbalance\` 示例和 \`signed_factor_v1\` 语义。
- 支持 Taker 与 Maker 两种撮合模式。
- 保留严格 maker 等价价队列、trade-through 和异常行情证据。
- 保留因果 decision_grid、EOF 日末边界和现金/PnL 守恒审计。
- 明确单品种、单手、无保证金、无组合持仓、无官方结算价和无交易所 FIFO 重建。

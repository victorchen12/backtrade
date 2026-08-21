# 贡献说明

所有核心修改必须保持 Data、Strategy、订单与撮合、Simulation、Accounting 和
产物审计之间的边界。

v0.1 的正式输入路径是 L2 市场快照和用户配置的有符号时间序列因子；
\`l1_imbalance\` 只是内置示例及可由 L2 派生的因子。
因子名称由 YAML 的 \`strategy.factor_name\` 和 \`factor_column\` 指定，
并且必须在因果 decision_grid 上对齐，不得读取未来数据。

修改撮合、生命周期或会计逻辑前，先增加针对性回归测试。
禁止提交真实 parquet、HTML 报告、运行日志、私有 profile 和临时文件。

提交前运行：

\`\`\`sh
python -m pytest --collect-only -q
python -m pytest -q
python -m compileall -q backtrade
git diff --check
\`\`\`

外部审计目录不属于本仓库。

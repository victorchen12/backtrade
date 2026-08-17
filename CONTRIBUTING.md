# Contributing

Backtrade changes must preserve the separation between strategy targets,
execution, matching, accounting, and artifact audit. Add a focused regression
test before changing core behavior.

From the repository root:

    python -m backtrade.cli check
    python -m backtrade.cli validate --config configs/ofi_compact_v9_single_day.yaml

The supported runtime path is canonical future_l2 market snapshots joined to
ofi_cks_best_level_5s on a causal decision grid. Do not add synthetic, full-tick,
external-directional, matrix, or legacy compact branches. Generated parquet,
logs, profiles with private paths, and temporary files do not belong in commits.
Every completed run must remain auditable: preserve the fixed compact_v9 schema,
input identities, final file hashes, latency evidence, and single-lot limits.

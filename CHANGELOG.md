# Changelog

## compact_v9 core recovery baseline

- Restored a valid Git repository at the Backtrade project root.
- Kept only the canonical future_l2 plus ofi_cks_best_level_5s decision-grid path.
- Fixed EOF/day-end distinction, zero-window lifecycle handling, strict maker
  equality/trade-through evidence, and single-lot cash/PnL conservation.
- Added final artifact hashes, input identities, config digest, and measured Git
  provenance to the compact_v9 manifest and audit.
- Removed unconnected external-directional, synthetic, legacy matching, and old
  compact branches from the supported tree.

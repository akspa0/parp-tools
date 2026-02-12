# Memory Bank - WoWRollback.RollbackTool (2025-11-08)

## Current Progress (concise)
- Main issue: LK ADT positions → Alpha WDT writeout (compute MAIN offsets, embed ADTs).
- MPQ overlay precedence implemented; DBC `DBFilesClient/*` prefers locale patches; plain patch support.
- Tee logging enabled via `--log-dir`/`--log-file`.

## TODOs (concise)
- LK ADT positions: compute MAIN offsets; embed ADTs; validate with `TryIsAlphaV18Wdt`; emit `tiles_written.csv`.
- Tests: `ArchiveLocator` ordering (incl. plain patch) and `MpqArchiveSource` DBC locale-first path.
- Liquids/placements tuning and diagnostics.

## Navigation
- activeContext.md — Source of truth for current work (focus + TODOs).
- progress.md — Snapshot of current status and upcoming items.
- systemPatterns.md — Key patterns (overlay precedence, WDT fallback).
- techContext.md — Runtime/env and module overview.
- projectbrief.md — Goals/Scope/Non‑goals.
- productContext.md — Why/UX principles/Limitations.

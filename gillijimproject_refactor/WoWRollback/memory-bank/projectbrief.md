# WoWRollback - Time-Travel Your WoW Maps


## Project Brief (concise)

### What it is
WoWRollback is a toolkit to “time‑travel” WoW maps by burying late‑added objects, fixing terrain, and exporting compatible LK ADTs. It is CLI‑first with a lean README and detailed context in memory‑bank/.

### Main Outstanding Issue
LK ADT positions → Alpha WDT writeout. Compute correct MAIN offsets and embed ADT payloads; validate round‑trip.

### Goals
- Roll back Alpha WDT/ADT content by UniqueID threshold (bury placements, fix MCNK holes, optional MCSH zeroing).
- Export LK ADTs and always emit `<Map>.wdt` in LK output.
- Map AreaIDs via CSV crosswalks (strict, no cross‑map leakage; 0.6.0 pivot opt‑in).
- Maintain MPQ overlay parity with client and DBC locale‑first safeguard.

### Scope
- Alpha → LK pipeline (analyze, modify, map, export) driven by `WoWRollback.Cli`.
- Diagnostics: tee logging (`--log-dir`/`--log-file`), kept/dropped assets CSVs, `objects_written.csv`, `mclq_summary.csv`.
- WDT tile presence fallback when archive scan yields zero tiles.

### Constraints
- BYOD: no copyrighted assets are included; user supplies clients/DBCs.
- CSV‑only crosswalk contract; never accept cross‑map results.
- Stable CSV schemas; GUI acts as a runner (orchestration is legacy).

### Current Focus
- Fix LK ADT positions → Alpha WDT writeout.
- Liquids: MH2O→MCLQ correctness; flags/heights; reduce `dont_render`.
- Placements: union MDNM/MONM; never gate; recompute MCRF; per‑tile counts.
- Logging: plain‑patch counts; optional DBC source line.

### Links
- memory-bank/activeContext.md — current focus, decisions, TODOs
- memory-bank/progress.md — snapshot of progress
- memory-bank/systemPatterns.md — overlay precedence, WDT fallback, mapping rules
- memory-bank/techContext.md — runtime, modules, CLI, overlay details


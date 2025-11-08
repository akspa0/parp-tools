# Product Context - WoWRollback.RollbackTool

## Product Brief (concise)

### Why
Create historical snapshots of WoW maps by burying late‑added placements, fixing terrain holes, and exporting compatible LK ADTs.

### Main Outstanding Issue
LK ADT positions → Alpha WDT writeout. Compute correct MAIN offsets, embed ADT payloads, validate round‑trip.

### How it works (pipeline)
- Phase 1 – Rollback (Alpha WDT): bury MDDF/MODF by UniqueID threshold, clear MCNK holes (all‑buried rule), optional MCSH zeroing, write MD5.
- Phase 2 – AreaTable mapping: CSV crosswalks via `--crosswalk-dir|--crosswalk-file`; `--area-remap-json` fallback; write 0 when unmapped; `Map.dbc` used only as guard.
- Phase 3 – Export LK ADTs: convert and write `<map>_x_y.adt` plus `<Map>.wdt` in LK output.

### UX principles
- CLI‑first; GUI is a runner (overlay + inline logs; no modals).
- Energy‑efficient preflight (skip‑if‑exists); BYOD.

### Current focus
- Liquids: MH2O→MCLQ correctness (flags/heights; reduce `dont_render`).
- Placements: union MDNM/MONM; never gate; recompute MCRF; per‑tile MDDF/MODF counts.
- Logging: tee logs (`--log-dir`/`--log-file`); optional DBC source MPQ line; plain‑patch counts.

### Known limitations
- Only works on extracted WDT files (not in MPQs... yet)
- Doesn't modify _obj0.adt or _obj1.adt files (terrain objects) — Won't fix (MapUpconverter already exists!)
- MCNK spatial calculation assumes flat terrain (good enough for most cases)
- Pre‑generation requires disk space (1–2 MB per map)

### Links
- memory-bank/activeContext.md — current focus & TODOs
- memory-bank/progress.md — snapshot of progress
- memory-bank/systemPatterns.md — overlay precedence & mapping rules
- memory-bank/techContext.md — runtime, modules, CLI


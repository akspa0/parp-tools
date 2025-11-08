# Active Context (2025-11-08)

## Current Progress (concise)
- Main issue: LK ADT positions data must be generated and written into Alpha WDT (MAIN offsets + embedded ADT data).
- MPQ overlay precedence implemented and verified in CLI logs.
- Plain patch detection: `patch(.locale).MPQ` treated as numeric order 1 in `ArchiveLocator`.
- DBFilesClient precedence fix: prefer locale patch MPQs before root patch MPQs in `MpqArchiveSource` for `DBFilesClient/*` (corrects `Map.dbc`).
- Tee logging: `--log-dir` and `--log-file` mirror console to a file.
- Alpha WDT monolithic pack: build completes; placements diagnostics CSVs available; liquids path present (needs tuning).

## TODOs (concise)
- LK ADT positions → Alpha WDT:
  - Derive per‑tile presence/ordering from LK inputs; compute MAIN offsets table.
  - Embed per‑tile ADT payloads into Alpha WDT and update MHDR/MCIN alignments.
  - Validate offsets/sizes with `TryIsAlphaV18Wdt` and round‑trip read.
  - Emit `tiles_written.csv` with tile offsets and byte sizes.
- Tests: `ArchiveLocator` ordering (incl. plain patch) and `MpqArchiveSource` DBC resolution.
- Logs: include plain-patch counts in numeric sections; add one-line DBC resolution source when verbose.
- Verify: run representative maps (CataclysmCTF, development) and confirm Map.dbc and overlay precedence.
- Placements: rebuild MDNM/MONM from union; never gate placements; recompute MCRF; per‑tile MDDF/MODF counts.
- Liquids: instrument MH2O→MCLQ, write `mclq_summary.csv`; verify flags/heights and reduce `dont_render`.
- Textures: tileset resize option (256) with alpha preservation.

## Current Focus
- CLI-first pipeline; GUI acts as a runner for Load → Prepare → Layers (no modal popups; overlay + inline logs).
- Energy-efficient preflight: skip work when cache outputs already exist (LK ADTs, tile_layers.csv, layers.json, areas.csv, DBCTool crosswalks).
- Presets: management in Settings; add “Load Preset” on the Load page.
- Adopt CsvHelper in GUI for robust CSV parsing (tolerant headers, 7/8-col variants).
- Gate Area Groups unless areas.csv is present and non-empty (show inline note otherwise).
- BYOD: never bundle copyrighted data; resolve from user-provided locations only.
- Global heatmap (build scope) with `heatmap_stats.json` persisted at build root.
- Layers scope control: Tile | Selection | Map; restore per‑tile lists deterministically.
- FDID pipeline: resolver + CSV enrichment (`fdid`, `asset_path`) + unresolved diagnostics.
- MCCV analyzer and overlay: detect “hidden by holes” and export per‑tile PNGs.
- CASC recompile routing: prefer LK client for WDT lookup; fall back to manual file picker.

## Decisions
- CLI-first with GUI runner and overlay; auto-tab navigation (Load→Build, Prepare→Layers); remove success/info modals.
- Keep CSV cache schema stable; normalize `tile_layers.csv` naming; GUI may fall back to `<map>_tile_layers.csv` and normalize.
- Feature gating: Area Groups UI only when areas.csv has data; do not synthesize from DBC alone.
- Energy efficiency: preflight/skip-if-exists for LK ADTs, crosswalks, tile layers, and layers.json.
- BYOD: tooling must not include copyrighted game data anywhere in repo/binaries.

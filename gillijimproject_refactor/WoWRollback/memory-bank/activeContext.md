# Active Context (2025-11-10)

## Current Progress (concise)
- Main issue: LK ADT positions data must be generated and written into Alpha WDT (MAIN offsets + embedded ADT data).
- MPQ overlay precedence implemented and verified in CLI logs.
- Plain patch detection: `patch(.locale).MPQ` treated as numeric order 1 in `ArchiveLocator`.
- DBFilesClient precedence fix: prefer locale patch MPQs before root patch MPQs in `MpqArchiveSource` for `DBFilesClient/*` (corrects `Map.dbc`).
- Tee logging: `--log-dir` and `--log-file` mirror console to a file.
- Alpha WDT monolithic pack: build completes; placements diagnostics CSVs available; liquids path present (needs tuning).
 - Options consolidated; added `SkipM2`, `ConvertModelsToLegacy`, `ConvertWmosToLegacy`, `AssetsSourceRoot`.
 - Archive-mode model export preserves `.m2` extension on disk when source is `.m2`.
 - Added placements manifests: `m2_used.csv` and `wmo_used.csv` emitted next to outputs.
 - Converter scaffolding added (Warcraft.NET-backed stubs) with `conversion_manifest.csv` output when enabled.

## TODOs (concise)
- LK ADT positions → Alpha WDT:
  - Derive per‑tile presence/ordering from LK inputs; compute MAIN offsets table.
  - Embed per‑tile ADT payloads into Alpha WDT and update MHDR/MCIN alignments.
  - Validate offsets/sizes with `TryIsAlphaV18Wdt` and round‑trip read.
  - Emit `tiles_written.csv` with tile offsets and byte sizes.
- Diagnostics (placements/liquids):
  - Confirm MCNK subchunk offset base (chunk‑start) empirically; document rule in code and docs.
  - Fix/verify MCRF dumper offsets; add FourCC checks and payload size validation; dump first indices.
  - Instrument MH2O→MCLQ builder per‑chunk (layers, LVF, masks, heights/depths) and write `mclq_summary.csv`.
  - Validate placement transforms and `chunk_idx` mapping; add a tiny CSV of sample positions.
- Tests: `ArchiveLocator` ordering (incl. plain patch) and `MpqArchiveSource` DBC resolution.
- Logs: include plain‑patch counts; add one‑line DBC source when verbose.
- Verify: representative maps (CataclysmCTF, development) for overlay precedence and Map.dbc path.
- Textures: tileset resize option (256) with alpha preservation.
- Modularization: design `WoWRollback.AssetManagement` and keep CLI thin.
- Converters (real impl): Phase 1→3 as staged earlier.
- Agentic AI initiative:
  - Design pipeline using local Qwen3‑Coder (or hosted) to automate diagnostics and comparisons.
  - Scope: chunk offset auditing, MCRF/MH2O validations, coordinate transform checks, CSV diffing.
  - Integrate as optional CLI mode with artifact generation; respect offline mode and secrets.

- Round-trip verification harness (MAIN FOCUS):
  - Design: AlphaWDT → LK ADTs → AlphaWDT; compare original vs rebuilt.
  - CLI: implement `alpha-roundtrip-verify` to run both conversions, collect artifacts, and produce diffs.
  - Comparators: chunk-aware diff (`wdt_diff.csv`, `adt_diff.csv`, `mcnk_diff.csv`, `placements_diff.csv`, `liquids_diff.csv`).
  - Acceptance: define allowed differences (padding, byte-order/padding-only changes, normalized names) and severity levels.
  - Integration: tee logs, write diagnostics alongside diffs, exit non-zero on severity ≥ error.

## Current Focus
- Round-trip verification harness as the main focus (build, compare, and gate with acceptance criteria).
- Liquids/placements diagnostics (MH2O→MCLQ correctness; MCRF indices/offsets; minimize `dont_render`).
- Offset/coordinate semantics: lock down MCNK offset base and placement transforms; assert where possible.
- CLI-first pipeline; GUI acts as a runner with overlay + inline logs (no modals).
- Energy‑efficient preflight and stable CSV schemas.
- Agentic AI: prototype a local/hosted agent mode to run automated checks and write diagnostic CSVs.

## Decisions
- CLI-first with GUI runner and overlay; auto-tab navigation (Load→Build, Prepare→Layers); remove success/info modals.
- Keep CSV cache schema stable; normalize `tile_layers.csv` naming; GUI may fall back to `<map>_tile_layers.csv` and normalize.
- Feature gating: Area Groups UI only when areas.csv has data; do not synthesize from DBC alone.
- Energy efficiency: preflight/skip-if-exists for LK ADTs, crosswalks, tile layers, and layers.json.
 - Asset export policy: preserve original on-disk extensions (`.m2`, `.wmo`). Alpha WDT name-table normalization to `.mdx` does not rename exported files.
- Modularization: introduce `WoWRollback.AssetManagement` to own asset export, gating, and format down‑conversion; keep `.Cli` orchestration thin.
 - Add isolation toggles `SkipWmos` and `SkipM2` to triage placement-caused crashes; emit `m2_used.csv`, `wmo_used.csv`, `textures.csv` to aid diagnosis.
- BYOD: tooling must not include copyrighted game data anywhere in repo/binaries.

# Active Context

## Current Focus
- Standalone .NET 9 console tools: `AlphaWdtInspector` and `AlphaLkToAlphaStandalone`.
- As of 2025-11-15, day-to-day work is focused on `AlphaLkToAlphaStandalone` roundtrip validation, with `AlphaWdtInspector` remaining the general diagnostics toolbox.

- Diagnose and fix LK ADT → Alpha WDT packing issues:
  - MCRF destination re-bucketing (tile/chunk in Alpha coords).
  - MCLQ synthesis from MH2O using forward FourCC (fallback to reversed).
  - Hardened MCSE extraction.
- Provide deterministic CLI and CSV diagnostics.

## Immediate Next Steps
1. Scaffold `AlphaWdtInspector/` and extract minimal format helpers (FourCC scan, chunk readers, minimal builders).
2. Implement `export-lk --alpha-wdt <path> --out <dir>`.
3. Implement `pack-alpha --lk-root <dir> --out <alpha.wdt> --dest-rebucket --emit-mclq --scan-mcse` (+ `--no-coord-xform` for A/B).
4. Emit CSVs: `mcnk_sizes.csv`, `placements_mddf.csv`, `placements_modf.csv`, `mcrf_diag.csv`.
5. Verify MCLQ present and non-zero MCRF on busy tiles.
6. Add `tile-diff` and `dump-adt` for deeper analysis.

## Decisions & Constraints
- Self-contained tool; no WoWRollback dependencies.
- MH2O detection uses forward FourCC with reversed fallback.
- Destination-based MCRF re-bucketing; discard cross-tile mismatches if map/tile disagree.
- Stable CSV schemas across runs.

## Verification Plan
- Check `mcnk_sizes.csv` for `mclq_bytes > 0` where MH2O exists.
- Check `mcrf_diag.csv` for non-zero refs on known busy tiles.
- Use `tile-diff` to compare MCNK subchunks (offsets/sizes/hex) vs original Alpha WDT.

## Format Summary
**Alpha v18 Format:**
- MCLY, MCRF: Have FourCC+size headers
- MCVT, MCNR, MCSH, MCAL, MCSE, MCLQ: NO headers (raw data only)

**LK 3.3.5 Format:**
- ALL sub-chunks: Have FourCC+size headers

## Format Specifications (Verified)
- **Alpha WDT/ADT**: `memory-bank/specs/Alpha-0.5.3-Format.md` (Definitive spec, replaces all previous notes)

## Documentation Created
1. `docs/MCNK-SubChunk-Audit.md` - Complete format comparison
2. `docs/MCAL-Data-Orientation-Bug.md` - Analysis of visual corruption
3. `docs/DataVisualizationTool-Design.md` - Visualization tool architecture
4. `WoWDataPlot/README.md` - Tool usage guide
5. `WoWDataPlot/LAYER_WORKFLOW.md` - Layer-based filtering workflow
6. `WoWDataPlot/Models/LayerInfo.cs` - Layer data models
7. `WoWDataPlot/Extractors/AlphaPlacementExtractor.cs` - Reuses gillijimproject parsers

## Next Steps
1. **Test MCNK fixes** with real Alpha 0.5.3 WDT → in-game client
2. **Build and test WoWDataPlot** with real Kalidar data
3. **Run layer analysis** on Kalidar to create JSON metadata
4. **Generate tile-layer images** for visual inspection
5. **Integrate layer filtering** into WoWRollback.Cli for selective rollback
6. **Create web UI** for interactive layer selection (future)

## Update 2025-11-13
- Fixed `pack-alpha` MCIN backpatching: MCIN entry offsets are now RELATIVE to `MHDR.data` (Alpha v18 rule).
- Added `mtex-audit` command: validates per-tile MTEX vs MCLY texture indices. Current repack shows 0 violations and no OOB MCNKs.
- Structural checks (`wdt-sanity`) previously matched `main_firstMcnkRel`; to re-run after monolithic port.
- Placements parity tooling (`placements-sweep-diff`) in place; we only swap MDDF/MODF payload bytes from LK ADTs.
- Constraint: All targets are ≤ 3.3.5. Assets must be filename-based (MTEX/MDNM/MONM). Do NOT use fileDataIDs.
- Next focused task: Port WoWRollback’s monolithic writer verbatim as `pack-alpha-mono` (MVER→MPHD(128)→MAIN→MDNM→MONM), then overlay placements-only changes.

## Update 2025-11-14 – Standalone LK→Alpha Converter
- Decision: introduce a dedicated LK→Alpha WDT converter, separate from WoWRollback, that mirrors AlphaWDTAnalysisTool semantics.
- Converter will:
  - Consume LK ADTs (e.g., from `export-lk`) plus AlphaWDTAnalysisTool crosswalk CSVs.
  - Apply reverse crosswalks (3.3.5→0.5.3) for textures and placements.
  - Rebuild Alpha WDT (MPHD/MAIN/MDNM/MONM + embedded ADTs) with **no asset gating**.
  - Preserve all MDDF/MODF placements and WMO UniqueIDs to keep WMO-heavy tiles intact.

## Update 2025-12-05 – Retroporting Tools & Asset Conversion

### Project Context: The Alpha Client
- **Target**: WoW Alpha 0.5.3.3368 (December 8, 2003)
- **History**: "Friends & Family" build leaked to press, became historically significant
- **Anniversary**: December 8, 2025 marks 22 years since the build date
- **Goal**: Bring modern WoW content (up to 12.x) into the Alpha client

### Retroporting Pipeline Status
| Component | Modern → Alpha | Alpha → Modern | Status |
|-----------|----------------|----------------|--------|
| ADT/WDT | ✅ Working | ✅ Working | `pack-alpha-mono` |
| BLP Tilesets | ✅ BlpResizer | N/A | **COMPLETE** |
| WMO | ✅ v17→v14 | ✅ v14→v17 | Needs testing |
| M2/MDX | ✅ M2→MDX | ✅ MDX→M2 | Framework ready |

### BlpResizer Tool
- **Purpose**: Resize modern tilesets (512-4096px) to Alpha max (256×256)
- **CASC Support**: Extracts directly from WoW install
- **Tested**: 7956 tilesets from WoW 12.x, all outputs look correct
- **Next**: Test in Alpha client with converted 12.x terrain

### Future Experiments
1. **Test 12.x terrain** with resized tilesets in Alpha client
2. **Validate WMO converters** with real Alpha/Modern assets
3. **Test M2/MDX converters** with simple static models first
4. **Add animation support** to M2/MDX incrementally
5. **Revisit Q3 BSP conversion** — WMO is restructured Q3 BSP, almost worked

### Technical Notes
- Alpha uses BLP2 format (not BLP1 as initially assumed)
- WMO v14 is monolithic (single file), v17+ splits root + groups
- MDX format is WC3-like, M2 is completely different chunked format
- Existing `WmoV14Parser` and `WmoV14ToV17Converter` are battle-tested from Q3 experiments

## Update 2025-11-15 – AlphaLkToAlphaStandalone Roundtrip

### Current Focus
- `AlphaLkToAlphaStandalone` .NET 9 console app.
- Goal: validate 0.5.3 ↔ 3.3.5 AreaID and data preservation via a full roundtrip:
  - Alpha WDT/ADTs → LK ADTs (AlphaWdtAnalyzer + DBCTool.V2 crosswalks).
  - LK ADTs → Alpha (standalone pipeline), with diagnostics.

### Implemented
- `convert` command:
  - CLI: `convert --lk-root <lk_tree_or_client> --map <MapName> [--out <dir>]`.
  - Accepts either extracted LK trees (`<root>/World/Maps/<MapName>`) or 3.3.5 client installs (`<root>/Data/*.MPQ`) via MPQ mode.
  - Scans LK ADTs (filesystem or MPQ) to build a 64×64 occupancy grid.
  - Emits:
    - `tiles_summary.csv` (tile presence), `input_files.csv`, `run_summary.txt`.
    - Stub Alpha WDT (MVER=18 + MPHD + MAIN with synthetic offsets, **no Alpha ADTs yet**).

- `roundtrip` command:
  - CLI: `roundtrip --alpha-wdt <Alpha_WDT> --lk-client <3.3.5_root> [--out <dir>]`.
  - Auto-detects repo root, DBD dir, Alpha & LK DBC roots, and DBCTool.V2 output root from repo layout + `test_data`.
  - Alpha → LK export:
    - Uses `AlphaWdtAnalyzer.Core.AdtExportPipeline` with DBCTool.V2 crosswalk CSVs.
    - Writes crosswalk-patched LK ADTs into `out/session_YYYYMMDD_HHmmss/lk_export/World/Maps/<MapName>/`.
    - Emits AlphaWdtAnalyzer CSVs under `lk_export/csv/maps/<MapName>/`.
  - AreaID diagnostics:
    - Writes `areaid_roundtrip.csv` under `out/session_.../diagnostics/`, joining:
      - Original Alpha per-MCNK "area IDs" from the monolithic WDT.
      - Crosswalked LK `McnkHeader.AreaId` per MCNK in the exported LK ADTs.
  - LK → Alpha leg:
    - Reuses `RunConvertCore` + `LkToAlphaWriter` to:
      - Scan the exported LK ADTs.
      - Emit a stub Alpha WDT and `tiles_summary.csv` based on tile presence.

### Open Work / Next Steps
- Implement a **real LK→Alpha writer** in `AlphaLkToAlphaStandalone`:
  - Consume exported LK ADTs from `lk_export/World/Maps/<MapName>`.
  - Rebuild Alpha ADTs and a monolithic Alpha WDT with correct offsets pointing to real tile data.
  - Preserve MDDF/MODF placements and WMO UIDs, honoring crosswalked semantics.
- Extend diagnostics:
  - Derive mismatch-focused views from `areaid_roundtrip.csv` (e.g. only non-equal Alpha vs LK area IDs).
  - Optionally join map/zone metadata from DBCTool.V2 tables for human-friendly reporting.
- Integrate `--lk-client` more deeply if needed:
  - Use it as a source of DBC/asset verification in addition to `test_data` trees.

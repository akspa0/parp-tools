# Active Context (Next)

- Current Focus: Core parity in Next with the original gillijimproject-csharp: WDT Alpha→LK conversion and ADT LK roundtrip, with Liquids integration. PM4 enhancements deferred until core is shipped.
- Recent Changes:
  - Established Next project memory bank and seeded plan.
  - Prepared domain design for MH2O and MCLQ representations with explicit dimensions and LVF support.
  - Implemented domain models under `next/src/GillijimProject.Next.Core/Domain/Liquids/`.
  - Implemented `LiquidsConverter` with MH2O→MCLQ and MCLQ→MH2O algorithms.
  - Added CLI flags: `--liquids`, `--liquid-precedence`, `--liquid-id-map`, `--green-lava`.
  - Implemented `AlphaMclqExtractor` and wired it into the Alpha→LK pipeline via CLI `convert`; conversion now uses real MCLQ data.
  - Updated docs (`architecture.md`, `cli.md`) to describe extractor behavior and limitations.
  - Fixed Next.Core CS0246 errors by adding `using System.Collections.Generic;` to `Services/UniqueIdAnalyzer.cs` and `Transform/AlphaToLkConverter.cs`.
  - Fixed `AlphaMclqExtractorTests` builder compile issue by removing out-parameter shadowing: simplified `BuildMcnkHeader(...)` and preserved `OfsLiquid` (100) / `SizeLiquid` (104) header offsets for patching.
- Next Steps:
  - Implement CLI `wdt-convert`: Alpha WDT → LK WDT and ADT Alpha → ADT LK for present tiles; integrate Liquids flags and write logs/run-dir.
  - Implement CLI `adt-roundtrip`: read→write→re-read, integrity validation; optional SHA compare and MCLQ report.
  - Wire shared run-dir/log helper; preserve subfolder structures in outputs.
  - Add unit/integration tests and docs for these commands.
- Decisions:
  - Support LVF Case 0 and 2 initially; defer 1 and 3 with TODOs.
  - Precedence default: magma > slime > river > ocean (configurable).
  - LiquidType mapping provided via JSON overrides; provide sane defaults with TODO to validate against LiquidType.dbc.
  - AdtLk holds a fixed-size `Mh2oByChunk` array (256) for per-MCNK liquids.
  - Offset-origin heuristics for `ofsLiquid`: header end (dataStart+128), data start, chunk begin.
  - Ocean heights inferred from `heightMin` when depth-only layout is present.
  - Unknown low-nibble tile types normalized to `None`.
  - Per-chunk parsing errors are non-fatal and result in a null MCLQ entry for that chunk.

## Liquids Plan (Approved)

- Domain Models
  - `LiquidVertexFormat` enum: Case0=Height+Depth, Case1=Height+UV, Case2=DepthOnly, Case3=Height+UV+Depth.
  - `Mh2oAttributes`: 8x8 fishable/deep masks.
  - `Mh2oInstance`: LiquidTypeId, Lvf, min/max height, x/y offset, width/height (1..8), ExistsBitmap, typed vertex data arrays.
  - `Mh2oChunk`: List of instances + optional attributes.
  - `MclqLiquidType` enum and `MclqTileFlags` (ForcedSwim=0x40, Fatigue=0x80); `MclqData` with 9x9 height/depth and 8x8 tile type/flags.
  - `LiquidTypeMapping` and `LiquidsOptions` (enable, precedence, green-lava, mapping).
- Conversion Algorithms
  - `Mh2oToMclq`: compose tiles per precedence; synthesize 9x9 height/depth; map deep/fatigue flags; ignore UVs for now.
  - `MclqToMh2o`: partition 8x8 tiles into rectangles; build instances with ExistsBitmap; slice 9x9 to sub-rects; derive attributes.
- Integration
  - Alpha→LK: MCLQ→MH2O via `LiquidsConverter` and pass to LK writer (Warcraft.NET adapter later).
  - LK→Alpha: MH2O→MCLQ writer path; ensure MCLQ written last in MCNK; omit MH2O when empty.
- CLI Flags
  - `--liquids on|off` (default on), `--liquid-precedence`, `--liquid-id-map <json>`, `--green-lava`.
- Tests
  - Unit: single-type pools, mixed types, case 2, rectangle splitting.
  - Round-trip: both directions.
  - Integration: CLI `convert` with fixtures (skip-if-missing).
- Docs
  - Architecture: data flow and limitations.
  - CLI: flags, examples, mapping schema.
- Logging & Validation
  - Exists bitmap coverage, rectangle bounds, height clamps, attribute consistency; remove empty instances.

## WDL Parsing Progress

- Completed M1–M4:
  - Split `Wdl`/`WdlTile` into `next/src/GillijimProject.Next.Core/Domain/Wdl.cs`.
  - Hardened `AlphaReader.ParseWdl()` (MVER tolerance, overflow-safe bounds, padding after `MARE`/`MAHO`, reversed FourCC helpers).
  - Implemented `MAHO` holes parsing (incl. reversed `OHAM`), with default zero masks when missing.
  - Added tests: normal + reversed `MAHO`, missing-`MAHO` default, odd-size chunk padding tolerance, and fixture-based skip-if-missing.
- Docs updated: `next/docs/wdl-parsing-plan.md` statuses reflect M1–M4 completed.
- Next:
  - Implement `WdlWriter` and minimal CLI verbs (`wdl-dump`, `wdl-build`) with roundtrip tests.
  - Keep Noggit behavior for write ordering (MARE→MAHO per tile) and include `MAHO` even when zeroed.

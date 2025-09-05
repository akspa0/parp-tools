# Active Context (Next)

- Current Focus: Implement MH2O↔MCLQ liquid conversion for the Next pipeline with clean domain models, robust converters, CLI flags, tests, and docs.
- Recent Changes:
  - Established Next project memory bank and seeded plan.
  - Prepared domain design for MH2O and MCLQ representations with explicit dimensions and LVF support.
- Next Steps:
  1) Implement domain models under `next/src/GillijimProject.Next.Core/Domain/Liquids/`.
  2) Implement `LiquidsConverter` with MH2O→MCLQ and MCLQ→MH2O algorithms.
  3) Integrate into Alpha→LK (MCLQ→MH2O) and LK→Alpha (MH2O→MCLQ) paths.
  4) Add CLI flags: `--liquids`, `--liquid-precedence`, `--liquid-id-map`, `--green-lava`.
  5) Add unit and round-trip tests; update docs (`architecture.md`, `cli.md`).
  6) Add validations and logging (exists bitmasks, bounds, clamps, empty cleanup).
- Decisions:
  - Support LVF Case 0 and 2 initially; defer 1 and 3 with TODOs.
  - Precedence default: magma > slime > river > ocean (configurable).
  - LiquidType mapping provided via JSON overrides; provide sane defaults with TODO to validate against LiquidType.dbc.

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

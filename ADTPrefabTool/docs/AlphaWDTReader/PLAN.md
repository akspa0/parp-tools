# AlphaWDTReader — Plan

This document defines a minimal, pragmatic plan to add a separate tool/library capable of reading Alpha-era combined WDT/ADT files (e.g., 0.5.3) as described in `docs/wowdev.wiki/Alpha.md`. The goal is to expose a stable API and basic OBJ export for validation, while keeping the implementation small.

## Goals
- Parse Alpha combined files (WDT + embedded ADTs) and expose tiles/chunks with decoded geometry and basic metadata.
- Normalize data into a structure similar to 3.x readers to ease downstream usage.
- Provide a simple validation/export path (e.g., OBJ per tile) using real Alpha data.

## References
- Spec: `docs/wowdev.wiki/Alpha.md` (MAIN/NIAM tile table, MHDR, MCIN, MCNK + Alpha-specific subchunks).
- Contrast: `docs/wowdev.wiki/ADT_v18.md` (referenced by Alpha spec for later format details).
- Existing reader baseline: `lib/wow.tools.local` (naming, constants, math helpers, 3.x expectations).

## Scope
- In-scope (Phase 1):
  - Read MVER, MPHD, MAIN (4096 entries), MDNM, MONM, optional top-level MODF.
  - For tiles present: parse MHDR → MCIN/MTEX/MDDF/MODF, then MCNK and its subchunks.
  - Decode MCVT (absolute heights; Alpha ordering) and MCNR (Alpha ordering), capture layer/alpha/refs/liquid when present.
  - Synthesize per-chunk world origins (see Compatibility) and provide 3.x-like accessors.
- Out-of-scope (Phase 1):
  - Complex material blending logic, full alpha map semantics, conversions back to 3.x.
  - Full WMO/doodad instance feature parity beyond name/index resolution.

## Key Differences vs 3.x
- Single monolithic file contains WDT + all ADTs (Alpha `MAIN` provides 64×64 tile offsets).
- MCNK header is 128 bytes but field placements differ; subchunk offsets are relative to end of MCNK header.
- MCVT/MCNR have no chunk names/sizes and different ordering (outer 81 then inner 64). MCVT are absolute heights (not relative to header fields).
- MCRF directly indexes MDNM/MONM (no MMDX/MWMO indirection).
- Per-chunk origins may be absent and must be synthesized.

## Data Model (Phase 1)
- AlphaWorld
  - Tiles[64][64]: AlphaTile? (null when absent)
  - DoodadNames: string[] (MDNM), WmoNames: string[] (MONM)
  - Optional GlobalMODF
- AlphaTile
  - Present: bool; MHDR/MCIN/MTEX/MDDF/MODF
  - Chunks[16][16]: AlphaChunk
- AlphaChunk
  - Header fields (flags, indices, counts)
  - Heights[145], Normals[145]
  - Layers, Alpha, Refs, Liquid (when present)
  - OriginWS: Vector3 (synthesized)

## Parsing Strategy
1. Read MVER, MPHD.
2. Parse MAIN → 4096 SMAreaInfo entries; record tile offsets/sizes.
3. Read MDNM/MONM (0-terminated strings) and optional top-level MODF.
4. For each present tile:
   - Seek MHDR; read offsets to MCIN/MTEX/MDDF/MODF.
   - Parse MCIN → 256 entries; for each present chunk:
     - Read 128-byte MCNK header (Alpha layout); offsets are relative to end of header.
     - Decode subchunks:
       - MCVT (absolute heights; Alpha ordering outer81+inner64)
       - MCNR (Alpha ordering)
       - MCLY/MCRF/MCAL/MCSH/MCLQ/MCSE when present
     - Compute `OriginWS` (Compatibility layer) and keep data in normalized structures.

## Compatibility Layer (world-space origin synthesis)
- Use known tile origin and chunk grid spacing from current 3.x path (`lib/wow.tools.local`) to avoid magic numbers.
- Compute per-tile world origin from tile X/Y indices.
- Chunk origin:
  - `OriginWS = TileOriginWS + (chunkX * strideX, chunkY * strideY, 0)`
  - Heights come directly from absolute MCVT values; do not add relative offsets as in 3.x.

## Integration Plan
- New project: `src/AlphaWDTReader/` (class library).
- Optional helper: `src/AlphaWDTReader.Tool/` (simple CLI to dump tiles/chunks and export OBJ per tile for validation).
- Not wired into ADTPreFabTool initially; keep separate and import as needed later.

## Milestones
- M1: Project skeleton + file scanner → list MAIN entries and MDNM/MONM.
- M2: Parse MHDR/MCIN; iterate MCNK headers for one tile.
- M3: Decode MCVT/MCNR; export simple height-only OBJ per tile.
- M4: Decode remaining subchunks minimally; implement `OriginWS` synthesis.
- M5: Validate multiple tiles; summarize via CSV (counts/extents) and visual check.
- M6: Optional: expose a basic API for downstream tools; document examples.

## Validation (real data)
- Source: real Alpha combined files (0.5.3). Add parallel tiles from 0.6.0+ if available to compare gross shapes.
- Checks:
  - Tile presence vs MAIN table.
  - Geometry sanity (no NaNs, expected extents, continuity across neighbor chunks).
  - MDNM/MONM index resolution for MCRF.
- Outputs:
  - OBJ per tile for quick visual validation.
  - CSV summary with per-tile stats (present chunks, layer counts, AABB).

## Risks & Mitigations
- Alpha variants deviate from doc: build offset assertions and graceful skips.
- Subchunk offset math is fragile: centralize offset helpers; unit tests over byte fixtures.
- Performance: monolithic seeks; batch reads per tile to reduce thrash.

## Deliverables
- `src/AlphaWDTReader/` library with public API.
- `src/AlphaWDTReader.Tool/` (optional) for scans/exports.
- Documentation: this PLAN + brief README/usage.

## Interim Workaround (ADTPreFabTool)
- For 0.5.3 ADTs converted to ~3.x that lack chunk origins, synthesize chunk origins using the same grid math as 3.x based on tile/MCNK indices.
- Detection: missing/zero origin → compute via `ComputeChunkOrigin(tileX,tileY,chunkX,chunkY)` and proceed.

## Open Questions
- Where to store sample Alpha files for repeatable tests?
- Any preference on public API names to match `wow.tools.local` conventions?

## Optional: Alpha → 3.x Converter

Purpose: Convert Alpha combined maps into a standard 3.x layout (1 WDT + up to 4096 ADTs) to enable reuse of existing readers and tools.

- **Inputs**: Alpha combined file (per `docs/wowdev.wiki/Alpha.md`).
- **Outputs**:
  - `MapName.wdt` (3.x format), with MAIN table populated for present tiles
  - `MapName_XX_YY.adt` files (3.x layout) for each present tile

### References
- Historical code for flow/order: `lib/gillijimproject` (WotLK path only; ignore Cata). The 2012 tool works broadly but lacks MCNK position fix-ups; use as a structural reference, not as a dependency.
- Constants/math for world/tile/chunk transforms: `lib/wow.tools.local` (avoid magic numbers).

### Core Algorithm
1. **Read Alpha** using AlphaWDTReader (MAIN → MHDR → MCIN → MCNK...).
2. **Build resource tables**:
   - From MDNM/MONM, create unique sets → write MMDX/MWMO; generate MMID/MWID indices.
   - Remap MCRF entries to MMID/MWID indices.
3. **For each present tile (64×64 grid)**:
   - Emit 3.x ADT container sections (MTEX/MDDF/MODF/etc.).
   - For each MCNK (16×16):
     - Write 3.x MCNK header fields.
     - Compute chunk `OriginWS = TileOriginWS + (chunkX*strideX, chunkY*strideY, 0)`.
     - Translate MCVT ordering (Alpha outer81+inner64 → 3.x expected order); keep absolute heights.
     - Translate MCNR ordering similarly.
     - Copy/translate MCLY/MCAL/MCLQ/MCSH/MCRF if compatible; mark TODOs for edge cases.
   - Generate consistent MCIN offsets/sizes for all subchunks.
4. **Emit WDT** with standard MAIN entries and flags.

### MCNK Origin Reconstruction
- Use `lib/wow.tools.local` to derive:
  - Tile world origin from tile indices (X,Y)
  - Per-chunk stride in world units
- Implement a single helper `ComputeChunkOrigin(tileX,tileY,chunkX,chunkY)` to standardize math.

### Validation (real data)
- Roundtrip test: Alpha → 3.x → load via existing ADT pipeline → export OBJ → visual check for:
  - Seamless chunk/tile continuity
  - Correct extents and plausible heights
  - Correct doodad/WMO path mapping
- CSV summary comparing tile/chunk counts and AABB stats before/after conversion.

### Out-of-scope (initial)
- Full fidelity of advanced alpha blending/material nuances
- Non-critical subchunk variants not present in target Alpha data

### Notes
- Only reference the **WotLK** path in `lib/gillijimproject`; skip Cataclysm-specific codepaths.
- The historical tool’s missing MCNK position math is addressed by our origin reconstruction above.

## Output Modes (water formats)

- **Default (3.x)**
  - Output MH2O-only.
  - Optional WLW rescue targets MH2O (additive-only; respects existing water).
  - No MCLQ emitted.

- **Legacy (pre-3.x, optional)**
  - Output MCLQ-only.
  - Mutually exclusive with MH2O emission.
  - Two fill strategies:
    - `preserve`: copy original MCLQ verbatim if present; error if absent.
    - `from-wlw`: rasterize WLW surfaces to MCLQ height/flags.

- **Knobs (minimal)**
  - `--output-adt-version pre3x|3x` (default: `3x`).
  - When `pre3x`:
    - `--mclq-fill preserve|from-wlw` (default: `preserve`).
    - `--wlw-rescue-tri-threshold <N>` applies when `from-wlw`.

- **Notes**
  - WLQ/WLM remain optional and non-authoritative; only consider if strictly validated and explicitly enabled later.
  - Validation uses real test data from `test_data/`.

## Water Handling (MCLQ → MH2O)

- **Goal**
  - Parse Alpha `MCLQ` water data in the reader and expose it as a structured grid.
  - During Alpha → 3.x conversion, author `MH2O` layers/masks from `MCLQ`.

- **Reader (AlphaWDTReader)**
  - When `MCLQ` is present on a chunk, parse and expose:
    - Grid resolution (per-chunk water grid) and world scaling.
    - Occupancy mask/flags, per-cell height/level array(s), and type hints when available.
  - Provide `AlphaWater` on `AlphaChunk` without converting formats.

- **Converter (Alpha → 3.x)**
  - If `MCLQ` exists, emit `MH2O`:
    - Build occupancy from the `MCLQ` grid.
    - Identify connected components; create one MH2O layer per component.
    - Generate per-layer tile mask; write per-cell heights (or uniform if constant).
    - Set `minHeight/maxHeight/heightLevel` from included cells.
    - Map Alpha water flags (river/ocean/magma) when present; otherwise default.
  - Handle holes by clearing corresponding mask bits; keep logic simple and robust.

- **Math/Alignment**
  - Reuse terrain constants and chunk origin: `ComputeChunkOrigin(tileX,tileY,chunkX,chunkY)` from 3.x math in `lib/wow.tools.local` to align water with terrain.

- **References**
  - `lib/noggit3` — examples of MCLQ ↔ MH2O conversions.
  - `lib/MapUpconverter` — supplemental water handling.
  - `docs/wowdev.wiki/Alpha.md` and ADT v18 docs for related structures.

- **Validation (real data)**
  - Select Alpha samples containing `MCLQ`.
  - After conversion, load 3.x ADTs via existing pipeline; verify MH2O loads.
  - Visual OBJ sanity: optional simple quads or layer AABBs; CSV per chunk (hasWater, cells, min/max level).

## WL* (WLW/WLQ/WLM) Integration & Pitfalls

- **Reference (POC)**
  - `docs/AlphaWDTReader/snippets/3D_wlw2obj.py` — working WLW reader and OBJ/GLTF visualizer. WLM/WLQ unreliable in practice; treat them as optional.

- **Scope Policy (initial)**
  - Use WLW only for analysis/sidecars; do not mutate ADT content based on WL* yet.
  - WLQ/WLM are considered subsets (generic slime/lava) but have structural uncertainties — parse conservatively and skip on mismatch.

- **Structural Uncertainty**
  - wowdev docs for WLW/WLQ/WLM may be incomplete; block sizes and field types can vary.
  - Enforce header checks (magic/version/unk06) and strict bounds before reading blocks.
  - If a computed read would exceed file length, skip the file with a clear note.

- **Coordinate & Type Mismatch**
  - Docs suggest integer vectors (`C3Vectori`), but some dumps appear as floats; the POC currently unpacks floats for vertices/coords.
  - Risk: misinterpreted vertex data yields distorted meshes/AABBs.
  - Mitigation: probe both int and float interpretations behind a small detector (range checks); prefer the one that yields sane AABBs.

- **Liquid Types**
  - WL* liquid types (ocean/river/magma/slime/fast/etc.) do not directly map to MH2O flags.
  - For now, treat as annotations only. Any mapping to MH2O flags must be conservative and documented.

- **Validation Strategy**
  - Run the POC over real WLW files to confirm basic mesh integrity (no NaNs, plausible extents).
  - Emit per-block bbox CSV for overlap checks with ADT tiles/chunks.
  - If WLQ is present next to a WLW of the same basename, compare block counts and bboxes; otherwise ignore WLQ.

- **Output Discipline**
  - Sidecars only (CSV/NDJSON/OBJ for visualization). No ADT writes based on WL* at this stage.
  - Keep paths sanitized under `project_output/.../` alongside existing reports.

- **Next Steps (optional)**
  - If WLQ/WLM support is required later, add a strict header+size validator and a dual int/float decode attempt with guards.
  - Add a tiny summary to `water_policy.md` on how WL* annotations could inform MH2O defaults once validated.

## WLW-based Liquid Rescue (optional)

- **Purpose**
  - Use WLW (source liquid definitions) to reconstruct missing/broken water in older data (esp. 0.5.3). These files contain raw 3D coordinates of welded planes that represent water surfaces placed on terrain.

- **Availability**
  - 0.5.3: WLW only.
  - 0.6.0–3.0: WLW + WLQ/WLM may exist; WLW remains authoritative.

- **Knobs (minimal)**
  - `--rescue-liquids-from-wlw` (default: off).
  - `--wlw-rescue-tri-threshold <N>` (ignore tiny/degenerate faces; default small value).

- **Algorithm (WLW-first)**
  1) Read WLW blocks → generate triangles (as in `docs/AlphaWDTReader/snippets/3D_wlw2obj.py`).
  2) For each ADT tile: transform triangles to world space; clip by tile/MCNK bounds.
  3) Rasterize to MH2O grid per chunk:
     - Occupancy = triangle coverage; height per cell = robust sample/median of hits.
     - Connected components → one MH2O layer per component; compute bbox, compress masks.
     - Set liquid type when known; else conservative default.
  4) Additive-only: fill gaps where MH2O is missing/broken; never overwrite valid existing water.

- **WLM/WLQ**
  - Treated as optional supplements in 0.6.0–3.0. Use only if headers/sizes validate and WLW corroborates; otherwise skip.

- **Validation**
  - Use real files from `test_data/` to compare masks pre/post; emit sidecar NDJSON/CSV/OBJ; open in noggit3 to confirm surfaces. 

- **Pitfalls**
  - Header/size variance: strict magic/version/unk06 checks and bounds; skip on mismatch.
  - Coordinates may appear int or float: guarded decode path; pick interpretation with plausible AABBs.
  - Types not always reliable: default conservatively.

## Prerequisite (Alpha MPQs)

- Alpha (0.5.3/0.5.5) map MPQs often lack internal filenames and cannot be read directly by common MPQ libs.
- Workflow: Use Zezula's MPQ Editor to extract the first, largest file; rename it to the map name (e.g., `Azeroth`) and place it under `test_data/0.5.3/World/Maps/<MapName>/`.
- See `docs/USER_GUIDE.md` for the step-by-step walkthrough and quick commands.

## References & Code Hotspots

- **wow.tools.local (3.x baseline)**
  - `lib/wow.tools.local/WoWFormatLib/WoWFormatLib/FileReaders/ADTReader.cs` — MCNK subchunks; origin math reference; `MCLQ` areas still TODO.
  - `lib/wow.tools.local/WoWFormatLib/WoWFormatLib/Structs/ADT.Struct.cs` — MCNK/MCIN/MH2O-related structs.

- **noggit3 (water conversion and map geometry)**
  - `lib/noggit3/src/noggit/liquid_layer.cpp|hpp`
  - `lib/noggit3/src/noggit/liquid_chunk.cpp|hpp`
  - `lib/noggit3/src/noggit/liquid_tile.cpp`
  - `lib/noggit3/src/noggit/MapTile.cpp`, `MapChunk.cpp`, `MapTile.h`, `MapChunk.h`, `MapHeaders.h`
  - Notes: implements MCLQ ↔ MH2O conversions and robust handling of oddly converted files; saving back requires explicit options.

- **gillijimproject (WotLK writer patterns)**
  - `lib/gillijimproject/wowfiles/lichking/AdtLk.cpp|h`, `McnkLk.cpp|h`
  - `lib/gillijimproject/wowfiles/alpha/McnkAlpha.cpp`
  - `lib/gillijimproject/wowfiles/Mh2o.cpp|h`, `ChunkHeaders.h`, `Mhdr.h`
  - Notes: good section ordering and MH2O emission patterns; replace its origin math with our synthesis.

- **MapUpconverter / Warcraft.NET (C# MH2O models)**
  - `lib/MapUpconverter/Warcraft.NET/Warcraft.NET/Files/ADT/Chunks/MH2O.cs`
  - `lib/MapUpconverter/Warcraft.NET/Warcraft.NET/Files/ADT/Chunks/MH2OOld.cs`
  - `lib/MapUpconverter/Warcraft.NET/Warcraft.NET/Files/ADT/Entries/MH2OHeader.cs`, `MH2OInstance.cs`, `MH2OInstanceVertexData.cs`
  - `lib/MapUpconverter/Warcraft.NET/Warcraft.NET/Files/ADT/Terrain/Wotlk/Terrain.cs`

- **wow.export (JS reference)**
  - `lib/wow.export/src/js/3D/loaders/ADTLoader.js` — alternative parsing viewpoint.

- **Docs**
  - `docs/wowdev.wiki/Alpha.md`, `docs/wowdev.wiki/ADT_v18.md` — ground truth for structures.

## Review Findings (skim summary)

- **Chunk origins**: Some 0.5.3→3.x ADTs lack MCNK origins; synthesize using 3.x constants (reuse math from `wow.tools.local`).
- **Height/normal ordering**: Alpha `MCVT` absolute heights and Alpha-specific vertex order; convert/reorder when emitting 3.x.
- **Water conversion**: Use noggit3 logic conceptually (components → layers/masks; heights). Author in C# using MapUpconverter MH2O structs.
- **Reader gaps**: Existing `ADTReader.cs` has TODOs for `MCLQ`; AlphaWDTReader will parse Alpha `MCLQ` independently.
- **Writer order**: Follow `gillijimproject` WotLK order; align to 4 bytes.
- **Flags default**: Unknown → generic water.

## Validation (Real Data, Minimal Knobs)

- **CSV per chunk**: id, mcvt min/max, mcnr present, hasWater, waterCells, waterMin/Max/Mean.
- **OBJ overlays**: optional quads/AABBs for water to check alignment.
- **Roundtrip**: Alpha→3.x loads without exceptions; MH2O verified; no MCLQ present.

## One-Shot Implementation Sequence

1. Reader: MAIN, MDNM/MONM, MHDR/MCIN, MCNK headers.
2. Terrain: MCVT/MCNR decode (Alpha order → normalized), origin synthesis.
3. Water read: MCLQ → `AlphaWater`.
4. Writer skeleton: WotLK order, constants, offset computation.
5. Water convert: MCLQ → MH2O (components→layers, masks, heights), omit MCLQ.
6. Resources: names/placements minimal viable.
7. Validation: offset/size checks, no MCLQ, CSV stats.
8. Run on real samples; inspect OBJ/CSV.

## Snippet Pack (docs/AlphaWDTReader/snippets/)

- `chunk_origin_math.cs` — canonical 3.x world/tile/chunk origin math.
- `vertex_ordering_maps.md` — Alpha↔3.x index maps (145 vertices).
- `mclq_to_mh2o.cs` — conversion routine (components→layers/masks/heights).
- `mh2o_struct_layout.cs` — annotated C# structs (MH2OHeader/Instance/Vertex).
- `adt_write_order.md` — WotLK write order, offsets, alignment rules.
- `connected_components.cs` — grid labeling with bbox + mask.
- `offset_safety_checklist.md` — post-write offset/size assertions.
- `water_policy.md` — MH2O-only policy & validation.

## Water Write Policy (MH2O-only)

- **Policy**: Output ADTs must use MH2O exclusively. If Alpha `MCLQ` exists, always convert to MH2O; do not emit legacy `MCLQ`.
- **Rationale**: Tools like noggit3 can load mixed states depending on save options, risking ADTs with both `MCLQ` and `MH2O`. We forbid mixed states for consistency and future-proofing.
- **Implementation**:
  - Convert `MCLQ` → MH2O during Alpha→3.x conversion.
  - Zero/omit any `MCLQ` subchunk on write (ensure MHDR/MCNK offset fields reflect absence).
  - Validate post-write that no `MCLQ` remains (sanity check pass over chunk headers).
- **Config**: No toggle initially (keep simple). If a toggle is added later, default remains MH2O-only.

## Additional Pitfalls

- **Chunk origins & coordinates**
  - Synthesize MCNK origin using 3.x constants; mirror `wow.tools.local` math.
  - Confirm per-subchunk offset base (MCNK start vs header end) before seeking.
  - Keep axis orientation consistent with existing 3.x path to avoid mirroring.

- **Vertex/grid ordering**
  - Alpha `MCVT` (145) order differs from 3.x; remap outer81+inner64.
  - `MCNR` normals may be signed bytes with scale; do not assume unit length.

- **Water traps (MCLQ ↔ MH2O)**
  - Forbid mixed states; MH2O-only output.
  - Multiple regions per chunk → one MH2O layer per connected component.
  - Holes respected via masks; minimize mask bbox per layer.
  - Handle uniform vs varying heights; set min/max/level appropriately.
  - Use WotLK MH2O layout; avoid Cataclysm divergences.

- **Flags & types**
  - Map Alpha water types conservatively; default to generic when unknown.
  - Don’t let terrain holes/flags incorrectly cull water.

- **Resources (names/placements)**
  - Deduplicate name tables (MMDX/MWMO) before referencing.
  - Alpha global MODF vs per-ADT: document chosen handling.

- **IO & integrity**
  - 4-byte alignment; recompute all offsets post-assembly.
  - Offsets either zero or within parent range; sizes match payload.
  - Final scan to ensure no `MCLQ` remains.

- **Versioning & scope**
  - Target 3.x/WotLK only. Minimal CLI knobs.

- **Performance**
  - Stream per tile/chunk; reuse buffers.

## Decisions & Policies

- **Water write policy**: MH2O-only; zero any MCLQ offsets on write; validate none remain.
- **Chunk origin policy**: Shared `ComputeChunkOrigin(tileX,tileY,chunkX,chunkY)` helper reused by reader/converter.
- **Height policy**: Preserve absolute heights; reorder only.
- **Layer policy**: Layers = connected components; no cross-merge.
- **Writer order**: Follow `gillijimproject` WotLK order; align to 4 bytes.
- **Flags default**: Unknown → generic water.

## Validation (Real Data, Minimal Knobs)

- **CSV per chunk**: id, mcvt min/max, mcnr present, hasWater, waterCells, waterMin/Max/Mean.
- **OBJ overlays**: optional quads/AABBs for water to check alignment.
- **Roundtrip**: Alpha→3.x loads without exceptions; MH2O verified; no MCLQ present.

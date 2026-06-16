# Implementation Plan: 064 — Blank Map Generation + Relational ADT Understanding

**Branch**: `v0.5.0-dev` | **Date**: 2026-06-16 | **Spec**: [spec.md](spec.md)

## Summary

Generate valid blank ADT/WDT/WDL files that load in the viewer, document the ADT format as a relational schema, and establish the foundation for Zarr-backed ADT data round-tripping.

## Existing Infrastructure

| Component | Status | Purpose |
|-----------|--------|---------|
| `LkAdtWriter` | Working | Writes full LK ADT from `LkAdtData` |
| `LkAdtReader` | Working | Reads LK split ADT into `LkAdtData` |
| `LkAdtData` | Working | Domain model for LK ADT |
| `LkWdtWriter` | Working | Writes LK WDT with MAIN mask |
| `WdlWriter` | Working | Writes WDL with flat heights |
| `AlphaWdtWriter` | Frozen (Rule 10) | Writes Alpha WDT — use without modification |
| `AlphaToLkConverter` | Working | Converts Alpha → LK (proves round-trip pipeline) |
| `LkToAlphaConverter` | Working | Converts LK → Alpha |
| `AdtTerrainWriter` | Working | Patches MCVT+MCNR in existing ADT |
| `AdtPlacementWriter` | Working | Patches MDDF+MODF in existing ADT |

**Key gap**: No `BlankAdtFactory` or equivalent. We can write `LkAdtData` but have no standard way to construct one with valid blank defaults. The round-trip tests in `LkToAlphaRoundTripTests.cs` show minimal `LkAdtData` construction, but not a reusable factory.

## Phase 1: Blank LK Map Generation

### P1-A: BlankAdtFactory

**File**: `wow-viewer/src/core/WowViewer.Core/Maps/BlankAdtFactory.cs`

Static class that produces `LkAdtData` with:
- Map name, tile X/Y from parameters
- Empty texture names list (`[]`)
- Empty model names list (`[]`)
- Empty world model names list (`[]`)
- Empty MDDF/MODF entries (`[]`)
- 256 MCNK chunks with flat terrain defaults:
  - IndexX/Y from loop (0..15)
  - Flags = 0
  - AreaId = 0
  - NLayers = 0 (or 1 with a single default "grass" texture?)
  - HoleMask = 0
  - BaseHeight = 0.0f (or standard default like 0.0)
  - Heights = 145 floats, all set to BaseHeight (flat)
  - Normals = 448 bytes, all pointing up (0,0,1 normalized to ADT encoding)
  - No shadow map, no alpha map, no liquid, no MCCV/MCLV
  - PosX/PosY from tile coordinate + chunk index math
- MhdrFlags = 0 (no MH2O, no MFBO)
- MfboFlightBounds = null

**Key question**: Does the viewer require at least one texture layer per MCNK? If so, NLayers=1 with a default texture and flat alpha. Need to test.

**Validation**: `LkAdtWriter.Build(blankData)` produces a byte array. Write to file. Run `map inspect`. Verify structure.

### P1-B: BlankWdtFactory + CLI Command

**File**: `wow-viewer/src/core/WowViewer.Core/Maps/BlankWdtFactory.cs` (or inline in CLI)

Construct WDT data with:
- MPHD flags for LK format
- MAIN grid with specified tile(s) flagged

**CLI command**: `map generate-blank-map --map-name <name> --tile-x <n> --tile-y <n> --output-dir <dir>`

Produces: WDT + ADT + WDL files for a single tile.

### P1-C: Viewer Validation

Load the generated blank map in the viewer. Confirm:
- No assertion errors
- No null reference exceptions
- Flat terrain renders (visible grid)
- No missing chunk warnings in log

**This is the gate.** If the viewer can't load the blank ADT, Phase 1 is not done.

### P1-D: Alpha Blank WDT (Optional)

If the user reopens AlphaWdtWriter (Rule 10), generate blank Alpha WDT using existing `AlphaWdtWriter.Build()` with blank `AlphaTileData`. Otherwise, skip — the existing writer already works for this when given valid input data.

## Phase 2: ADT Relational Schema + Round-Trip Proof

### P2-A: Relational Schema Document

**File**: `wow-viewer/docs/architecture/adt-relational-schema.md`

Document every LK ADT chunk as a database table:

```
TABLE MHDR (offset 0x00 from file start)
  col: flags          UINT32   offset 0x00
  col: ofsMcin        UINT32   offset 0x04   FK→MCIN
  col: nMtex          UINT32   offset 0x08   count(MTEX entries)
  col: nMmdx          UINT32   offset 0x0C   count(MMDX entries)
  col: ofsMmdx        UINT32   offset 0x10   FK→MMDX
  col: ofsMmid        UINT32   offset 0x14   FK→MMID
  col: nMwmo          UINT32   offset 0x18   count(MWMO entries)
  col: ofsMwmo        UINT32   offset 0x1C   FK→MWMO
  ...

TABLE MCNK (256 rows, offset from MCIN)
  col: flags          UINT32   offset 0x00
  col: indexX         INT32    offset 0x04
  col: indexY         INT32    offset 0x08
  col: nLayers        UINT32   offset 0x0C   count(MCLY entries in this chunk)
  col: nDoodadRefs    UINT32   offset 0x10   count(MCRF doodad refs)
  col: ofsMcal        UINT32   offset 0x24   FK→MCAL within this chunk
  col: nWmoRefs       UINT32   offset 0x38   count(MCRF WMO refs)
  ...sub-chunks: MCVT, MCNR, MCLY, MCRF, MCAL, MCSH, MCLQ, MCCV, MCLV
```

Cross-reference map:
- MCIN[i].offset → MCNK[i] file offset
- MCLY[j].textureId → MTEX[MCLY[j].textureId]
- MCRF doodad refs → MDDF indices → MMDX + MMID
- MCRF WMO refs → MODF indices → MWMO + MWID
- MHDR.ofsMcin → MCIN offset
- MHDR.ofsMtex → MTEX offset (etc.)

### P2-B: Lossless Round-Trip Proof

Read a real LK ADT via `LkAdtReader`, write it back via `LkAdtWriter`, compare bytes.

If not byte-identical, identify every difference and fix either the reader or writer until round-trip is lossless.

This is a critical proof point: if we can't round-trip a real file, our writer has bugs.

## Phase 3: Zarr ADT Datastore (Stretch)

### P3-A: Zarr Schema for ADT Tables

Define Zarr group structure:
- `/metadata` — map name, tile coords, format version
- `/textures` — string array of texture paths (MTEX)
- `/models` — string array of M2 paths (MMDX)
- `/world_models` — string array of WMO paths (MWMO)
- `/placements_m2` — structured array (MDDF: nameId, uid, pos, rot, scale)
- `/placements_wmo` — structured array (MODF: nameId, uid, pos, rot, bounds, flags, etc.)
- `/chunks/{i}` — per-chunk group with MCVT, MCNR, MCLY, MCAL, MCRF, MH2O sub-arrays

### P3-B: Read ADT → Zarr, Modify, Write Back

Read a real ADT into the Zarr store. Change a texture name. Write back out. Confirm only the affected bytes changed.

## Complexity Tracking

| Principle | Status | Notes |
|-----------|--------|-------|
| Repo Independence | ✅ | All code under `wow-viewer/` |
| Library-First | ✅ | Factory in Core, CLI is thin wrapper |
| Real-Data Validation | ✅ | Must load in viewer + round-trip real files |
| One Phase at a Time | ✅ | P1 must validate before P2 starts |
| AlphaWdtWriter Frozen | ✅ | P1-D skips unless user reopens |
| No ADT patching | ✅ | Generate-from-scratch only |
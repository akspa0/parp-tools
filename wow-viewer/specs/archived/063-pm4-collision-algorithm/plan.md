# Implementation Plan: 063 — PM4 Collision Algorithm Reverse Engineering

**Date**: 2026-06-15 | **Spec**: [spec.md](spec.md)

## Summary

Add a `pm4 dump-collision` command to the inspect tool that dumps WMO group collision data (MOPY/MOVT) alongside PM4 per-Object surface data (MSUR/MSVT/MSCN/MPRL) for a known tile+OID pair, enabling side-by-side comparison to reverse the collision simplification algorithm.

## Technical Context

**Language**: C# (.NET 10) — extend existing `WowViewer.Tool.Inspect`

**Key libraries already exist**:
- `WowViewer.Core.PM4.Services.Pm4ResearchReader` — reads PM4 chunks into `Pm4KnownChunkSet`
- `WowViewer.Core.IO.Files.ArchiveVirtualFileReader` — reads files from staged client MPQs
- `WowViewer.Core.IO.Wmo.WmoSummaryReader` — reads WMO root file
- `WowViewer.Core.IO.Wmo.WmoGroupInfoSummaryReader` — reads WMO group entry data from root
- `WowViewer.Core.IO.Wmo.WmoGroupReader` — NEEDS CHECK: reads WMO group files with MOPY/MOVT
- `WowViewer.Core.IO.Maps.AdtPlacementReader` — reads _obj0.adt placements

**New code needed**: A WMO group collision reader that extracts MOPY (triangle flags + vertex refs) and MOVT (vertices) from .wmo group files, computes per-triangle normals.

## Constitution Check

| Principle | Status | Notes |
|-----------|--------|-------|
| Repo Independence | ✅ | All new code in `wow-viewer/tools/inspect/` |
| Library-First | ✅ | Add WMO group collision reader to `Core.IO.Wmo` |
| Real-Data Validation | ✅ | Uses staged client 3.3.5 + dev tile PM4s |
| No H:\CLIENTS | ✅ | Uses `output/tmp/wowarchive-clients/` |

## Implementation Phases

### Phase 1: WMO Group Collision Reader

**Goal**: Extract per-group collision triangles from WMO group files.

Write `WmoGroupCollisionReader` in `WowViewer.Core.IO.Wmo` that:
- Reads a WMO group file (e.g., `foo_000.wmo`)
- Extracts MOPY (collision triangle entries, 8 bytes each)
- Extracts MOVT (collision vertices, float32 xyz)
- Extracts MOVI (triangle indices, uint16)
- Computes per-triangle normals from MOVT + MOVI

**Validation**: Dump for DUSKWOODABANDONED_BARN group 0 — verify triangle count matches known data.

### Phase 2: CLI Command `pm4 dump-collision`

**Goal**: CLI command that takes `--tile <x_y>` and `--oid <objectId>`, dumps both WMO collision and PM4 surfaces.

Uses the existing OID→placement correlation logic:
1. Read PM4 file for the tile
2. Find all MSUR entries for the given OID
3. Read MSVT, MSCN, MSVI, MPRL chunks
4. For each surface: dump normal, plane distance, MSCN position, vertices
5. Read _obj0.adt for the tile
6. Find the closest MODF placement to the PM4 segment's MSCN center
7. Read the WMO file from the staged client
8. For each WMO group: dump MOPY triangle normals + MOVT vertices
9. Also read the WMO group file directly for full collision data
10. Print both datasets with alignment

### Phase 3: Analysis Script

**Goal**: Process all known OID→WMO pairs and produce a research report.

Run the CLI on all 13 OIDs on tile 24_35 and produce a summary showing the mapping between WMO triangle counts and PM4 surface counts for each OID.

## Project Structure

### New files
```text
wow-viewer/src/core/WowViewer.Core.IO/Wmo/
├── WmoGroupCollisionReader.cs     # Phase 1: reads MOPY/MOVT/MOVI from group files
├── WmoGroupCollisionModels.cs     # Models for MOPY entry, MOVT entry

wow-viewer/tools/inspect/WowViewer.Tool.Inspect/
├── ...Program.cs additions...     # Phase 2: pm4 dump-collision command
```

### Existing files changed
```text
wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs
  — Add "dump-collision" command handler
```

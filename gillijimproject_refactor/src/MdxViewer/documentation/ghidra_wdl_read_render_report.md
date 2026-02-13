# Ghidra Report: WDL File Read + Render Pipeline (WoW Alpha 0.5.3 / Build 3368)

Date: 2026-02-13  
Binary: `WoWClient.exe` (with PDB)  
Focus: End-to-end WDL (`.wdl`) handling — file read, data structures, culling, rendering.

---

## Executive Summary

WDL is handled as an optional low-detail horizon terrain layer:

- Loaded by `CMap::LoadWdl` during map load.
- Parsed as `MVER` -> `MAOF` -> per-cell `MARE` chunks.
- Stored into `CMap::areaLowTable` (4096-cell table, 64x64).
- Culled by `CWorldScene::CullHorizon`.
- Rendered by `CWorldScene::RenderHorizon` via `CMap::RenderAreaLow` and a dynamic GX buffer callback.
- If WDL is missing or disabled, normal terrain path still renders.

---

## 1) Load Call Graph (Map -> WDL)

1. `NewWorldHandler` -> `LoadNewWorld` (`0x00401BF0`)
2. `LoadNewWorld` -> `CWorld::LoadMap` (`0x00663100`)
3. `CWorld::LoadMap` -> `CMap::Load` (`0x0067F7F0`)
4. `CMap::Load` -> `CMap::LoadWdl` (`0x0067FA20`)
5. `CMap::Load` then continues with WDT loading (`LoadWdt` at `0x0067FDE0`)

---

## 2) WDL Loader Function and Behavior

## `CMap::LoadWdl` (`0x0067FA20`)

### Path construction
- Uses format string `%s\\%s.wdl`.

### I/O wrappers used
- `SFile::Open` (`0x0063CB70`)
- `SFile::Read` (`0x0063D6D0`)
- `SFile::SetFilePointer` (`0x0063DD20`)
- `SFile::Close` (`0x0063D9D0`)

### Parse/validate sequence
1. Read chunk header -> must be `MVER` (`0x4D564552`).
2. Read version (uint32) -> must be `0x12`.
3. Read next chunk header -> must be `MAOF` (`0x4D414F46`).
4. Read fixed `0x4000` bytes of MAOF offsets (4096 x uint32).
5. For each nonzero MAOF entry:
   - Seek absolute file offset.
   - Read chunk header -> must be `MARE` (`0x4D415245`).
   - Allocate one low-area object (`0x8C8` bytes).
   - Read `0x442` bytes (`545` x int16 heights).
   - Convert heights to float and compute per-area bounds.
6. Close file.

### Notes on validation style
- Token/version mismatches call `_SErrDisplayError*` logging/assert helpers.
- Decompilation shows reads continue in path; behavior can be assert/log-mode dependent.

---

## 3) Inferred WDL Binary Layout (Client-Expected)

## File-level
- `ChunkHeader`: `MVER`, `size`
- `uint32 version` = `0x12`
- `ChunkHeader`: `MAOF`, `size`
- `uint32 offsets[64*64]` = `4096` entries (`0x4000` bytes)
  - `0` means no low-detail area for that cell.
  - nonzero is absolute file offset.

## Per nonzero offset target
- `ChunkHeader`: `MARE`, `size`
- `int16 heights[545]` (`0x442` bytes)
  - topology is `17x17` outer (`289`) + `16x16` inner (`256`) = `545` samples.

---

## 4) Runtime Data Mapping

- MAOF index maps into `CMap::areaLowTable` (4096 entries).
- Each populated entry stores:
  - float-converted height samples (545)
  - derived bounds (used by horizon culling)

Associated globals/tables observed in decompilation:
- MAOF read table at `0x00E64E90`
- low-area table base at `0x00E5CB98`

---

## 5) Render Call Graph (WDL)

1. `CWorldScene::Render` (`0x0066A9D0`)
2. `CullSortTable` (`0x0066CA50`) -> `CullHorizon` (`0x0066CAD0`)
   - Builds visible low-area linked list (`visAreaLowList`) from `areaLowTable`.
3. `RenderHorizon` (`0x0066DAF0`)
   - Iterates visible low areas.
   - Calls `CMap::RenderAreaLow` (`0x0069F360`) when low-detail mode is enabled.
4. `CMap::RenderAreaLow`
   - Configures render state and dynamic GX buffer.
   - Uses callback `GxBufDynLowDetailCallback` (`0x0069F3F0`).
5. Callback fills geometry:
   - `CreateAreaLowDetailVertices` (`0x0069F440`) -> 545 verts
   - `CreateAreaLowDetailIndices` (`0x0069F5C0`) -> `0xC00` indices
6. Draw submit via GX (`GxBufRender`).

---

## 6) Low-detail Mesh Topology (as rendered)

Per low-area cell:
- Vertices: `545`
- Indices: `0xC00` (`3072`)
- Cell tessellation: 16x16 tiles, 4 triangles per tile (horizon/low-detail layout).

Vertex path includes packed color and expected dynamic-buffer stride consistent with low-detail callback output.

---

## 7) Culling/Enable Behavior

- `CullHorizon` performs frustum/clip checks and visibility filtering on `areaLowTable` entries.
- Rendering low detail is gated by world enable bit `0x04000000`.
  - Toggle path observed: `CWorld::ConsoleCommand_ShowLowDetail` (`0x006660C0`).
- If WDL is absent, no low areas are queued; horizon low-detail pass contributes nothing.
- Main chunk terrain rendering remains active independently.

---

## 8) Unload/Cleanup

- `CWorld::UnloadMap` (`0x00663150`) frees `areaLowTable` entries and related map allocations.
- Low-detail buffers/structures are reset with map unload lifecycle.

---

## 9) Practical Implications for External Viewers

To emulate client behavior:

1. Treat WDL as optional low-detail overlay.
2. Parse only the chunks this client consumes: `MVER`, `MAOF`, `MARE`.
3. Require version `0x12` for this path.
4. Use 64x64 MAOF absolute-offset table.
5. Decode each `MARE` as 545 signed 16-bit heights -> float.
6. Build per-cell bounds early for horizon culling parity.
7. Render using 545-vertex / 3072-index topology per visible low-area cell.
8. Keep a fallback path: normal terrain rendering must work without WDL.

---

## 10) Function Table (Quick Reference)

- `0x0067FA20` `CMap::LoadWdl`
- `0x0067F7F0` `CMap::Load`
- `0x00663100` `CWorld::LoadMap`
- `0x00663150` `CWorld::UnloadMap`
- `0x0066CAD0` `CWorldScene::CullHorizon`
- `0x0066DAF0` `CWorldScene::RenderHorizon`
- `0x0069F360` `CMap::RenderAreaLow`
- `0x0069F3F0` `CMap::GxBufDynLowDetailCallback`
- `0x0069F440` `CreateAreaLowDetailVertices`
- `0x0069F5C0` `CreateAreaLowDetailIndices`
- `0x006660C0` `CWorld::ConsoleCommand_ShowLowDetail`

---

## 11) Uncertainty Notes

- Some decompiled struct member names in low-area/bounds structs are type-noisy due optimization and symbol overlap.
- Control-flow around error handlers may differ between assert/log runtime modes.
- Despite that, the chunk order, fixed reads, table dimensions, and render call chain are strongly anchored by function behavior and address-level tracing.

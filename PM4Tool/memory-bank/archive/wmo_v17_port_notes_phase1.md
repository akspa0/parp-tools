# WMO v17 Port – Phase 1 Notes (from wow.export JS)

_Date: 2025-07-13_

This document captures the **behavioural insights** extracted while reading `wow.export` sources. It is intended as an implementation guide for subsequent phases.

---

## 1. Generic Helpers (`LoaderGenerics.js`)

| Function | Purpose | Porting Note |
|----------|---------|--------------|
| `ReadStringBlock(buf, size)` | Reads a null-terminated string table and returns an object mapping offset→string. | We already mimic this in `MOTXParser` for textures. Create a shared helper `StringBlockParser.Parse(byte[])` for reuse with MOGN/MODN. |
| `ReadBoneWeights`, `ReadUV`, etc. | M2-specific – not required for WMO. | — |

## 2. WMO Group Loading (`WMOLoader.js`)

1. **Chunk loop logic** – For both root and group files the loader iterates **until end of buffer** reading `<id><size><data>` where `id` is little-endian but stored reversed (**`REVM` ⇒ `MVER`** rule we already handle via `Array.Reverse`).
2. **Group files naming** – The root uses `getGroup(index)` to lazily load external groups: `basename_###.wmo` (3-digit) or fallback `_####` (4-digit). We already replicate this logic in `V17WmoFile.Load` when scanning the directory.
3. **Shared Geometry** – New v17 root contains `MOVV` (vertices float3) & `MOVB` (uint16 index triplets). Each group header (`MOGP`) references them via `FirstVertex/VertexCount` and `FirstIndex/IndexCount`. If these counts are zero the group holds its own `MOVT/MOVI`.
4. **Face Flags** – `MOPY` 2-byte records per triangle: lower byte = material ID, upper byte contains flags (collision, render, etc.). `WMORenderer` investigates `flag & 0x04` to skip collision-only surfaces when rendering. We need a `MOPYParser` v17 version returning `(materialId, flags)` list.
5. **Liquid** – `MLIQ` chunk parsing kept minimal: counts, corner, materialId, vertices[height+data], tiles array. Optional; can be skipped for initial OBJ export.
6. **Portals / Fog** – Optional (`MOPV`, `MOPT`, `MOPR`, `MFOG`) – defer to later phase.

## 3. Renderer Insights (`WMORenderer.js`)

* Groups are sorted into opaque vs translucent render lists based on `materials[matId].blendMode` and `flags`.
* Collision-only triangles (`flags & 0x04`) are ignored for visual export.
* Vertex normals are not stored; renderer derives them per-face if absent.

## 4. Exporter Behaviour (`WMOExporter.js`)

* Exports a single OBJ by iterating all groups, emitting `g Group_#` and `usemtl Material_##` before faces.
* UVs are skipped for collision surfaces.
* Materials file (`.mtl`) lists `map_Kd` pointing at extracted textures. Same as our existing `WmoObjExporter` workflow.

## 5. Immediate Implementation Tasks

1. **Parsers**
   * `MOPYParser.Parse(byte[]) -> List<(byte materialId, byte flags)>`.
   * `StringBlockParser` generic helper.
2. **Model classes**
   * `FaceFlags` list in `V17Group` to store parsed MOPY records.
3. **Group Loader**
   * `V17GroupLoader.Load(Stream, sharedVerts?, sharedTris?)` returning `V17Group`.
   * Handles per-group `MOVT/MOVI` fallback.
4. **OBJ Export Filtering**
   * Skip triangles where `(flags & 0x04) != 0` (collision-only) in visual export path.

These notes will guide Phase 2 development without overwhelming the PR with full feature scope.

---

_End of notes_

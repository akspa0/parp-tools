# WMO v17 Chunk Mapping (wow.export → WoWToolbox.Core.v2)

> Generated 2025-07-13 by Cascade – initial draft for Phase 1.

This table documents every chunk handler implemented in `docs/apps/wow.export/src/js/3D/loaders/WMOLoader.js`, its FourCC ID, purpose, and the planned equivalent C# data model/parser under `WoWToolbox.Core.v2.Foundation.WMO.V17`.

| FourCC | Handler in WMOLoader.js | Planned C# Struct / Parser | Notes |
|--------|------------------------|----------------------------|-------|
| MVER | 0x4D564552 | `ChunkMver` | Version (expect 17). 4 bytes. |
| MOHD | 0x4D4F4844 | `MOHDHeader` (reuse v14 model) | Root header counts & flags. |
| MOTX | 0x4D4F5458 | `MotxBlock` → reuse `MOTXParser` | Null-terminated string block of texture names. |
| MOMT | 0x4D4F4D54 | `MOMTMaterial[]` | 64-byte material definitions. |
| MOGN | 0x4D4F474E | `MognStringBlock` | Group name strings. |
| MOGI | 0x4D4F4749 | `MOGIEntry[]` | Per-group info inc. AABB, flags, name index. 32 bytes each. |
| GFID | 0x47464944 | `GfidList` | External group fileDataIDs (uint32[]). |
| MOPV | 0x4D4F5056 | `PortalVertex[]` | Float3 vertex list. |
| MOPT | 0x4D4F5054 | `PortalTriangle[]` | startVertex, count, plane eqn[4]. |
| MOPR | 0x4D4F5052 | `PortalReference[]` | portalIndex, groupIndex, side, filler. 8 bytes. |
| MFOG | 0x4D464F47 | `FogEntry[]` | 48-byte fog records. Optional for render. |
| MODS | 0x4D4F4453 | `DoodadSet[]` | 32-byte doodad set records. |
| MODI | 0x4D4F4449 | `uint[] FileDataIDs` | Doodad IDs. |
| MODN | 0x4D4F444E | `ModnStringBlock` | Doodad names (strings). |
| MODD | 0x4D4F4444 | `DoodadDef[]` | 40-byte doodad definitions. |
| MLIQ | 0x4D4C4951 | `MLIQChunk` | Liquid info & vertex/tile data. Optional. |
| MOCV | 0x4D4F4356 | `VertexColor[]` | Optional packed vertex colours (uint32 each). |
| MDAL | 0x4D44414C | `uint AmbientColor` | Optional ambient colour (uint32). |
| MOGP | 0x4D4F4750 | `MOGPGroupHeader` (reuse v14) | Group header followed by sub-chunks. |
| MOVV | 0x4D4F5656 | `Vector3[] Vertices` (shared) | New in v17; packed float3. |
| MOVB | 0x4D4F5642 | `ushort[] Indices` (shared) | New in v17; 3-uint16 triangles. |
| MOVT | 0x4D4F5654 | `Vector3[] Vertices` (legacy per-group) | Retained for backward compatibility. |
| MOVI | 0x4D4F5649 | `ushort[] Indices` (legacy per-group) | Retained. |
| MOPY | 0x4D4F5059 | `byte[] FaceFlags` | 2-byte face material/flags per triangle. |

### Pending Investigation
* `MLIQ`, `MOCV`, `MDAL` structs need confirmation of exact layout.
* Collision batches (`numBatchesA/B/C` in `MOGP`) require dedicated structs in later phases.

---

## Next Actions
1. Commit this mapping document (done).
2. Stub C# structs under `Foundation.WMO.V17.Chunks` matching the table above (fields only, no logic).
3. Extend `V17ChunkReader` consumer helpers to deserialize these structs where required.

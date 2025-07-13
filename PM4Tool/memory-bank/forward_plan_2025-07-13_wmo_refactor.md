# WMO v17 Refactor Plan (based on wow.export)

Date: 2025-07-13
Author: Cascade
Status: DRAFT – living document until implementation is complete

## Objective
Port the proven WMO v17 implementation from *wow.export* into **WoWToolbox.Core.v2** so the library can parse, analyse and export any v17 WMO with parity to the reference tool.

## High-level Milestones
1. **Analysis & Mapping**
   • Fully read `docs\apps\wow.export\src\js\3D\loaders\WMOLoader.js` plus helper files.  
   • Document chunk handlers, data structures, and control flow.

2. **Core Library Port (Foundation.WMO.V17)**
   1. Chunk Reader – keep current reader but verify against wow.export (4-byte aligned, reversed FourCC).
   2. Root Loader (`V17WmoRoot`) – implement handlers for every root-level chunk:
      * `MVER`, `MOHD`, `MOTX`, `MOMT`, `MOGI`, `MOGN`, `MFOG`, `MOSB`, etc.
   3. Group Loader (`V17WmoGroup`) – implement handlers for every group-level chunk:
      * `MVER`, `MOGP`, `MOVV/MOVB` (shared) or `MOVT/MOVI` (legacy), `MOBA`, `MOPY`, `MOVI` collision indices, etc.
   4. Shared Geometry – slice vertices/indices from root `MOVV/MOVB` via `FirstVertex/VertexCount` & `FirstIndex/IndexCount`.
   5. Flags & Materials – port material/shader handling so render vs collision surfaces are distinguished.

3. **Integration with Tools**
   • Update `ChunkInspector`, `WmoObjExporter`, comparison logic to rely on new loaders.  
   • Stream group OBJs/CSVs using real geometry.

4. **Testing**
   • Create regression tests loading real WMO v17 samples in `test_data\wmo_v17`.  
   • Verify vertex/face counts match wow.export outputs (SHA-1 on exported OBJ).

5. **Deprecation & Cleanup**
   • Remove interim `V17WmoFile` hacks after parity is reached.  
   • Update documentation & memory-bank activeContext.md.

## Detailed Task Breakdown
### Phase 1 – Mapping
- [ ] Annotate every chunk handler in WMOLoader.js with equivalent C# data model.
- [ ] Create mapping table FourCC → C# parser.

### Phase 2 – Library Port
- [ ] Create `Foundation.WMO.V17.Chunks` namespace with strongly-typed structs for each chunk.
- [ ] Implement `V17ChunkReader` (done) – verify padding.
- [ ] Implement `V17RootLoader.Load()`
- [ ] Implement `V17GroupLoader.Load()`
- [ ] Implement `MovbParser` (uint16 triplets → triangles)

### Phase 3 – Tools Update
- [ ] Modify `WmoObjExporter` to use new loaders.
- [ ] Expand `ChunkInspector --exportgroups` to iterate all `MSUR` surfaces and all WMO groups.
- [ ] Implement AABB + vertex-overlap matcher copied from wow.export renderer.

### Phase 4 – Tests & Validation
- [ ] Add XUnit integration tests loading sample WMO & asserting vertex/face counts.
- [ ] Cross-validate OBJ hashes against wow.export outputs.

### Phase 5 – Documentation & Clean-up
- [ ] Update `activeContext.md` with progress notes.
- [ ] Remove deprecated parser code.
- [ ] Final QA with large production files.

## Risks & Mitigations
| Risk | Mitigation |
|------|-----------|
| Feature creep from wow.export (e.g., doodad sets, portals) | Scope only essential geometry first, add extras iteratively |
| Performance on large WMO | Stream parsing; avoid full in-memory structures where possible |
| Divergent coordinate systems | Validate coordinates with shared sample exports |

---
*End of plan – will evolve as implementation proceeds.*

# PM4 Exporter Complete Refactor Plan

## Executive Summary
The PM4 exporter requires a complete ground-up rewrite due to severe corruption and fundamental architectural issues. This plan addresses the critical discoveries from recent analysis and implements the correct object grouping strategy based on MSLK.ParentIndex_0x04.

## Critical Discoveries
1. **Object Identification**: Individual PM4 objects are identified by **MSUR.Index_0x01** (surface index), which maps to **MSLK.Unknown_0x00**; geometry layers are clustered by **MSUR.Unknown_0x10**.
2. **Cross-tile References**: ~64% vertex data loss due to missing adjacent tile vertices
3. **Memory Bank**: Severe file corruption requires complete rewrite
4. **Legacy Parity**: Need byte-for-byte identical output to legacy exporters
5. **Reference PoC Commit**: Working exporter logic is in commit `2e3f9ebfac5cb7dc34cfeee829036a3f62ebccde` – use this as ground truth for porting.

## Observed Implementation Drift (July 26 2025)
Despite the above discoveries being reiterated multiple times, refactors have repeatedly regressed to prior, incorrect grouping logic (e.g., SurfaceKey-based grouping).  This indicates:
- Incomplete propagation of Critical Discoveries into code changes.
- Legacy files (e.g., `NewPm4Exporter.cs`) being patched rather than replaced, re-introducing outdated assumptions.
- Lack of automated regression tests that fail when grouping deviates from **MSUR.Index_0x01 → MSLK.Unknown_0x00**.

**Action Items**
1. Freeze old exporter paths; create a clean `Pm4PerObjectExporter` that exclusively uses the validated keys.
2. Add unit test: given sample PM4, assert assembled object count equals proof-of-concept.
3. Add SHA regression test against legacy OBJ output.
4. Code reviews must cross-check every change against this plan before merge.

## Architecture Overview

### Core Components
1. **Pm4SceneLoader** - Unified scene loading with cross-tile vertex resolution
2. **MsurObjectAssembler** - Groups by MSUR.IndexCount for complete objects
3. **MslkHierarchyBuilder** - Builds object relationships via ParentIndex_0x04
4. **LegacyObjExporter** - Ensures identical output format to legacy tools
5. **CrossTileVertexResolver** - Handles missing vertex data from adjacent tiles

### Data Flow
```
PM4File → SceneLoader → ObjectAssembler → VertexResolver → ObjExporter
```

## Implementation Strategy

### Phase 1: Foundation (Clean Architecture)
- [ ] Create new Pm4Exporter class from scratch
- [ ] Implement reflection-based MSLK field access for ParentIndex_0x04
- [ ] Add robust error handling for dynamic property access
- [ ] Establish clean separation of concerns

### Phase 2: Object Grouping
- [ ] Implement grouping strategy:
  - Cluster surfaces by **MSUR.Unknown_0x10** (geometry layer)
  - Within each cluster, group by **MSUR.Index_0x01** and resolve to **MSLK.Unknown_0x00** for object assembly
- [ ] Handle container nodes (MspiFirstIndex = -1) appropriately
- [ ] Build hierarchical object relationships
- [ ] Ensure proper object scale (38K-654K triangles per building)

### Phase 3: Vertex Resolution
- [ ] Implement cross-tile vertex loading system
- [ ] Handle high/low pair encoding for 32-bit indices
- [ ] Resolve missing vertex data from adjacent tiles
- [ ] Ensure complete geometry without fragmentation

### Phase 4: Legacy Compatibility
- [ ] Port exact legacy OBJ export format
- [ ] Implement coordinate system fixes (X-axis inversion)
- [ ] Ensure byte-for-byte output parity
- [ ] Add comprehensive regression tests

### Phase 5: Collision & Server Integration (MSCN)
- [ ] Implement `--collision` flag: export MSCN triangles only.
- [ ] Write minimal OBJ/GLB or binary mesh for collision.
- [ ] Build simple BVH/uniform grid accelerator for queries.
- [ ] Prototype Recast/Detour (or SharpNav) nav-mesh generation using walkable MSCN flags.
- [ ] Document MSCN field semantics and flipping rules.

- [ ] Port exact legacy OBJ export format
- [ ] Implement coordinate system fixes (X-axis inversion)
- [ ] Ensure byte-for-byte output parity
- [ ] Add comprehensive regression tests

## Technical Specifications

### MSLK Entry Processing
```csharp
// Key fields to extract dynamically
- ParentIndex_0x04: uint (object grouping key)
- MspiFirstIndex: int (geometry start)
- MspiIndexCount: int (geometry count)
- ReferenceIndex: uint (cross-references)
```

### Object Assembly Logic
1. **Primary Grouping**: Group MSLK entries by ParentIndex_0x04
2. **Geometry Collection**: Aggregate triangles from all child entries
3. **Vertex Mapping**: Map indices to actual MSVT vertices
4. **Export Format**: Legacy OBJ format with proper coordinate transformation

### Error Handling
- Robust reflection-based property access
- Graceful handling of missing chunks
- Comprehensive logging for debugging
- Validation of vertex index bounds

## Testing Strategy

### Regression Tests
1. **SHA Comparison**: Verify byte-for-byte output parity with legacy
2. **Object Count**: Validate correct number of building objects
3. **Triangle Count**: Ensure 38K-654K triangles per building
4. **Vertex Integrity**: Check for missing/out-of-bounds indices

### Integration Tests
1. **Cross-tile Loading**: Test with multi-tile PM4 files
2. **Memory Usage**: Validate reasonable memory consumption
3. **Performance**: Benchmark against legacy export times
4. **Edge Cases**: Handle empty objects, missing data

## Migration Path

### Step 1: Clean Implementation
- Create completely new exporter class
- Implement core grouping logic
- Add basic vertex resolution

### Step 2: Legacy Integration
- Port exact legacy export format
- Add PM4File.LegacyAliases for compatibility
- Implement coordinate transformation

### Step 3: Testing & Validation
- Run comprehensive regression tests
- Compare outputs with legacy tools
- Fix any discrepancies

### Step 4: Deployment
- Replace old exporter completely
- Update CLI commands
- Update documentation

## Risk Mitigation

### Data Loss Prevention
- Comprehensive vertex index validation
- Cross-tile reference resolution
- Missing vertex detection and handling

### Compatibility Assurance
- Legacy output format preservation
- Coordinate system consistency
- Error handling for edge cases

## Success Criteria
- [ ] Build passes with zero errors
- [ ] All PM4 exports match legacy outputs (SHA256)
- [ ] Object grouping produces meaningful building-scale objects
- [ ] Cross-tile vertex references resolved correctly
- [ ] Comprehensive test coverage ≥80%
- [ ] Documentation updated with new architecture

## Timeline
- **Phase 1**: 2-3 days (clean implementation)
- **Phase 2**: 1-2 days (legacy integration)
- **Phase 3**: 2-3 days (testing & validation)
- **Phase 4**: 1 day (deployment & documentation)

Total estimated time: 5-7 days for complete refactor

## Current Status
- [x] Memory bank plan created
- [ ] Phase 1: Clean implementation pending
- [ ] Phase 2: Legacy integration pending
- [ ] Phase 3: Testing & validation pending
- [ ] Phase 4: Deployment pending

## Next Steps
1. Create new Pm4Exporter class from scratch
2. Implement reflection-based MSLK field access
3. Add MSLK.ParentIndex_0x04 grouping logic
4. Build cross-tile vertex resolution system
5. Port legacy OBJ export format
6. Run comprehensive regression tests
7. Update documentation and CLI commands

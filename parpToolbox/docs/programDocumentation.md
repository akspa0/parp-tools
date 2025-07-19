# Program Documentation - parpToolbox

## PM4/PD4 Implementation Status

### Current Implementation Challenges

#### PM4 Grouping Logic
**Current Issue**: PM4 export produces ~825 groups instead of expected 10-20
**Root Cause**: Complex multi-object relationships not properly understood
**Solution Path**: Port legacy `MsurObjectExporter` grouping algorithm

#### Multi-Object Complexity
- Single PM4 contains data for multiple distinct objects
- Objects may share vertices, indices, or surface definitions
- Grouping algorithm must correctly separate individual objects
- Link data (MSLK) provides crucial object separation information

#### Mathematical Relationships
- PM4 objects have complex spatial relationships
- Coordinate systems may vary between objects within same file
- Surface definitions may reference shared geometry pools
- Property lists (MPRL) define object-specific transformations

### PD4 Implementation Status
**Current Implementation**: ✅ **Stable**
- PD4 export pipeline functional and validated
- Vertex-only OBJ export prevents viewer crashes
- Coordinate transformations implemented
- Surface grouping logic operational

**Export Capabilities**:
- Point cloud OBJ generation (vertices only)
- Proper coordinate transformation for MSVT
- Surface-based grouping via MSUR
- Timestamped output via ProjectOutput utility

### PM4 Implementation Status
**Current Implementation**: ⚠️ **Needs Work**
- Basic PM4 loading functional
- Export produces incorrect group counts (825 vs 10-20 expected)
- Grouping algorithm requires legacy port from `MsurObjectExporter`
- Coordinate transformations implemented but grouping logic flawed

**Known Issues**:
1. **Grouping Logic**: Current algorithm doesn't properly separate objects
2. **Surface Mapping**: MSUR → MSLK relationships not correctly resolved
3. **Index Validation**: Face index remapping may corrupt geometry
4. **Output Validation**: Generated OBJ files need geometry validation

**Next Steps**:
1. Port legacy `MsurObjectExporter` grouping routine
2. Implement proper surface range matching via MSLK `ReferenceIndex`
3. Validate output counts against real PM4 data
4. Ensure PD4 export stability during PM4 fixes

## Export Considerations

### PM4 Grouping Strategy
- Use MSLK `ReferenceIndex` to cluster related surfaces
- Match surface ranges to separate individual objects
- Validate group counts against expected object counts
- Preserve object spatial relationships

### Coordinate Systems
- Apply MSVT coordinate transformations consistently
- Handle multi-object coordinate space variations
- Maintain precision for server-side navigation accuracy
- Validate transformed coordinates against expected ranges

### Output Format
- Generate separate OBJ files per object group
- Use descriptive naming based on object properties
- Include material definitions where applicable
- Route all output through ProjectOutput utility

## Research Notes

**Legacy Reference**: `MsurObjectExporter` in legacy codebase contains authoritative grouping logic
**Test Data**: Real PM4 files should produce 10-20 object groups, not 825
**Validation**: Compare output counts with legacy implementation results
**Stability**: Ensure PM4 changes don't break PD4 export pipeline

## Development History

### (2025-07-18 15:00) - Documentation & Format Understanding
- **New Documentation Created**: Fresh PM4 and PD4 format documentation in `docs/formats/`
- **Format Relationships Clarified**:
  - **PM4**: Phased model descriptors (complex, pre-2016, multi-object)
  - **PD4**: Individual object data with full precision (post-2016 split)
  - **WMO**: Compressed geometry, mathematically different from PM4/PD4
  - **No direct correlation** between WMO and PD4 objects confirmed

### (2025-07-14 22:51) - PM4 Export Issues
- PM4 export produced **825** groups (expected ~10–20). Grouping algorithm still incorrect.
- MSUR chunk loader rewritten to 32-byte authoritative spec; alignment confirmed.
- **Root Cause**: Complex multi-object relationships in PM4 not properly understood
- **Solution Path**: Port legacy `MsurObjectExporter` grouping algorithm
- **Critical**: Ensure PD4 export stability while fixing PM4 logic

### WMO Export Status
WMO → OBJ export has been fully validated: façade planes are correctly filtered by default, and group/file naming matches in-game names. The command-line pipeline works reliably when arguments are passed via an executable build; the `dotnet run --` quirk is still under investigation.

### Next Priority Tasks
1. **Port `MsurObjectExporter` grouping routine** from legacy codebase
2. **Implement proper surface range matching** via MSLK `ReferenceIndex`
3. **Validate PM4 group counts** against real data (target: 10-20 groups)
4. **Maintain PD4 export stability** during PM4 fixes
5. **Update memory bank** with current format understanding
